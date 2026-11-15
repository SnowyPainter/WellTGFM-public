"""Read-only checkpoint evaluation and six-domain relational-pattern diagnostics."""
import itertools,json,hashlib,time,argparse
from pathlib import Path
from collections import defaultdict
from dataclasses import replace
import numpy as np
import torch,yaml
from sklearn.metrics import average_precision_score
from experiments.common.data import Events
from experiments.common.runtime import atomic_json,gpu_guard
from experiments.transfer.cross_representation import build
from experiments.common import source_patterns as patterns
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'experiments/transfer/outputs/cross_representation_two_tkg_v1'
OUT=BASE/'analysis/source_diagnostics'
NAMES=['wikipedia','reddit','mooc','lastfm','ICEWS14','GDELT']
SEEDS=[7,17,37]


def negatives(d,seed):
    if d.metadata['family']=='ctdg':return d.negatives(seed+100000)
    rng=np.random.default_rng(seed+100000)
    train=d.indices(0);vocab=np.unique(np.r_[d.src[train],d.dst[train]])
    tail=defaultdict(set);head=defaultdict(set)
    for u,r,v,t in zip(d.src,d.relation,d.dst,d.time):
        tail[int(u),int(r),float(t)].add(int(v));head[int(v),int(r),float(t)].add(int(u))
    out=np.zeros((len(d.src),2),dtype=np.int64)
    def draw(block):
        for _ in range(100):
            x=int(rng.choice(vocab))
            if x not in block:return x
        allowed=np.array([x for x in vocab if x not in block]);assert len(allowed)
        return int(rng.choice(allowed))
    for i in np.r_[d.indices(0),d.indices(1)]:
        u,r,v,t=int(d.src[i]),int(d.relation[i]),int(d.dst[i]),float(d.time[i])
        out[i]=draw(tail[u,r,t]),draw(head[v,r,t])
    return out


def pattern_analysis(data,negs):
    patterns.DATASETS=tuple(NAMES)
    tables_all={};rows=[]
    for seed in SEEDS:
        tables={}
        for name,d in data.items():
            neg=negs[seed,name]
            if neg.ndim==1:records=patterns.features(d,neg,2048)
            else:
                records=patterns.features(d,neg[:,0],1024)
                records+=patterns.features(replace(d,src=d.dst,dst=d.src),neg[:,1],1024)
            tables[name]=patterns.aggregate(records)
        tables_all[str(seed)]=tables;rows+=patterns.compare(tables,seed)
    groups=defaultdict(list)
    for row in rows:groups[row['source_a'],row['source_b'],row['view'],row['cell']].append(row)
    robust=[]
    for key,rs in groups.items():
        if len(rs)==3 and all(r['train_discovered'] and r.get('val_strong_opposition',False) for r in rs) and len({r['a_train_p']>r['b_train_p'] for r in rs})==1:
            robust.append(dict(source_a=key[0],source_b=key[1],view=key[2],cell=key[3],rows=rs,mean_ce_excess=float(np.mean([r['ce_excess'] for r in rs]))))
    robust.sort(key=lambda x:-x['mean_ce_excess'])
    atomic_json(OUT/'pattern_counts.json',tables_all)
    atomic_json(OUT/'pattern_comparisons.json',rows)
    atomic_json(OUT/'patterns.json',dict(train_candidates=sum(r['train_discovered'] for r in rows),validation_confirmed=sum(r['train_discovered'] and r.get('val_strong_opposition',False) for r in rows),robust=robust))


@torch.no_grad()
def evaluate_source(model,d,neg,cfg):
    model.bind(d);model.eval();model.reset();scores=[]
    batch=cfg['tkg_batch_size'] if neg.ndim==2 else cfg['batch_size']
    for ids in d.batches(d.indices(1),batch):
        model.begin_batch(ids)
        for start in range(0,len(ids),cfg['microbatch']):
            part=ids[start:start+cfg['microbatch']]
            scores.append(model.score(part,neg[part]).float().cpu().numpy())
    scores=np.concatenate(scores);labels=np.zeros_like(scores);labels[:,0]=1
    return dict(ap=float(average_precision_score(labels.ravel(),scores.ravel())),queries=len(scores)),scores


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--max-sources',type=int,choices=[2,5],default=2);args=parser.parse_args()
    progress_name='progress_all.json' if args.max_sources==5 else 'progress.json'
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    data={n:Events.load(ROOT/'data/benchmarks/events32768','ctdg' if n in NAMES[:4] else 'tkg',n) for n in NAMES}
    negs={(s,n):negatives(d,s) for s in SEEDS for n,d in data.items()}
    if args.max_sources==2 or not (OUT/'patterns.json').exists():pattern_analysis(data,negs)
    print('Data and evaluation candidates ready',flush=True)
    # All singleton/pair configurations in every eligible target context.
    jobs=[]
    for target in NAMES[:4]:
        pool=[n for n in NAMES if n!=target]
        for seed in SEEDS:
            for k in range(1,args.max_sources+1):
                for sources in itertools.combinations(pool,k):jobs.append((target,seed,list(sources)))
    records=[]
    with gpu_guard('cuda:0',0):
        for i,(target,seed,sources) in enumerate(jobs):
            key='+'.join(sources);folder=OUT/target/key/f'seed{seed}';result=folder/'result.json'
            ck=BASE/target/key/f'seed{seed}/pretrain/resume.pt'
            sha=hashlib.sha256(ck.read_bytes()).hexdigest()
            if result.exists():
                rec=json.loads(result.read_text());assert rec['checkpoint_sha256']==sha
            else:
                cfg=yaml.safe_load((BASE/(target+'.yaml')).read_text());model=build(cfg,data,seed)
                state=torch.load(ck,map_location='cpu',weights_only=False);assert state['step']==60
                model.load_state_dict(state['model']);del state
                rec=dict(target_context=target,seed=seed,sources=sources,checkpoint=str(ck.relative_to(ROOT)),checkpoint_sha256=sha,metrics={})
                folder.mkdir(parents=True,exist_ok=True)
                for name in sources:
                    metrics,scores=evaluate_source(model,data[name],negs[seed,name],cfg)
                    rec['metrics'][name]=metrics;np.savez_compressed(folder/(name+'_scores.npz'),scores=scores)
                assert hashlib.sha256(ck.read_bytes()).hexdigest()==sha
                atomic_json(result,rec);del model
            records.append(rec);atomic_json(OUT/progress_name,dict(completed=i+1,total=len(jobs)))
            print(f'{i+1}/{len(jobs)} {target} {seed} {key}',flush=True)
    atomic_json(OUT/('all_source_evaluations.json' if args.max_sources==5 else 'source_evaluations.json'),records)
    lookup={(r['target_context'],r['seed'],tuple(r['sources'])):r for r in records}
    pairs=[]
    for a,b in itertools.combinations(NAMES,2):
        rs=[]
        for target in NAMES[:4]:
            if target in (a,b):continue
            for seed in SEEDS:
                joint=lookup[target,seed,(a,b)]
                rs.append(dict(target_context=target,seed=seed,a_single=lookup[target,seed,(a,)]['metrics'][a]['ap'],b_single=lookup[target,seed,(b,)]['metrics'][b]['ap'],a_joint=joint['metrics'][a]['ap'],b_joint=joint['metrics'][b]['ap']))
        pairs.append(dict(source_a=a,source_b=b,rows=rs))
    atomic_json(OUT/'source_pairs.json',pairs)
    atomic_json(OUT/progress_name,dict(status='complete',checkpoints=len(jobs),source_evaluations=sum(len(r['sources']) for r in records)))

if __name__=='__main__':main()
