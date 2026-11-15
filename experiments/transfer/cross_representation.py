"""Learning Across shared encoder: TKG sources, CTDG target link prediction only."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import time
import numpy as np
import torch
import yaml
from experiments.common import protocol
from experiments.common.data import Events
from experiments.common.engine import train_batch,save_predictions
from experiments.common.runtime import atomic_json,atomic_torch,seed_all,evaluate,gpu_guard,capture_rng,restore_rng
from models.ours.joint import CrossRepresentationPredictor


def typed_negatives(data,seed):
    """Train-only bidirectional entity corruption, timestamp filtered, train vocabulary."""
    rng=np.random.default_rng(seed);ids=data.indices(0)
    vocab=np.unique(np.r_[data.src[ids],data.dst[ids]])
    tails={};heads={}
    for i in ids:
        u,r,v,t=int(data.src[i]),int(data.relation[i]),int(data.dst[i]),float(data.time[i])
        tails.setdefault((u,r,t),set()).add(v);heads.setdefault((v,r,t),set()).add(u)
    out=np.zeros((len(data.src),2),np.int64)
    def draw(blocked):
        for _ in range(100):
            v=int(rng.choice(vocab))
            if v not in blocked:return v
        allowed=np.array([v for v in vocab if v not in blocked])
        if not len(allowed):raise ValueError('No legal negative')
        return int(rng.choice(allowed))
    for i in ids:
        u,r,v,t=int(data.src[i]),int(data.relation[i]),int(data.dst[i]),float(data.time[i])
        out[i]=draw(tails[u,r,t]),draw(heads[v,r,t])
    return out


def build(cfg,data,seed):
    seed_all(seed)
    return CrossRepresentationPredictor(data,cfg['target'],cfg['history_length'],cfg['device'],seed)

def optimizer(model,cfg):
    return torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])

def pretrain(cfg,data,sources,seed,out,loss_fn=None):
    result=out/'result.json'
    if result.exists():return torch.load(out/'final.pt',map_location='cpu',weights_only=False)['shared']
    model=build(cfg,data,seed);opt=optimizer(model,cfg)
    batches={n:list(data[n].batches(data[n].indices(0),cfg['tkg_batch_size'] if data[n].metadata['family']=='tkg' else cfg['batch_size'])) for n in sources}
    start=0;seen={n:0 for n in sources};queries={n:0 for n in sources};history=[]
    if (out/'resume.pt').exists():
        ck=torch.load(out/'resume.pt',map_location=cfg['device'],weights_only=False)
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);restore_rng(ck['rng'],cfg['device'])
        start=ck['step'];seen=ck['events_seen'];queries=ck['queries_seen'];history=ck['history']
    negatives={};beg=time.monotonic()
    for step in range(start,cfg['pretrain_updates']):
        name=sources[step%len(sources)];local=step//len(sources);cycle=local//len(batches[name])
        if (name,cycle) not in negatives:
            d=data[name];negatives[name,cycle]=typed_negatives(d,seed+cycle) if d.metadata['family']=='tkg' else d.negatives(seed+cycle)
        ids=batches[name][local%len(batches[name])];model.bind(data[name])
        loss=train_batch(model,data[name],ids,negatives[name,cycle],cfg,opt,1.,loss_fn=loss_fn)
        seen[name]+=len(ids);queries[name]+=len(ids)*(2 if data[name].metadata['family']=='tkg' else 1)
        history.append(dict(step=step+1,source=name,loss=loss))
        if (step+1)%12==0 or step+1==cfg['pretrain_updates']:
            atomic_torch(out/'resume.pt',dict(step=step+1,model=model.state_dict(),optimizer=opt.state_dict(),rng=capture_rng(cfg['device']),events_seen=seen,queries_seen=queries,history=history))
            print(json.dumps(dict(stage='pretrain',sources=sources,seed=seed,step=step+1,seconds=time.monotonic()-beg,loss=loss)),flush=True)
    state=model.shared_state();atomic_torch(out/'final.pt',dict(shared=state,seed=seed,sources=sources,steps=cfg['pretrain_updates']))
    atomic_json(result,dict(status='complete',seed=seed,sources=sources,updates=cfg['pretrain_updates'],events_seen=seen,queries_seen=queries,history=history))
    return state


def adapt(cfg,data,shared,seed,out):
    if (out/'result.json').exists():return
    model=build(cfg,data,seed)
    if shared is not None:model.load_shared_state(shared)
    d=data[cfg['target']];model.bind(d);opt=optimizer(model,cfg)
    fixed=d.negatives(seed+100000);best=-float('inf');history=[];start=0
    if (out/'resume.pt').exists():
        ck=torch.load(out/'resume.pt',map_location=cfg['device'],weights_only=False)
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);restore_rng(ck['rng'],cfg['device'])
        start=ck['epoch'];history=ck['history'];best=ck['best']
    for epoch in range(start,cfg['epochs']):
        neg=d.negatives(seed+epoch);loss=[];beg=time.monotonic()
        for ids in d.batches(d.indices(0),cfg['batch_size']):loss.append(train_batch(model,d,ids,neg,cfg,opt,1.))
        metrics,scores=evaluate(model,d,1,fixed,cfg)
        history.append(dict(epoch=epoch+1,validation=metrics,train_loss=float(np.mean(loss))))
        if metrics['ap']>best:
            best=metrics['ap'];atomic_torch(out/'best.pt',dict(model=model.state_dict(),epoch=epoch+1))
            atomic_json(out/'best_metrics.json',metrics);save_predictions(out/'validation_predictions.npz',d,1,fixed,scores)
        atomic_torch(out/'resume.pt',dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=capture_rng(cfg['device']),epoch=epoch+1,history=history,best=best))
        print(json.dumps(dict(stage='target',seed=seed,path=str(out),epoch=epoch+1,seconds=time.monotonic()-beg,ap=metrics['ap'])),flush=True)
    atomic_json(out/'result.json',dict(status='complete',metrics=json.loads((out/'best_metrics.json').read_text()),seed=seed,history=history,split='validation'))


def source_conditions(cfg):
    if cfg.get('all_source_subsets'):
        pool=cfg['sources']
        return [[]]+[list(x) for k in range(1,len(pool)+1) for x in itertools.combinations(pool,k)]
    return [[],[cfg['sources'][0]],[cfg['sources'][1]],cfg['sources']]

def report(cfg,root):
    rows=[];conditions=['+'.join(s) if s else 'target_only' for s in source_conditions(cfg)]
    for seed in cfg['seeds']:
        row=dict(seed=seed)
        for c in conditions:
            p=root/c/f'seed{seed}/target/result.json'
            if p.exists():row[c]=json.loads(p.read_text())['metrics']['ap']
        rows.append(row)
    atomic_json(root/'summary.json',dict(rows=rows,conditions=conditions,split='validation'))
    lines=[f"# CTDG + TKG 공동 사전학습 → {cfg['target']}",'',
           '동일 TGAT + TKG 입력부 구조, 총 사전학습 60회, Target 적응 5 epoch. 값은 Validation AP.','',
           '| 반복(난수 시드) | '+' | '.join(conditions)+' |','|---|'+'---:|'*len(conditions)]
    for r in rows:lines.append('| '+str(r['seed'])+' | '+' | '.join(f'{r[c]:.4f}' if c in r else '진행 전' for c in conditions)+' |')
    if all(all(c in r for c in conditions) for r in rows):
        lines.append('| 평균 ± 표준편차 | '+' | '.join(f"{np.mean([r[c] for r in rows]):.4f} ± {np.std([r[c] for r in rows],ddof=1):.4f}" for c in conditions)+' |')
    lines+=['','CTDG와 TKG는 실제로 같은 history encoder 파라미터를 갱신한다. TKG 관계·방향 임베딩과 event projection, Domain별 head는 private이다. Target 적응에는 공유 파라미터만 전달하며 Target 인터페이스는 동일 난수로 새로 초기화한다.',
            'TKG는 학습 Source로만 사용한다. Head/tail corruption으로 사전학습하며, 최종 평가는 지정된 CTDG Target의 Link Prediction이다. Test는 평가하지 않는다.',
            'ICEWS14는 동일 Timestamp 보존으로 32,761 events(상한 32,768)이다. 총 업데이트는 동일하지만 TKG fact 하나에 두 방향 query가 있으므로 TKG nominal batch는 1,024 facts, CTDG는 2,048 events다. Timestamp 그룹으로 실제 batch 크기는 늘어날 수 있으며 노출량과 query 수를 기록한다.',
            '기존 TGAT 결과와 구조가 다르므로 이 표 내부의 새 Target-only/CTDG-only 기준선을 사용한다. 단일 Source에서 60회, 공동에서는 총 60회를 Source별로 균등 배분하며 나머지는 고정 순서로 배정한다.']
    (root/'README.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=Path(__file__).with_name('cross_representation.yaml'));p.add_argument('--execute',action='store_true');args=p.parse_args()
    cfg=yaml.safe_load(args.config.read_text());root=protocol.ROOT/cfg['output_root']
    data={n:Events.load(protocol.ROOT/cfg['data_root'],cfg['families'][n],n) for n in cfg['datasets']}
    for d in data.values():
        if d.metadata['max_events']!=cfg['max_events']:raise ValueError('Budget mismatch')
    conditions=source_conditions(cfg)
    jobs=[dict(seed=seed,sources=s) for seed in cfg['seeds'] for s in conditions]
    stages=len(cfg['seeds'])*sum(2 if s else 1 for s in conditions)
    plan=dict(config=cfg,jobs=jobs,stages=stages,actual_events={n:len(d.src) for n,d in data.items()})
    if not args.execute:print(json.dumps(plan,indent=2));return
    protected=protocol.ROOT/'experiments/transfer/outputs/v1'
    if root.resolve()==protected.resolve() or root.resolve().is_relative_to(protected.resolve()):
        raise ValueError('Separate cross-representation output required')
    root.mkdir(parents=True,exist_ok=True)
    legacy_paths=[p for name in ('target_only','mooc','reddit+mooc') for p in (protected/'uniform/lastfm'/name).rglob('best.pt')]
    legacy={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in legacy_paths}
    integrity=root/'legacy_integrity.json'
    if integrity.exists() and json.loads(integrity.read_text())['sha256']!=legacy:raise ValueError('Legacy checkpoints changed')
    atomic_json(integrity,dict(status='before',sha256=legacy))
    provenance=dict(plan=plan,implementation=protocol.code_hashes(),data={n:d.metadata for n,d in data.items()})
    manifest=root/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())!=provenance:raise ValueError('Code/config/data changed; use new output path')
    atomic_json(manifest,provenance);torch.set_num_threads(cfg['cpu_threads']);report(cfg,root)
    with gpu_guard(cfg['device'],0):
        for i,job in enumerate(jobs):
            seed=job['seed'];sources=job['sources'];out=root/('+'.join(sources) if sources else 'target_only')/f'seed{seed}'
            atomic_json(root/'progress.json',dict(status='running',condition=i+1,total=len(jobs),**job))
            shared=None
            if sources:
                (out/'pretrain').mkdir(parents=True,exist_ok=True);shared=pretrain(cfg,data,sources,seed,out/'pretrain')
            (out/'target').mkdir(parents=True,exist_ok=True);adapt(cfg,data,shared,seed,out/'target');report(cfg,root)
        after={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in legacy}
        if after!=legacy:raise RuntimeError('Legacy checkpoints changed')
        atomic_json(integrity,dict(status='verified_unchanged',sha256=legacy))
        atomic_json(root/'progress.json',dict(status='complete',conditions=len(jobs),stages=stages))
    print('Cross-representation suite complete.',flush=True)

if __name__=='__main__':main()
