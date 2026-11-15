"""Outcome-independent experiment definitions and immutable source checkpoints."""
import itertools
from pathlib import Path

from experiments.transfer_validation.core import (ROOT, HERE, BASE, OLD, TARGETS, SEEDS,
    POOL, read, canonical_hash, digest, DEFAULT)

METHODS = ['finetune', 'lp_ft', 'refine', 'tfra', 'joint', 'batch']
MAIN = METHODS[:4]
OUT = HERE/'outputs/v1'


def case(target, sources=(), preseed=7, profile='original', backbone='tgat', source_budget=None):
    return dict(target=target, sources=list(sources), pretrain_seed=preseed if sources else None,
                profile=profile, backbone=backbone, source_budget=source_budget)


def source_checkpoint(c, out=OUT):
    if not c['sources']:
        return None
    if c['profile']=='original' and c['backbone']=='tgat' and c['source_budget'] is None:
        ordered=[n for n in POOL if n in c['sources']]
        p=OLD/c['target']/'+'.join(ordered)/f"seed{c['pretrain_seed']}"/'pretrain/final.pt'
        if not p.exists():
            raise FileNotFoundError(p)
        return p
    key=canonical_hash(c)[:20]
    return out/'pretraining'/key/'final.pt'


def make_job(cohort, c, method, seed, config=None, out=OUT, **extra):
    ck=source_checkpoint(c,out)
    job=dict(cohort=cohort, **c, method=method, adaptation_seed=seed,
             config={**DEFAULT, **(config or {})}, checkpoint=str(ck.relative_to(ROOT)) if ck else None, **extra)
    # All checkpoint hashes are recorded at execution. Keeping them out of the job
    # definition makes a planned job identical before/after its checkpoint exists.
    identity={k:v for k,v in job.items() if k!='checkpoint_sha256'}
    job['id']=cohort+'__'+canonical_hash(identity)[:20]
    return job


def panel_cases():
    seen=set(); cases=[]
    for j in read(BASE/'inputs/plan.json')['jobs']:
        key=(j['target'],tuple(sorted(j['sources'])))
        if key not in seen:
            seen.add(key);cases.append(case(j['target'],j['sources']))
    assert len(cases)==18
    return cases


def small_cases():
    # Same outcome-independent smallest-composition rule as the head intervention.
    p=panel_cases(); cases=[]
    for t in TARGETS:
        choices=[c for c in p if c['target']==t and c['sources']]
        cases.extend([min(choices,key=lambda c:(len(c['sources']),'+'.join(sorted(c['sources'])))),case(t)])
    return cases


def grid(method):
    configs=[]
    for lr in [1e-5,1e-4,1e-3]:
        for reg in ([1e-4,1e-3,1e-2] if method in {'tfra','joint','batch'} else [0.,.1,.2]):
            cfg=dict(learning_rate=lr,epochs=20,include_epoch0=True)
            cfg['ridge_lambda' if method in {'tfra','joint','batch'} else 'dropout']=reg
            configs.append(cfg)
    return configs


def tuning_jobs(out=OUT):
    return [make_job('tuning',c,m,7,cfg,out,trial=i)
            for i in range(9) for m in METHODS for cfg in [grid(m)[i]] for c in small_cases()]


def mechanism_jobs(out=OUT):
    methods=['tfra','joint','joint_init','full_step','batch','frozen','initial_fixed']
    selected={(c['target'],tuple(c['sources'])) for c in small_cases() if c['sources']}
    return [make_job('mechanism',c,m,a,dict(epochs=5),out,
             diagnostics=(m in {'tfra','joint','joint_init'} and (c['target'],tuple(c['sources'])) in selected))
            for a in SEEDS for c in panel_cases() for m in methods]


def head_jobs(out=OUT):
    return [make_job('head_components',c,m,a,dict(epochs=5),out,hidden_seed=h,final_seed=f)
            for a in SEEDS for h in SEEDS for f in SEEDS for c in small_cases() for m in ['finetune','tfra']]


def winner_config(winners,m,epochs=20):
    return {**winners[m]['config'], 'epochs':epochs}


def confirmation_jobs(winners,out=OUT):
    return [make_job('long_budget',c,m,a,winner_config(winners,m,50),out)
            for a in SEEDS for c in small_cases() for m in METHODS]


def factorial_jobs(winners,out=OUT):
    cases=[]
    for t in TARGETS:
        pool=[n for n in TARGETS if n!=t]
        compositions=[list(x) for k in [1,2,3] for x in itertools.combinations(pool,k)]+[pool+['ICEWS14','GDELT']]
        cases.extend(case(t,s,p) for s in compositions for p in SEEDS)
        cases.append(case(t))
    return [make_job('seed_factorial',c,m,a,winner_config(winners,m),out)
            for a in SEEDS for c in cases for m in MAIN]


def breadth_cases():
    cases=[]
    # New TFRA domains selected by name, before inspecting any adaptation outcomes.
    new=sorted(read(ROOT/'configs/mint_networks.json')['packages']['test'])[:2]
    assert new==['ADX','BAG']
    for t in TARGETS+new:
        profile='mint' if t in new else 'original'
        backbones=['tgat','dygformer'] if t in new else ['dygformer']
        pool=[n for n in TARGETS if n!=t]
        for b in backbones:
            for s in [[pool[0]],pool]:
                for p in SEEDS:
                    cases.append(case(t,s,p,profile,b,300))
            cases.append(case(t,profile=profile,backbone=b))
    return cases


def large_cases():
    return [c for t in TARGETS for c in
            [case(t,[n for n in TARGETS if n!=t],p,'large','tgat',300) for p in SEEDS]+[case(t,profile='large')]]


def breadth_jobs(winners,out=OUT):
    return [make_job('breadth' if c['profile']!='large' else 'scale',c,m,a,winner_config(winners,m),out)
            for a in SEEDS for c in breadth_cases()+large_cases() for m in MAIN]


def query_budget_jobs(winners,out=OUT):
    cases=[]
    for t in TARGETS:
        pool=[n for n in POOL if n!=t]
        cases.extend([case(t,[pool[0]]),case(t,pool),case(t)])
    return [make_job('query_budget',c,m,a,{**winner_config(winners,m),'query_fraction':f},out)
            for f in [.1,.25,1.] for a in SEEDS for c in cases for m in METHODS]


def source_budget_cases():
    paths=[('wikipedia',['reddit'],['reddit','lastfm']),
           ('reddit',['wikipedia','GDELT'],['wikipedia','ICEWS14','GDELT']),
           ('reddit',['mooc','lastfm'],['mooc','lastfm','ICEWS14']),
           ('mooc',['wikipedia','lastfm'],['wikipedia','lastfm','GDELT'])]
    cases=[]
    for t,s0,s1 in paths:
        for s in [s0,s1]:
            for p in SEEDS:
                for per_source in [None,60,300]:
                    budget=None if per_source is None else per_source*len(s)
                    c=case(t,s,p,source_budget=budget)
                    if c not in cases: cases.append(c)
    cases.extend(case(t) for t in sorted({x[0] for x in paths}))
    return cases


def source_budget_jobs(winners,out=OUT):
    return [make_job('source_budget',c,m,a,winner_config(winners,m),out)
            for a in SEEDS for c in source_budget_cases() for m in ['finetune','tfra']]


def protocol():
    return dict(version=1, authorized='2026-11-07: user requested all six validation packages',
        targets=TARGETS,new_targets=['ADX','BAG'],pretraining_seeds=SEEDS,adaptation_seeds=SEEDS,
        tuning=dict(methods=METHODS,trials=9,epochs=20,seed=7,
                    include_epoch0=True,
                    selection='highest mean Validation AP; target-equal, transfer and scratch equally weighted; lower trial index wins ties',
                    additional_lp_ft_C=[.001,.01,.1,1,10,100,1000]),
        extended_budget_epochs=50,confirmation_epochs=20,mechanism_epochs=5,
        evaluation=dict(validation_negative_seed=100007,test_negative_seed=200007,
                        additional_test_candidates=['random_10','historical_10'],
                        final_test='only after every development job completes, verification passes, and manifest seals all models',
                        historical='previous neighbors strictly before batch start, filtered at query timestamp; random fallback logged'),
        query_budget=dict(fractions=[.1,.25,1.],selection='nested fixed random subset of Train queries; chronological updates',
                          history='all causal observed edges available equally; this measures supervised update-query budget, not graph-observation budget'),
        larger_data=dict(events=131072,offset=32768,reason='later disjoint window avoids using the original small benchmark Test in development',
                         split='70/15/15, timestamp safe, new time scale fitted on Train'),
        source_budget=dict(total_original=60,updates_per_source=[60,300]),
        no_test_adaptation=True,no_test_model_selection=True,
        statistics='paired method differences; crossed seed bootstrap within each fixed Target/source set; shared scratch draws; no IID treatment of 900 edges')
