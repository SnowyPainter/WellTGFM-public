"""Fixed grids: no Test-driven settings, new datasets, or scale expansion."""
from pathlib import Path
from experiments.transfer_validation.design import panel_cases, case, source_checkpoint
from experiments.robust.training import ROOT, DEFAULT, SEEDS, canonical_hash

HERE = Path(__file__).parent
OUT = HERE/'outputs/v2_tgn'
MAIN = ['finetune', 'lp_ft', 'refine', 'tfra']
BUDGET_METHODS = MAIN + ['joint', 'batch', 'initial_fixed', 'joint_init']
BACKBONE_METHODS = MAIN + ['batch']
LABELS = dict(finetune='Fine-tuning', lp_ft='LP-FT', refine='REFINE', tfra='TFRA',
              joint='Joint MSE', batch='Batch-wise Ridge Profile',
              initial_fixed='TFRA-Fixed', joint_init='TFRA-Joint')


def job(c, method, seed, config=None, out=OUT):
    cfg = {**DEFAULT, **(config or {})}
    ck = source_checkpoint(c, out)
    result = dict(**c, method=method, adaptation_seed=seed, config=cfg,
                  checkpoint=str(ck.relative_to(ROOT)) if ck else None)
    result['id'] = canonical_hash(result)[:24]
    # Cohorts are views over physical runs, enabling exact cross-stage reuse.
    result['cohort'] = 'robust'
    return result


def plan(out=OUT):
    jobs, stages = {}, {}
    def add(stage, c, method, seed, config=None, **view):
        j = job(c, method, seed, config, out)
        jobs[j['id']] = j
        stages.setdefault(stage, []).append(dict(job_id=j['id'], **view))
    panel = panel_cases()
    # One-factor changes; the central configuration belongs to both sweeps.
    for axis, values in [('ridge_lambda', [1e-4,1e-3,1e-2,1e-1,1.]),
                         ('learning_rate', [1e-5,1e-4,1e-3])]:
        for value in values:
            for seed in SEEDS:
                for c in panel:
                    add('sensitivity', c, 'tfra', seed, {axis:value}, axis=axis, value=value)
    for backbone in ['graphmixer','tgn']:
        for seed in SEEDS:
            for c in panel:
                bc = {**c, 'backbone':backbone, 'source_budget':60}
                for method in BACKBONE_METHODS:
                    add('backbone_'+backbone, bc, method, seed)
    for p in SEEDS:
        for seed in SEEDS:
            for c in panel:
                if not c['sources'] and p != 7:
                    continue
                sc = {**c, 'pretrain_seed':p if c['sources'] else None}
                for method in MAIN:
                    add('pretrain_seed', sc, method, seed)
    # Same trajectory gives best-Validation checkpoints within each budget.
    for seed in SEEDS:
        for c in panel:
            for method in BUDGET_METHODS:
                for budget in [5,20,50]:
                    add('budget', c, method, seed, dict(epochs=50, save_budgets=[5,20,50]), budget=budget)
    return dict(version=2, jobs=list(jobs.values()), stages=stages,
        panel=panel, adaptation_seeds=SEEDS, pretraining_seeds=SEEDS,
        selection='largest Validation AP within each fixed budget; epochs start at 1',
        reference='same backbone/method/settings/target/adaptation-seed scratch; scratch shared across pretraining seeds',
        aggregation='equal Target weights after within-Target averages; negative and positive parts computed per run',
        data='unchanged events32768 artifacts, chronological 70/15/15 split',
        tuning='one-factor sensitivity, all grid values reported; no winner selected on Test',
        manuscript_autowrite=False)
