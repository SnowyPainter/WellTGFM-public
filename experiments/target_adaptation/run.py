"""Target-fitted readout and Encoder alternating adaptation."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from experiments.target_adaptation.runtime import ROOT, HERE, digest, canonical_hash, array_hash, build, load_inputs
from experiments.target_adaptation.readout import hidden, labels, refit
from experiments.common.engine import save_predictions
from experiments.common.runtime import atomic_json, atomic_torch, capture_rng, restore_rng, evaluate, gpu_guard

def execute(job, cfg, data, out, phash, smoke=False):
    dest = out / 'ridge_epoch' / job['job_id']
    dest.mkdir(parents=True, exist_ok=True)
    ck = ROOT / job['checkpoint'] if job['checkpoint'] else None
    assert job['pretrain_seed'] in (7, None) and job['adaptation_seed'] in (7, 17, 37)
    if ck:
        assert digest(ck) == job['checkpoint_sha256']
    manifest = dict(job=job, method='ridge_epoch', config=cfg, protocol_sha256=phash, runner_sha256=digest(Path(__file__)), solver_sha256=digest(Path(__file__).with_name('readout.py')), ridge_lambda=0.001, support_queries=256, query_queries=0, encoder_updates_per_epoch=12 if True else 0, head='fixed original hidden layer; last linear refit on full Train each epoch', dropout='disabled during support,query,and readout fitting', direction_objective='support MSE under epoch-fitted fixed readout', training_readout='full Train refit at epoch start', test_used=False)
    if (dest / 'manifest.json').exists():
        assert json.loads((dest / 'manifest.json').read_text()) == manifest
    atomic_json(dest / 'manifest.json', manifest)
    if (dest / 'result.json').exists() and (not smoke):
        return
    started = time.monotonic()
    seed = job['adaptation_seed']
    d = data[job['target']]
    model = build(cfg, data, seed)
    if ck:
        model.load_shared_state(torch.load(ck, map_location='cpu', weights_only=False)['shared'])
    model.bind(d)
    for p in model.parameters():
        p.requires_grad_(False)
    ep = list(model.encoders[model.target].parameters())
    for p in ep:
        p.requires_grad_(True)
    opt = torch.optim.Adam(ep, lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    fixed = d.negatives(100007)
    batches = list(d.batches(d.indices(0), cfg['batch_size']))
    history = []
    decisions = []
    best = -float('inf')
    best_epoch = None
    start = 0
    prior = 0.0
    if (dest / 'resume.pt').exists() and (not smoke):
        s = torch.load(dest / 'resume.pt', map_location=cfg['device'], weights_only=False)
        model.load_state_dict(s['model'])
        opt.load_state_dict(s['optimizer'])
        restore_rng(s['rng'], cfg['device'])
        history = s['history']
        decisions = s['decisions']
        best = s['best']
        best_epoch = s['best_epoch']
        start = s['epoch']
        prior = s['seconds']
    for epoch in range(start, cfg['epochs']):
        negatives = d.negatives(seed + epoch)
        losses = []
        refit(model, d, negatives, cfg)
        for (bi, native) in enumerate(batches):
            support = native[np.linspace(0, len(native) - 1, min(256, len(native)), dtype=int)]
            assert np.isin(support, d.indices(0)).all()
            model.bind(d)
            model.eval()
            opt.zero_grad(set_to_none=True)
            model.begin_batch(native)
            xs = hidden(model, d, support, negatives)
            ys = labels(len(support), xs.device, xs.dtype)
            w = model.heads[model.target][2].weight[0]
            b = model.heads[model.target][2].bias[0]
            loss = ((xs @ w + b - ys) ** 2).mean() + 0.001 * w.square().sum()
            if not torch.isfinite(loss):
                raise FloatingPointError('Ridge query loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(ep, cfg['grad_clip'])
            if not torch.isfinite(norm):
                raise FloatingPointError('Ridge gradient')
            opt.step()
            losses.append(float(loss.detach()))
            decisions.append(dict(epoch=epoch + 1, batch=bi, audit=dict(query_forwards=1, query_backwards=1, support_size=len(support), query_size=0, support_max_time=float(d.time[support].max()), training_risk=float(loss.detach()), loss=float(loss.detach()), gradient_norm=float(norm))))
            if smoke and len(decisions) == 3:
                atomic_json(dest / 'smoke.json', dict(status='passed', decisions=decisions))
                return
        fit = refit(model, d, negatives, cfg)
        (metrics, scores) = evaluate(model, d, 1, fixed, cfg)
        history.append(dict(epoch=epoch + 1, validation=metrics, train_loss=float(np.mean(losses)) if losses else None, readout=fit))
        if metrics['ap'] > best:
            best = metrics['ap']
            best_epoch = epoch + 1
            atomic_torch(dest / 'best.pt', dict(model=model.state_dict(), epoch=best_epoch))
            save_predictions(dest / 'validation_predictions.npz', d, 1, fixed, scores)
        atomic_torch(dest / 'resume.pt', dict(model=model.state_dict(), optimizer=opt.state_dict(), rng=capture_rng(cfg['device']), history=history, decisions=decisions, best=best, best_epoch=best_epoch, epoch=epoch + 1, seconds=prior + time.monotonic() - started))
        print(json.dumps(dict(method='ridge_epoch', job_id=job['job_id'], epoch=epoch + 1, ap=metrics['ap'])), flush=True)
        if smoke:
            atomic_json(dest / 'smoke.json', dict(status='passed', readout=fit))
            return
    with np.load(dest / 'validation_predictions.npz') as arrays:
        ehash = array_hash(arrays)
    atomic_json(dest / 'result.json', dict(status='complete', method='ridge_epoch', job=job, ap=best, best_epoch=best_epoch, last_epoch_ap=history[-1]['validation']['ap'], history=history, decisions=decisions, seconds=prior + time.monotonic() - started, protocol_sha256=phash, evaluation_sha256=ehash, checkpoint_sha256=job['checkpoint_sha256'], evaluation_negative_seed=100007, query_forwards=sum((x['audit']['query_forwards'] for x in decisions)), query_backwards=sum((x['audit']['query_backwards'] for x in decisions)), full_train_readout_passes=10, encoder_updates=len(decisions), initialization=None))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=HERE / 'outputs/reproduction')
    parser.add_argument('--cases', nargs='+')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.out.resolve() == (HERE / 'outputs').resolve():
        raise ValueError('Published results are immutable; use a new output directory')
    (jobs, cfgs, data) = load_inputs(load_data=args.execute)
    if args.cases:
        known = {j['case_id'] for j in jobs if j['sources']}
        if set(args.cases) - known:
            raise ValueError('Unknown case; only the fixed development panel is allowed')
        targets = {j['target'] for j in jobs if j['case_id'] in args.cases}
        jobs = [j for j in jobs if j['case_id'] in args.cases or (not j['sources'] and j['target'] in targets)]
    if args.smoke:
        jobs = [next((j for j in jobs if not j['sources'])), next((j for j in jobs if j['sources']))]
    protocol = dict(configs=cfgs, pretrain_seed=7, evaluation_negative_seed=100007, epochs=5, model_selection='best_validation_ap', test_used=False)
    phash = canonical_hash(protocol)
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / 'protocol.json').exists():
        assert json.loads((args.out / 'protocol.json').read_text()) == protocol
    atomic_json(args.out / 'protocol.json', protocol)
    atomic_json(args.out / 'execution_plan.json', dict(jobs=jobs, method='ridge_epoch', smoke=args.smoke))
    print(json.dumps(dict(jobs=len(jobs), execute=args.execute)), flush=True)
    if not args.execute:
        return
    torch.set_num_threads(2)
    with gpu_guard('cuda:0', 0):
        for job in jobs:
            execute(job, cfgs[job['target']], data, args.out, phash, args.smoke)
            torch.cuda.empty_cache()
    atomic_json(args.out / 'progress.json', dict(status='smoke_complete' if args.smoke else 'complete'))
if __name__ == '__main__':
    main()
