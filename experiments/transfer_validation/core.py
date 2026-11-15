"""Training primitives for the prospective experiment suite.

No function in this module evaluates Test. Every training job writes an immutable
manifest, epoch-resumable optimizer/RNG state, and its Validation-selected model.
"""
import copy
import hashlib
import json
import shutil
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from torch import nn

from experiments.common.data import Events
from experiments.common.backbones import SharedLinkPredictor
from experiments.common.engine import train_batch, save_predictions
from experiments.common.runtime import (atomic_json, atomic_torch, capture_rng,
                                        restore_rng, seed_all, evaluate)
from experiments.target_adaptation.runtime import ROOT, canonical_hash, digest
from experiments.target_adaptation.readout import ridge, labels
from models.ours.joint import CrossRepresentationPredictor, tie_shared

HERE = Path(__file__).resolve().parent
BASE = ROOT / 'experiments/target_adaptation'
OLD = ROOT / 'experiments/transfer/outputs/cross_representation_two_tkg_v1'
TARGETS = ['wikipedia', 'reddit', 'mooc', 'lastfm']
SEEDS = [7, 17, 37]
POOL = TARGETS + ['ICEWS14', 'GDELT']
MSE_METHODS = {'tfra', 'joint', 'joint_init', 'full_step', 'batch', 'frozen', 'initial_fixed'}
DEFAULT = dict(learning_rate=1e-4, ridge_lambda=1e-3, dropout=.1,
               weight_decay=0., grad_clip=1., batch_size=2048, tkg_batch_size=1024,
               microbatch=512, history_length=32, device='cuda:0', epochs=5,
               query_fraction=1., support_queries=256)


def read(path):
    return json.loads(Path(path).read_text())


@lru_cache(maxsize=24)
def event_data(root, family, name):
    return Events.load(ROOT / root, family, name)


def data_for(job):
    if job.get('profile', 'original') == 'original':
        names = POOL if job.get('backbone', 'tgat') == 'tgat' else TARGETS
        return {n: event_data('data/benchmarks/events32768', 'ctdg' if n in TARGETS else 'tkg', n)
                for n in names}
    if job['profile'] == 'mint':
        names = TARGETS
        data = {n: event_data('data/benchmarks/events32768', 'ctdg', n) for n in names}
        data[job['target']] = event_data('data/benchmarks/mint32768', 'ctdg', job['target'])
        return data
    if job['profile'] == 'large':
        return {n: event_data('data/benchmarks/tfra_validation_late131072', 'ctdg', n)
                for n in TARGETS}
    raise ValueError(job['profile'])


class MultiDyGFormer(nn.Module):
    """Same cross-domain sharing contract as TGAT, using CTDG-only DyGFormer."""
    def __init__(self, data, target, cfg, seed):
        super().__init__()
        self.target = target
        self.history = cfg['history_length']
        self.encoders = nn.ModuleDict()
        self.heads = nn.ModuleDict()
        self._models = {}
        self.bindings = {id(d): n for n, d in data.items()}
        for n in [target] + sorted(n for n in data if n != target):
            p = SharedLinkPredictor(data[n], 'dygformer', self.history, cfg['device'], seed)
            self.encoders[n] = p.encoder
            self.heads[n] = p.head
            self._models[n] = p
            if n != target:
                tie_shared(p.encoder, self.encoders[target])
        self.bind(data[target])

    def bind(self, data):
        self.name = self.bindings[id(data)]
        self.data = data
        self._models[self.name].bind(data)

    def reset(self):
        self._models[self.name].reset()

    def begin_batch(self, ids):
        self._models[self.name].begin_batch(ids)

    def observe(self, ids):
        pass

    def score(self, ids, negatives):
        return self._models[self.name].score(ids, negatives)

    def shared_state(self):
        return {k: v.detach().cpu().clone() for k, v in self.encoders[self.target].state_dict().items()}

    def load_shared_state(self, state):
        self.encoders[self.target].load_state_dict(state, strict=True)

    @property
    def encoder(self):
        return self.encoders[self.name]


def build(job, cfg, data, seed):
    seed_all(seed)
    if job.get('backbone', 'tgat') == 'tgat':
        model = CrossRepresentationPredictor(data, job['target'], cfg['history_length'], cfg['device'], seed)
    else:
        model = MultiDyGFormer(data, job['target'], cfg, seed)
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = cfg['dropout']
        elif isinstance(m, nn.MultiheadAttention):
            m.dropout = cfg['dropout']
    return model


def features(model, d, ids, negatives):
    n = len(ids)
    u = np.tile(d.src[ids], 2)
    v = np.r_[d.dst[ids], negatives[ids]]
    t = np.tile(d.time[ids], 2)
    enc = model.encoders[model.target]
    kw = {} if isinstance(model, MultiDyGFormer) else {'num_neighbors': model.history}
    a, b = enc.compute_src_dst_node_temporal_embeddings(u, v, t, **kw)
    head = model.heads[model.target]
    return head[1](head[0](torch.cat((a, b), -1)))


class ResidualPredictor(nn.Module):
    """Same REFINE mapping as the completed comparison, also usable on DyGFormer."""
    def __init__(self, job, cfg, data, shared=None):
        super().__init__()
        self.source = build(job, cfg, data, job['adaptation_seed'])
        if shared is not None:
            self.source.load_shared_state(shared)
        self.source.requires_grad_(False)
        self.residual = build(job, cfg, data, job['adaptation_seed'])
        self.residual.requires_grad_(False)
        self.residual.encoders[job['target']].requires_grad_(True)
        self.residual.heads[job['target']][0].requires_grad_(True)
        self.classifier = nn.Linear(344, 1, device=cfg['device'])
        self.bind(data[job['target']])

    @property
    def encoder(self):
        return self.residual.encoder

    def bind(self, data):
        self.data = data
        self.source.bind(data)
        self.residual.bind(data)

    def reset(self):
        self.source.reset()
        self.residual.reset()

    def begin_batch(self, ids):
        self.source.begin_batch(ids)
        self.residual.begin_batch(ids)

    def observe(self, ids):
        pass

    def train(self, mode=True):
        super().train(mode)
        self.source.eval()
        return self

    def score(self, ids, negative):
        neg = np.empty(len(self.data.src), np.int64)
        neg[ids] = negative
        with torch.no_grad():
            a = features(self.source, self.data, ids, neg)
        b = features(self.residual, self.data, ids, neg)
        return self.classifier(torch.cat((b, a), -1)).reshape(2, len(ids)).T


def model_for(job, cfg, data):
    shared = None
    if job.get('checkpoint'):
        p = ROOT / job['checkpoint']
        if job.get('checkpoint_sha256'):
            assert digest(p) == job['checkpoint_sha256']
        shared = torch.load(p, map_location='cpu', weights_only=False)['shared']
    if job['method'] == 'refine':
        return ResidualPredictor(job, cfg, data, shared)
    model = build(job, cfg, data, job['adaptation_seed'])
    if shared is not None:
        model.load_shared_state(shared)
    # Separate hidden/final-layer seeds while restoring all other randomness.
    if job.get('hidden_seed') is not None or job.get('final_seed') is not None:
        state = capture_rng(cfg['device'])
        for key, index in [('hidden_seed', 0), ('final_seed', 2)]:
            if job.get(key) is not None:
                donor = build(job, cfg, data, job[key])
                model.heads[job['target']][index].load_state_dict(donor.heads[job['target']][index].state_dict())
                del donor
        restore_rng(state, cfg['device'])
    return model


def train_queries(d, fraction):
    """Nested deterministic label budgets; causal history remains available to all arms."""
    ids = d.indices(0)
    if not 0 < fraction <= 1:
        raise ValueError('Invalid supervised-query fraction')
    # A fixed permutation makes all fraction budgets nested; sort restores time order.
    order = np.random.default_rng(800007).permutation(ids)
    return np.sort(order[:max(1, int(len(ids) * fraction))])


def selected_batches(d, cfg):
    allowed = train_queries(d, cfg['query_fraction'])
    mask = np.zeros(len(d.src), bool)
    mask[allowed] = True
    return [(native, native[mask[native]]) for native in d.batches(d.indices(0), cfg['batch_size'])
            if mask[native].any()]


def limited(ids, size):
    return ids[np.linspace(0, len(ids)-1, min(size, len(ids)), dtype=int)]


@torch.no_grad()
def extract(model, d, negatives, cfg, split=0):
    model.bind(d)
    model.eval()
    xs, ys = [], []
    batches = selected_batches(d, cfg) if split == 0 else [(x, x) for x in d.batches(d.indices(split), cfg['batch_size'])]
    for native, selected in batches:
        model.begin_batch(native)
        for off in range(0, len(selected), cfg['microbatch']):
            ids = selected[off:off+cfg['microbatch']]
            x = features(model, d, ids, negatives)
            xs.append(x)
            ys.append(labels(len(ids), x.device, x.dtype))
    return torch.cat(xs), torch.cat(ys)


@torch.no_grad()
def fit(model, d, negatives, cfg):
    x, y = extract(model, d, negatives, cfg)
    w, b = ridge(x, y, cfg['ridge_lambda'])
    head = model.heads[model.target][2]
    head.weight.copy_(w[None])
    head.bias.copy_(b.reshape(1))
    return dict(examples=len(y), mse=float(((x@w+b-y)**2).mean()))


@torch.no_grad()
def logistic_probe(model, d, negatives, cfg):
    state = capture_rng(cfg['device'])
    x, y = extract(model, d, negatives, cfg)
    vx, vy = extract(model, d, d.negatives(100007), cfg, split=1)
    x, y = x.cpu().double().numpy(), (y.cpu().numpy() > 0).astype(int)
    vx, vy = vx.cpu().double().numpy(), (vy.cpu().numpy() > 0).astype(int)
    best, trials = None, []
    for c in [.001, .01, .1, 1., 10., 100., 1000.]:
        lr = LogisticRegression(C=c, solver='newton-cholesky', max_iter=2000, tol=1e-8).fit(x, y)
        ap = float(average_precision_score(vy, lr.decision_function(vx)))
        trials.append(dict(C=c, validation_ap=ap, iterations=int(lr.n_iter_.max())))
        if best is None or ap > best[0]:
            best = ap, c, lr
    head = model.heads[model.target][2]
    head.weight.copy_(torch.as_tensor(best[2].coef_, device=head.weight.device, dtype=head.weight.dtype))
    head.bias.copy_(torch.as_tensor(best[2].intercept_, device=head.bias.device, dtype=head.bias.dtype))
    restore_rng(state, cfg['device'])
    return dict(selected_C=best[1], trials=trials)


def gradient_snapshot(model, d, negatives, cfg):
    """At the SAME theta and MSE objective, measure current vs fitted full gradients."""
    model.eval()
    head = model.heads[model.target][2]
    current = {k: v.detach().clone() for k, v in head.state_dict().items()}
    params = list(model.encoders[model.target].parameters())
    count = len(train_queries(d, cfg['query_fraction']))
    def full_gradient():
        model.zero_grad(set_to_none=True)
        mse = 0.
        for native, selected in selected_batches(d, cfg):
            model.begin_batch(native)
            for off in range(0, len(selected), cfg['microbatch']):
                ids = selected[off:off+cfg['microbatch']]
                z = features(model, d, ids, negatives)
                y = labels(len(ids), z.device, z.dtype)
                loss = ((head(z).reshape(-1)-y)**2).mean()
                (loss * len(ids)/count).backward()
                mse += float(loss.detach()) * len(ids)/count
        g = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).detach().reshape(-1) for p in params])
        return g, mse + cfg['ridge_lambda'] * float(head.weight.detach().square().sum())
    with torch.enable_grad():
        a, qa = full_gradient()
        fit(model, d, negatives, cfg)
        b, qb = full_gradient()
    head.load_state_dict(current)
    model.zero_grad(set_to_none=True)
    return dict(objective_current=qa, objective_fitted=qb, epsilon=qa-qb,
                gradient_gap=float((a-b).norm()), current_norm=float(a.norm()), fitted_norm=float(b.norm()),
                cosine=float(torch.dot(a,b)/(a.norm()*b.norm()).clamp_min(1e-30)))


def train_job(job, out, *, smoke=False):
    cfg = {**DEFAULT, **job.get('config', {})}
    dest = out / 'jobs' / job['id']
    dest.mkdir(parents=True, exist_ok=True)
    if (out/'test_freeze.json').exists() and not (dest/'result.json').exists():
        raise RuntimeError('Training is sealed after final Test begins')
    data = data_for(job)
    manifest = dict(job=job, config=cfg, code_sha256=digest(Path(__file__)),
                    checkpoint_sha256=digest(ROOT/job['checkpoint']) if job.get('checkpoint') else None,
                    data_artifacts={n:d.metadata.get('artifacts',{}) for n,d in data.items()},
                    split='validation', test_used=False)
    if (dest/'manifest.json').exists():
        assert read(dest/'manifest.json') == manifest, 'Training manifest changed'
    atomic_json(dest/'manifest.json', manifest)
    if (dest/'result.json').exists():
        return read(dest/'result.json')
    start_time = time.monotonic()
    d = data[job['target']]
    model = model_for(job, cfg, data)
    model.bind(d)
    method = job['method']
    if method in MSE_METHODS:
        model.requires_grad_(False)
        model.encoders[job['target']].requires_grad_(method != 'frozen')
        model.heads[job['target']][2].requires_grad_(method in {'joint', 'joint_init'})
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=cfg['learning_rate'], weight_decay=cfg['weight_decay']) if params else None
    history, diagnostics = [], []
    best, best_epoch, first, prior, updates, fits = -float('inf'), None, 0, 0., 0, 0
    initial_validation = None
    if (dest/'resume.pt').exists():
        ck = torch.load(dest/'resume.pt', map_location=cfg['device'], weights_only=False)
        model.load_state_dict(ck['model'])
        if opt: opt.load_state_dict(ck['optimizer'])
        restore_rng(ck['rng'], cfg['device'])
        history, diagnostics = ck['history'], ck['diagnostics']
        best, best_epoch, first, prior = ck['best'], ck['best_epoch'], ck['epoch'], ck['seconds']
        updates, fits = ck['updates'], ck['fits']
        initial_validation = ck.get('initial_validation')
    else:
        neg = d.negatives(job['adaptation_seed'])
        if method == 'lp_ft':
            atomic_json(dest/'probe.json', logistic_probe(model, d, neg, cfg))
        if method in {'joint_init', 'initial_fixed'}:
            fit(model, d, neg, cfg); fits += 1
    fixed = d.negatives(100007)
    if first==0 and cfg.get('include_epoch0',False) and initial_validation is None:
        original_head = None
        if method in MSE_METHODS:
            original_head=copy.deepcopy(model.heads[model.target][2].state_dict())
            fit(model,d,d.negatives(job['adaptation_seed']),cfg);fits+=1
        initial_validation,scores=evaluate(model,d,1,fixed,cfg)
        best,best_epoch=initial_validation['ap'],0
        atomic_torch(dest/'best.pt',dict(model=model.state_dict(),epoch=0))
        save_predictions(dest/'validation_predictions.npz',d,1,fixed,scores)
        if original_head is not None:model.heads[model.target][2].load_state_dict(original_head)
    for epoch in range(first, min(cfg['epochs'], 1) if smoke else cfg['epochs']):
        neg = d.negatives(job['adaptation_seed'] + epoch)
        losses = []
        if method == 'tfra':
            fit(model, d, neg, cfg); fits += 1
        batches = selected_batches(d, cfg)
        for bi, (native, chosen) in enumerate(batches):
            if method == 'frozen':
                break
            if method not in MSE_METHODS:
                # Keep the native batch history cutoff even when labels are subsampled.
                # train_batch resets cutoff using ids[0], so use the explicit loop below.
                model.train(); model.begin_batch(native); opt.zero_grad(set_to_none=True)
                total = 0.
                for off in range(0, len(chosen), cfg['microbatch']):
                    ids = chosen[off:off+cfg['microbatch']]
                    scores = model.score(ids, neg[ids]); truth = torch.zeros_like(scores); truth[:,0] = 1.
                    loss = nn.functional.binary_cross_entropy_with_logits(scores, truth)
                    if not torch.isfinite(loss): raise FloatingPointError('BCE loss')
                    (loss*len(ids)/len(chosen)).backward(); total += float(loss.detach())*len(ids)/len(chosen)
            else:
                if method == 'full_step':
                    fit(model, d, neg, cfg); fits += 1
                model.eval(); model.begin_batch(native); opt.zero_grad(set_to_none=True)
                ids = limited(chosen, cfg['support_queries'])
                z = features(model, d, ids, neg); y = labels(len(ids), z.device, z.dtype)
                head = model.heads[model.target][2]
                if method == 'batch':
                    with torch.no_grad():
                        w, b = ridge(z.detach(), y, cfg['ridge_lambda'])
                        head.weight.copy_(w[None]); head.bias.copy_(b.reshape(1))
                    fits += 1
                loss = ((head(z).reshape(-1)-y)**2).mean() + cfg['ridge_lambda']*head.weight.square().sum()
                if not torch.isfinite(loss): raise FloatingPointError('MSE loss')
                loss.backward(); total = float(loss.detach())
            norm = nn.utils.clip_grad_norm_(params, cfg['grad_clip'])
            if not torch.isfinite(norm): raise FloatingPointError('Gradient')
            opt.step(); updates += 1; losses.append(total)
            if job.get('diagnostics') and bi in {0, len(batches)-1}:
                state = capture_rng(cfg['device'])
                diagnostics.append(dict(epoch=epoch+1, batch=bi, **gradient_snapshot(model,d,neg,cfg)))
                restore_rng(state, cfg['device'])
            if smoke and bi == 1:
                break
        training_head = None
        if method in MSE_METHODS:
            training_head = copy.deepcopy(model.heads[model.target][2].state_dict())
            fit(model, d, neg, cfg); fits += 1
        metrics, scores = evaluate(model, d, 1, fixed, cfg)
        if not np.isfinite(metrics['ap']): raise FloatingPointError('Validation AP')
        elapsed = prior + time.monotonic()-start_time
        history.append(dict(epoch=epoch+1, validation=metrics, train_loss=float(np.mean(losses)) if losses else None,
                            seconds=elapsed, encoder_updates=updates, readout_fits=fits))
        if metrics['ap'] > best:
            best, best_epoch = metrics['ap'], epoch+1
            atomic_torch(dest/'best.pt', dict(model=model.state_dict(), epoch=best_epoch))
            save_predictions(dest/'validation_predictions.npz', d, 1, fixed, scores)
        if job['cohort']=='long_budget' and epoch+1 in {5,20,50}:
            shutil.copy2(dest/'best.pt',dest/f'budget{epoch+1}.pt')
        # Pure evaluation refitting must NOT replace a jointly trained or stale head.
        if method in {'joint', 'joint_init', 'initial_fixed'}:
            model.heads[model.target][2].load_state_dict(training_head)
        atomic_torch(dest/'resume.pt', dict(model=model.state_dict(), optimizer=opt.state_dict() if opt else None,
            rng=capture_rng(cfg['device']), history=history, diagnostics=diagnostics, best=best,
            best_epoch=best_epoch, epoch=epoch+1, seconds=elapsed, updates=updates, fits=fits,
            initial_validation=initial_validation))
        print(json.dumps(dict(job=job['id'], epoch=epoch+1, ap=metrics['ap'], seconds=elapsed)), flush=True)
    result = dict(status='smoke_complete' if smoke else 'complete', job=job, ap=best, best_epoch=best_epoch,
                  history=history, diagnostics=diagnostics, seconds=prior+time.monotonic()-start_time,
                  train_queries=len(train_queries(d,cfg['query_fraction'])), updates=updates, fits=fits,
                  initial_validation=initial_validation,
                  checkpoint_sha256=manifest['checkpoint_sha256'], test_used=False)
    atomic_json(dest/('smoke.json' if smoke else 'result.json'), result)
    return result
