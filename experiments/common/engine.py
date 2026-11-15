"""Source-only pretraining, target adaptation and explicit final evaluation. Dry-run default."""
import argparse
import copy
import fcntl
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from experiments.common.runtime import (atomic_json, atomic_torch, capture_rng, restore_rng,
                                         seed_all, gpu_guard, pass_batches, evaluate)
from experiments.common import protocol
from experiments.common.backbones import SharedLinkPredictor

def comparison_contract(cfg):
    return dict(pretrain_updates_per_epoch=cfg.get('pretrain_updates_per_epoch'), **{k:cfg[k] for k in ('protocol','data_root','max_events','epochs','batch_size','history_length','learning_rate','weight_decay')})



def training_schedule(data, names, batch_size, epoch, budget=None):
    batches = {n:list(data[n].batches(data[n].indices(0),batch_size)) for n in names}
    if any(not b for b in batches.values()):raise ValueError('Empty training stream')
    if budget is None:
        for row in itertools.zip_longest(*(batches[n] for n in names)):
            for n,ids in zip(names,row):
                if ids is not None:yield n,ids
    else:
        if not isinstance(budget,int) or budget < len(names):
            raise ValueError('Update budget must cover every source at least once per epoch')
        # Continuous round-robin across epochs; cycle chronological batches per source.
        for step in range(epoch*budget,(epoch+1)*budget):
            n=names[step % len(names)]
            yield n,batches[n][(step//len(names)) % len(batches[n])]


def save_predictions(path, item, split, negatives, scores):
    ids=item.indices(split)
    tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f,event_ids=item.edge_id[ids],src=item.src[ids],dst=item.dst[ids],
            time=item.raw_time[ids],negatives=negatives[ids],scores=scores.reshape(-1,2),
            labels=np.tile([1,0],(len(ids),1)))
    tmp.replace(path)


def train_batch(model, data, ids, negatives, cfg, optimizer, weight, loss_fn=None):
    model.train();model.begin_batch(ids)
    if not any(p.requires_grad for p in model.encoder.parameters()):model.encoder.eval()
    optimizer.zero_grad(set_to_none=True)
    total = 0.
    for offset in range(0,len(ids),cfg['microbatch']):
        part = ids[offset:offset+cfg['microbatch']]
        scores = model.score(part,negatives[part])
        truth = torch.zeros_like(scores);truth[:,0] = 1.
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(scores,truth)
                if loss_fn is None else loss_fn(scores))
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
        (loss*weight*len(part)/len(ids)).backward()
        total += float(loss.detach())*len(part)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['grad_clip'])
    if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient')
    if weight > 0:optimizer.step()
    return total/len(ids)


def validate_parent(parent, cfg, stage, target, metadata):
    info = parent['provenance']
    if info['backbone'] != cfg['backbone']:
        raise ValueError('Checkpoint backbone mismatch')
    if stage in ('frozen','finetune'):
        if info['stage'] != 'pretrain' or target in info['source_networks']:
            raise ValueError('Transfer requires a source-only checkpoint excluding this target')
    if stage == 'evaluate':
        if info['stage'] not in ('target_only','frozen','finetune') or info['target'] != target:
            raise ValueError('Evaluate requires a checkpoint adapted to this target')
        if info['data'] != metadata:
            raise ValueError('Target data changed since training')


def run(args, cfg):
    if min(cfg[k] for k in ('epochs','batch_size','microbatch','history_length')) <= 0:
        raise ValueError('Training dimensions must be positive')
    request = protocol.plan(cfg,args.stage,args.target,args.sources)
    requires_checkpoint = args.stage in ('frozen','finetune','evaluate')
    if requires_checkpoint != bool(args.checkpoint):
        raise ValueError('This stage requires --checkpoint' if requires_checkpoint else 'This stage starts from scratch')
    if not all(d['ready'] for d in request['data']):
        raise FileNotFoundError('Prepare JODIE: python -m experiments.common.prepare_data')
    torch.set_num_threads(cfg['cpu_threads'])
    names = tuple(d['network'] for d in request['data'])
    data = {name:protocol.load(cfg,name) for name in names}
    metadata = {name:item.metadata for name,item in data.items()}
    parent_path = Path(args.checkpoint).resolve() if args.checkpoint else None
    parent = torch.load(parent_path,map_location='cpu',weights_only=False) if parent_path else None
    if parent is not None:
        validate_parent(parent,cfg,args.stage,args.target,metadata)
        info=parent['provenance']
        if info['seed'] != args.seed or info['target'] != args.target or info['source_networks'] != list(args.sources):
            raise ValueError('Parent checkpoint target/source set/seed does not match this run')
        if comparison_contract(info['config']) != comparison_contract(cfg):
            raise ValueError('Parent training budget/config mismatch')
        if info['entry'] != args.entry or info['config']['protocol'] != cfg['protocol']:
            raise ValueError('Parent experiment/entry protocol mismatch')
    parent_hash = hashlib.sha256(parent_path.read_bytes()).hexdigest() if parent_path else None
    provenance = dict(stage=args.stage,target=args.target,source_networks=request['source_networks'],
                      backbone=cfg['backbone'],entry=args.entry,seed=args.seed,data=metadata,
                      config=cfg,implementation=protocol.code_hashes(),parent_sha256=parent_hash)
    out = Path(args.out) if args.out else protocol.output_path(cfg,args.entry,args.target,args.sources,args.seed,args.stage)
    out.mkdir(parents=True,exist_ok=True)
    with (out/'run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'config.json').exists() and json.loads((out/'config.json').read_text()) != provenance:
            raise ValueError('Code/config/data changed; select a new output directory')
        if (out/'result.json').exists():print('Already complete',out);return
        atomic_json(out/'config.json',provenance)
        with gpu_guard(cfg['device'],cfg['minimum_free_gpu_gib']):
            seed_all(args.seed)
            if cfg['device'].startswith('cuda'):torch.cuda.reset_peak_memory_stats(cfg['device'])
            model = SharedLinkPredictor(data[names[0]],cfg['backbone'],cfg['history_length'],cfg['device'],args.seed)
            if parent is not None:
                if args.stage in ('evaluate',):model.load_state_dict(parent['model'])
                else:
                    # New target head starts from the same seed as target-only.
                    model.encoder.load_state_dict({k.removeprefix('encoder.'):v for k,v in parent['model'].items() if k.startswith('encoder.')})
            if args.stage in ('evaluate',):
                item = data[names[0]]
                negatives = item.negatives(args.seed+100000)
                metrics,scores = evaluate(model,item,2,negatives,cfg)
                save_predictions(out/'test_predictions.npz',item,2,negatives,scores)
                atomic_json(out/'result.json',dict(status='complete',metrics=metrics,stage=args.stage,parent_sha256=parent_hash,
                    experiment=cfg['experiment'],entry=args.entry,target=args.target,sources=list(args.sources),seed=args.seed,
                    split='test',comparison_contract=dict(comparison_contract(cfg),target_artifacts=metadata.get(args.target,{}).get('artifacts'))))
                print(out);return
            if args.stage == 'frozen':
                model.encoder.requires_grad_(False)
            optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                         lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
            history = [];best = -float('inf')
            if (out/'last.pt').exists():
                saved = torch.load(out/'last.pt',map_location=cfg['device'],weights_only=False)
                model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
                history = saved['history'];best = saved['best']
                restore_rng(saved['rng'],cfg['device'])
            fixed_negative = {n:d.negatives(args.seed+100000) for n,d in data.items()}
            for epoch in range(len(history),cfg['epochs']):
                start = time.monotonic()
                weights = {n:1. for n in names}
                negatives = {n:d.negatives(args.seed+epoch) for n,d in data.items()}
                losses = {n:[] for n in names};updates = 0;events_seen = {n:0 for n in names}
                budget=cfg.get('pretrain_updates_per_epoch') if args.stage=='pretrain' else None
                for name,ids in training_schedule(data,names,cfg['batch_size'],epoch,budget):
                    model.bind(data[name])
                    value = train_batch(model,data[name],ids,negatives[name],cfg,optimizer,weights[name])
                    losses[name].append(value);updates += 1;events_seen[name] += len(ids)
                validation = {}
                for name,item in data.items():
                    model.bind(item)
                    validation[name],_ = evaluate(model,item,1,fixed_negative[name],cfg)
                selection = float(np.mean([m['ap'] for m in validation.values()]))
                record = dict(epoch=epoch+1,validation=validation,selection_ap=selection,weights=weights,
                              train_loss={n:float(np.mean(v)) for n,v in losses.items()},updates=updates,
                              events_seen=events_seen,seconds=time.monotonic()-start)
                if cfg['device'].startswith('cuda'):
                    record['peak_allocated_gib'] = torch.cuda.max_memory_allocated(cfg['device'])/2**30
                    record['peak_reserved_gib'] = torch.cuda.max_memory_reserved(cfg['device'])/2**30
                history.append(record)
                payload = dict(model=model.state_dict(),provenance=provenance,epoch=epoch+1,selection_ap=selection)
                if selection > best:
                    best = selection;atomic_torch(out/'best.pt',payload)
                atomic_torch(out/'last.pt',dict(**payload,optimizer=optimizer.state_dict(),
                                             history=history,best=best,rng=capture_rng(cfg['device'])))
                atomic_json(out/'progress.json',dict(history=history));print(json.dumps(record),flush=True)
            selected = max(history,key=lambda r:r['selection_ap'])
            metrics = selected['validation'].get(args.target) if args.stage != 'pretrain' else None
            if args.stage != 'pretrain':
                selected_checkpoint=torch.load(out/'best.pt',map_location=cfg['device'],weights_only=False)
                model.load_state_dict(selected_checkpoint['model'])
                item=data[args.target];model.bind(item)
                metrics,scores=evaluate(model,item,1,fixed_negative[args.target],cfg)
                save_predictions(out/'validation_predictions.npz',item,1,fixed_negative[args.target],scores)
            atomic_json(out/'result.json',dict(status='complete',stage=args.stage,best_validation_ap=best,
                        experiment=cfg['experiment'],entry=args.entry,target=args.target,sources=list(args.sources),seed=args.seed,
                        split='validation',metrics=metrics,comparison_contract=dict(comparison_contract(cfg),target_artifacts=metadata.get(args.target,{}).get('artifacts')),
                        parameters=sum(p.numel() for p in model.parameters()),history=history,
                        note='Training/validation only. Final test requires an explicit evaluate stage.'))
            print('Saved',out,flush=True)

