"""Shared planning/dispatch; importing or planning never starts training."""
import argparse
import importlib
import json
from pathlib import Path
from experiments.common import protocol


def main(config_file, entries):
    p=argparse.ArgumentParser()
    p.add_argument('--config',default=str(config_file))
    p.add_argument('--entry',choices=entries,required=True)
    p.add_argument('--stage',choices=protocol.STAGES,required=True)
    p.add_argument('--target',required=True);p.add_argument('--sources',nargs='*',default=[])
    p.add_argument('--seed',type=int,default=7);p.add_argument('--checkpoint');p.add_argument('--out')
    p.add_argument('--device',choices=('cpu','cuda:0'));p.add_argument('--execute',action='store_true')
    args=p.parse_args();cfg=protocol.config(args.config)
    if args.device:cfg['device']=args.device
    args.sources=list(protocol.canonical_sources(cfg,args.target,args.sources))
    if cfg['experiment']=='prediction' and args.stage!='target_only':
        expected=set(cfg['datasets'])-{args.target}
        if set(args.sources)!=expected:p.error('Experiment 2 uses all three other datasets as sources')
    request=protocol.plan(cfg,args.stage,args.target,args.sources)
    needs_parent=args.stage in ('frozen','finetune','evaluate')
    if args.execute and needs_parent and not args.checkpoint:p.error('Specify --checkpoint for this stage')
    if args.checkpoint and not needs_parent:p.error('This stage starts from scratch')
    spec=entries[args.entry]
    if not args.execute:
        print(json.dumps(dict(status='dry_run',entry=args.entry,implementation=spec,seed=args.seed,
                             checkpoint=args.checkpoint,**request),indent=2));return
    if not spec['runner']:
        raise SystemExit(f"{args.entry} is not connected: {spec['reason']}")
    module,name=spec['runner'].split(':')
    getattr(importlib.import_module(module),name)(args,cfg)
