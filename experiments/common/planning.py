"""Build explicit stage dependencies without submitting or training any job."""
import argparse
import json
import sys
from pathlib import Path
from experiments.common import protocol


def build_jobs(cfg, entries, selected=None, include_test=False):
    chosen=selected or cfg['entries']
    if set(chosen)-set(entries):raise ValueError('Unknown entry')
    module=f"experiments.{cfg['experiment']}.run"
    jobs=[]
    def add(entry,target,sources,seed,stage,parent=None):
        out=protocol.output_path(cfg,entry,target,sources,seed,stage)
        command=[sys.executable,'-m',module,'--entry',entry,'--stage',stage,'--target',target,
                 '--seed',str(seed),'--out',str(out)]
        if sources:command+=['--sources',*sources]
        if parent:command+=['--checkpoint',str(parent/'best.pt')]
        job_id='/'.join((entry,target,'+'.join(sources) or 'target_only',str(seed),stage))
        jobs.append(dict(id=job_id,entry=entry,target=target,sources=list(sources),seed=seed,stage=stage,
                         output=str(out),checkpoint=None if parent is None else str(parent/'best.pt'),
                         status=entries[entry]['status'],reason=entries[entry].get('reason'),command=command))
        return out
    for target in cfg['datasets']:
        for seed in cfg['seeds']:
            # A single common-backbone target-only control is reused across methods/source sets.
            if cfg['experiment']=='transfer':
                out=add('uniform',target,(),seed,'target_only')
                if include_test:add('uniform',target,(),seed,'evaluate',out)
            for sources in protocol.source_sets(cfg['datasets'],target,cfg['experiment']=='prediction'):
                for entry in chosen:
                    pre=add(entry,target,sources,seed,'pretrain')
                    adapted=add(entry,target,sources,seed,'finetune',pre)
                    if include_test:add(entry,target,sources,seed,'evaluate',adapted)
    return jobs


def main(config_file,entries):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--entries',nargs='+',choices=entries);p.add_argument('--include-test',action='store_true')
    p.add_argument('--output',type=Path)
    args=p.parse_args();cfg=protocol.config(config_file)
    jobs=build_jobs(cfg,entries,args.entries,args.include_test)
    result=dict(experiment=cfg['experiment'],jobs=jobs,
                note='Plan only. No experiment is launched. Commands stay dry-run until --execute is explicitly added.')
    blob=json.dumps(result,indent=2)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(blob)
        print(f'{len(jobs)} stage jobs written to {args.output}')
    else:print(blob,end='')
