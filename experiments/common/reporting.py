"""Read final result records; never train, select checkpoints, or mix test/validation."""
import argparse
import json
import statistics
from pathlib import Path


def collect(root,split):
    rows=[];seen=set()
    for path in sorted(Path(root).rglob('result.json')):
        result=json.loads(path.read_text())
        if result.get('status')!='complete' or result.get('split')!=split or not result.get('metrics'):
            continue
        if result['stage'] not in ('target_only','finetune','evaluate'):continue
        row={k:result[k] for k in ('entry','target','sources','seed','metrics','comparison_contract')}
        row['sources']=sorted(row['sources'])
        row['path']=str(path)
        key=(row['entry'],row['target'],tuple(row['sources']),row['seed'])
        if key in seen:raise ValueError(f'Duplicate result for {key}; choose one output version')
        seen.add(key);rows.append(row)
    return rows


def compatible(a,b):
    if a['comparison_contract']!=b['comparison_contract']:
        raise ValueError('Compared runs have different data/evaluation/training contracts')


def gains(rows):
    controls={(r['target'],r['seed']):r for r in rows if not r['sources'] and r['entry']=='uniform'}
    transfer=[];addition=[];missing=[]
    for row in rows:
        if not row['sources']:continue
        control=controls.get((row['target'],row['seed']))
        if control is None:
            missing.append(dict(target=row['target'],seed=row['seed'],reason='target_only_missing'))
        else:
            compatible(row,control)
            transfer.append(dict(entry=row['entry'],target=row['target'],sources=row['sources'],seed=row['seed'],
                                 **{f'{m}_gain':row['metrics'][m]-control['metrics'][m] for m in ('ap','auc')}))
        for previous in rows:
            if (previous['entry'],previous['target'],previous['seed']) != (row['entry'],row['target'],row['seed']):continue
            a,b=set(previous['sources']),set(row['sources'])
            if a and a < b and len(b)==len(a)+1:
                compatible(row,previous)
                addition.append(dict(entry=row['entry'],target=row['target'],seed=row['seed'],
                                     before=previous['sources'],after=row['sources'],added=sorted(b-a),
                                     **{f'{m}_gain':row['metrics'][m]-previous['metrics'][m] for m in ('ap','auc')}))
    return dict(target_only_gains=transfer,source_addition_gains=addition,missing_controls=missing)


def summaries(rows):
    groups={}
    for row in rows:groups.setdefault((row['entry'],row['target'],tuple(row['sources'])),[]).append(row)
    output=[]
    for (entry,target,sources),items in sorted(groups.items()):
        record=dict(entry=entry,target=target,sources=list(sources),seeds=sorted(r['seed'] for r in items))
        for metric in ('ap','auc'):
            values=[r['metrics'][metric] for r in items]
            record[metric+'_mean']=statistics.mean(values)
            record[metric+'_std']=statistics.stdev(values) if len(values)>1 else None
        output.append(record)
    return output


def main(config_file,transfer=False):
    from experiments.common.protocol import config,ROOT
    cfg=config(config_file)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT/cfg['output_root'])
    p.add_argument('--split',choices=('validation','test'),default='validation')
    p.add_argument('--output',type=Path)
    args=p.parse_args();rows=collect(args.root,args.split)
    payload=dict(split=args.split,rows=rows,summary=summaries(rows),
                 expected_seeds=cfg['seeds'],note='Missing seeds/results remain missing; no imputation. Signed seed-paired differences, not significance claims.')
    if transfer:payload.update(gains(rows))
    blob=json.dumps(payload,indent=2)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(blob)
    else:print(blob,end='')
