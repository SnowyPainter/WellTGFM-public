"""JODIE source subsets and shared comparison contracts."""
import hashlib
import itertools
from pathlib import Path
import yaml
from experiments.common.data import Events

ROOT = Path(__file__).resolve().parents[2]
STAGES = ('pretrain', 'target_only', 'frozen', 'finetune', 'evaluate')


def config(path):
    cfg = yaml.safe_load(Path(path).read_text())
    if len(cfg['datasets']) != len(set(cfg['datasets'])):
        raise ValueError('Duplicate datasets')
    return cfg


def source_sets(datasets, target, full_only=False):
    pool = tuple(n for n in datasets if n != target)
    sizes = (len(pool),) if full_only else range(1,len(pool)+1)
    return [tuple(s) for size in sizes for s in itertools.combinations(pool,size)]


def canonical_sources(cfg, target, sources):
    if target not in cfg['datasets']:
        raise ValueError('Unknown target')
    sources = tuple(sources or ())
    if len(sources) != len(set(sources)) or target in sources or set(sources)-set(cfg['datasets']):
        raise ValueError('Sources must be distinct known datasets excluding the target')
    return tuple(n for n in cfg['datasets'] if n in sources)


def load(cfg, name):
    data = Events.load(ROOT/cfg['data_root'],'ctdg',name)
    if data.metadata['max_events'] != cfg['max_events']:
        raise ValueError('Event budget does not match config')
    return data


def plan(cfg, stage, target, sources):
    sources = canonical_sources(cfg,target,sources)
    if stage == 'target_only' and sources:
        raise ValueError('Target-only has no sources')
    if stage in ('pretrain','frozen','finetune') and not sources:
        raise ValueError('Specify at least one source')
    names = sources if stage == 'pretrain' else (target,)
    entries = []
    for name in names:
        meta = ROOT/cfg['data_root']/'ctdg'/name/'metadata.json'
        entries.append(dict(network=name,ready=meta.exists(),metadata=str(meta)))
    return dict(stage=stage,target=target,source_networks=list(sources),data=entries,
                device=cfg['device'],epochs=cfg['epochs'],batch_size=cfg['batch_size'],microbatch=cfg['microbatch'])


def output_path(cfg, entry, target, sources, seed, stage):
    key = '+'.join(canonical_sources(cfg,target,sources)) or 'target_only'
    return ROOT/cfg['output_root']/entry/target/key/f'seed{seed}'/stage


def code_hashes():
    files = list((ROOT/'experiments').rglob('*.py'))
    files = [p for p in files if 'outputs' not in p.parts]
    files += list((ROOT/'models').rglob('*.py'))+[ROOT/'etl/prepare_benchmark.py']
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
