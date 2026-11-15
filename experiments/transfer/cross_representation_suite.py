"""Exhaustive CTDG-target transfer with ICEWS14/GDELT source expansion."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import yaml
from experiments.common import protocol
from experiments.common.runtime import atomic_json
from experiments.transfer.cross_representation import source_conditions


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    base = yaml.safe_load(Path(__file__).with_name('cross_representation.yaml').read_text())
    targets = ['wikipedia', 'reddit', 'mooc', 'lastfm']
    datasets = targets + ['ICEWS14', 'GDELT']
    root = protocol.ROOT / 'experiments/transfer/outputs/cross_representation_two_tkg_v1'
    configs = []
    for target in targets:
        cfg = dict(base, datasets=datasets,
                   families={n: 'ctdg' if n in targets else 'tkg' for n in datasets},
                   target=target, sources=[n for n in datasets if n != target],
                   all_source_subsets=True,
                   output_root=str((root / target).relative_to(protocol.ROOT)))
        assert len(source_conditions(cfg)) == 32
        configs.append(cfg)
    plan = dict(datasets=datasets, targets=targets, seeds=base['seeds'],
                conditions_per_target_per_seed=32, conditions=384,
                tkg_transfer_conditions=288, matched_control_conditions=96,
                stages=756, pretrain_updates=60,
                note='31 nonempty source subsets plus target-only; matched private-head controls; CTDG targets only; validation only.')
    if not args.execute:
        print(json.dumps(plan, indent=2)); return
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / 'suite_plan.json', plan)
    try:
        for i, cfg in enumerate(configs):
            path = root / (cfg['target'] + '.yaml')
            blob = yaml.safe_dump(cfg, sort_keys=False)
            if path.exists() and path.read_text() != blob:
                raise ValueError('Suite config changed')
            path.write_text(blob)
            atomic_json(root / 'suite_progress.json', dict(status='running', target=cfg['target'], target_index=i+1, total_targets=4))
            subprocess.run([sys.executable, '-u', '-m', 'experiments.transfer.cross_representation',
                            '--config', str(path), '--execute'], cwd=protocol.ROOT, check=True)
    except Exception as exc:
        atomic_json(root / 'suite_progress.json', dict(status='failed', error=str(exc)))
        raise
    atomic_json(root / 'suite_progress.json', dict(status='complete', targets=4, conditions=384, stages=756))


if __name__ == '__main__':
    main()
