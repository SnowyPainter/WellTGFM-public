"""Combine completed Test ablations and describe their seed variability.

Reads saved metrics only. No training, model selection, or manuscript editing.
Run with --plot to regenerate the manuscript's Test Pareto figure.
"""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev, stdev

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "experiments/target_adaptation/outputs/test_v1/analysis"
NEW = ROOT / "experiments/transfer_validation/outputs/readout_mechanism_v2/analysis"
OUT = ROOT / "experiments/target_adaptation/outputs/ablation_test_v2"
METHODS = {
    "finetune": "Fine-tuning",
    "frozen": "Frozen Encoder",
    "warmstart": "Ridge Warm-start",
    "batch": "Batch-wise Ridge Profile",
    "joint": "Joint MSE",
    "initial_fixed": "TFRA-Fixed",
    "joint_init": "TFRA-Joint",
    "tfra": "TFRA",
}
SEEDS = (7, 17, 37)
METRICS = ("target_only_ap", "ap", "mean_gain", "D_G", "B_G")


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def condition(row):
    return row["target"], tuple(sorted(row["sources"].split("+"))), int(row["adaptation_seed"])


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def dominates(a, b):
    av, bv = (a["ap"], -a["D_G"], a["B_G"]), (b["ap"], -b["D_G"], b["B_G"])
    return all(x >= y for x, y in zip(av, bv)) and any(x > y for x, y in zip(av, bv))


def analyze():
    old_path, new_path = OLD / "panel_transfer.csv", NEW / "test_transfer.csv"
    old, new = read_csv(old_path), read_csv(new_path)
    assert len(old) == 252 and len(new) == 126
    old_tfra = {condition(r): r for r in old if r["method"] == "tfra"}
    new_tfra = {condition(r): r for r in new if r["method"] == "tfra"}
    assert len(old_tfra) == 42 and old_tfra.keys() == new_tfra.keys()
    for key, row in old_tfra.items():
        for field in ("ap", "target_only_ap", "gain"):
            assert abs(float(row[field]) - float(new_tfra[key][field])) < 1e-12
    # Keep the original six arms; add only the two new arms, with no duplicate TFRA runs.
    combined = old + [r for r in new if r["method"] in {"initial_fixed", "joint_init"}]
    grouped = defaultdict(dict)
    baselines = {}
    for row in combined:
        method, key = row["method"], condition(row)
        assert key not in grouped[method]
        assert row["split"] == "test" and row["evaluation"] == "random_1"
        assert row["backbone"] == "tgat" and int(row["pretrain_seed"]) == 7
        assert key[2] in SEEDS and key[0] not in key[1]
        ap, baseline, gain = (float(row[k]) for k in ("ap", "target_only_ap", "gain"))
        assert abs(ap - baseline - gain) < 1e-12
        bkey = method, key[0], key[2]
        assert abs(baselines.setdefault(bkey, baseline) - baseline) < 1e-12
        grouped[method][key] = dict(target_only_ap=baseline, ap=ap, mean_gain=gain,
                                    D_G=max(-gain, 0), B_G=max(gain, 0))
    assert set(grouped) == set(METHODS) and len(baselines) == 8 * 4 * 3
    targets = sorted({key[0] for key in old_tfra})
    configs = sorted({key[:2] for key in old_tfra})
    assert len(targets) == 4 and len(configs) == 14
    seed_rows, panel = [], {}
    for method, label in METHODS.items():
        runs = grouped[method]
        assert runs.keys() == old_tfra.keys()
        per_seed = []
        for seed in SEEDS:
            target_metrics = [
                {metric: mean(r[metric] for (t, _, s), r in runs.items() if t == target and s == seed)
                 for metric in METRICS} for target in targets
            ]
            row = dict(method=method, adaptation_seed=seed,
                       **{metric: mean(r[metric] for r in target_metrics) for metric in METRICS})
            per_seed.append(row)
            seed_rows.append(row)
        summary = dict(method=method, label=label, runs=42, adaptation_seeds=list(SEEDS))
        for metric in METRICS:
            values = [r[metric] for r in per_seed]
            summary[metric] = mean(values)
            summary[metric + "_sd"] = stdev(values)
        # Preserve the benchmark's distinct definition of Seed SD: mean within-config population SD.
        summary["seed_sd"] = mean(
            mean(pstdev(runs[t, sources, seed]["ap"] for seed in SEEDS)
                 for t, sources in configs if t == target) for target in targets
        )
        panel[method] = summary

    # Verify means and the original Seed SD against both previously published summaries.
    for path in (OLD / "panel_summary.csv", NEW / "test_summary.csv"):
        for row in read_csv(path):
            if row["target"] == "target_equal_mean":
                for metric in (*METRICS, "seed_sd"):
                    assert abs(panel[row["method"]][metric] - float(row[metric])) < 1e-12
    for method, row in panel.items():
        row["dominated_by"] = [m for m, other in panel.items() if dominates(other, row)]
        row["pareto"] = not row["dominated_by"]

    seed_lookup = {(r["method"], r["adaptation_seed"]): r for r in seed_rows}
    paired, paired_summary = [], []
    for other in ("initial_fixed", "joint_init"):
        for metric in METRICS:
            diffs = [seed_lookup["tfra", seed][metric] - seed_lookup[other, seed][metric] for seed in SEEDS]
            paired.extend(dict(comparison="tfra - " + other, adaptation_seed=seed, metric=metric, difference=d)
                          for seed, d in zip(SEEDS, diffs))
            paired_summary.append(dict(comparison="tfra - " + other, metric=metric,
                                       mean=mean(diffs), sd=stdev(diffs),
                                       positive=sum(d > 0 for d in diffs), negative=sum(d < 0 for d in diffs)))
    OUT.mkdir(parents=True, exist_ok=True)
    report = dict(
        inputs=[dict(path=str(p.relative_to(ROOT)), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                for p in (old_path, new_path)],
        aggregation="Per seed: average configurations within each Target, then equal weight across four Targets.",
        uncertainty="Sample SD (ddof=1) across three seed-level aggregates; descriptive, not a confidence interval.",
        seed_sd_definition="Target-equal mean of within-configuration AP population SD (ddof=0), as in Table 8.",
        pretrain_seed=7, source_configurations=14, transfer_runs_per_method=42, target_only_runs_per_method=12,
        objectives=["ap maximize", "D_G minimize", "B_G maximize"],
        panel=panel, paired_summary=paired_summary,
        checks=dict(condition_coverage=True, matched_target_only=True, original_summaries=True, tfra_replay_agreement=True),
    )
    (OUT / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    write_csv(OUT / "seed_metrics.csv", seed_rows)
    write_csv(OUT / "paired_differences.csv", paired)
    write_csv(OUT / "paired_summary.csv", paired_summary)
    write_csv(OUT / "summary.csv", list(panel.values()))

    header = ["| 방법 | Target-only Test AP | 전이 Test AP ↑ | 평균 전이 이득 \\(\\bar G\\) ↑ | 전이 피해 \\(D_G\\) ↓ | 전이 이득 \\(B_G\\) ↑ |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in panel.values():
        def cell(metric):
            return f"{row[metric]:.6f} ± {row[metric + '_sd']:.6f}"
        header.append(f"| {row['label']} | {row['target_only_ap']:.6f} | {cell('ap')} | {row['mean_gain']:+.6f} | {cell('D_G')} | {cell('B_G')} |")
    (OUT / "table9.md").write_text("\n".join(header) + "\n")
    print("\n".join(header))
    print("Nondominated:", ", ".join(m for m, r in panel.items() if r["pareto"]))
    print("Saved verified aggregation to", OUT.relative_to(ROOT))
    return panel


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    panel = analyze()
    if args.plot:
        from plots.tfra_pareto_test import render
        render(panel)
