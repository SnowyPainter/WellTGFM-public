"""Isolated three-epoch comparisons: old serial versus optimized serial/parallel."""

import concurrent.futures, multiprocessing, time, sys, os
from pathlib import Path
import numpy as np, torch
from experiments.execution.parallel_budget import worker
from experiments.robust.training import ROOT, read
from experiments.common.runtime import gpu_guard, atomic_json

OUT = ROOT / "experiments/robust/outputs/v2_tgn/repairs/tgat_speed/benchmark"


def exact_tree(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(exact_tree(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(exact_tree(x, y) for x, y in zip(a, b))
    return a == b


def main():
    old = read(ROOT / "experiments/robust/outputs/v2_tgn/protocol.json")
    methods = [
        "finetune",
        "lp_ft",
        "refine",
        "tfra",
        "joint",
        "batch",
        "initial_fixed",
        "joint_init",
    ]
    jobs = []
    for method in methods:
        j = next(
            j
            for j in old["jobs"]
            if j["method"] == method
            and j["target"] == "wikipedia"
            and j["sources"] == ["reddit"]
            and j["config"]["epochs"] == 50
            and j["adaptation_seed"] == 7
        )
        j = {
            **j,
            "config": {**j["config"], "epochs": 3, "save_budgets": [3]},
            "id": "execution_check_" + method,
        }
        jobs.append(j)
    mps = os.environ.get("TFRA_BENCH_MPS") == "1"
    optimized = "optimized_mps" if mps else "optimized_parallel"
    results = read(OUT.parent / "benchmark.json") if mps else {}
    modes = (
        [(optimized, 3, True)]
        if mps
        else [("original_serial", 1, False), (optimized, 3, True)]
    )
    with gpu_guard("cuda:0", 0):
        for name, workers, fast in modes:
            root = OUT / name
            root.mkdir(parents=True, exist_ok=True)
            start = time.monotonic()
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=workers, mp_context=multiprocessing.get_context("spawn")
            ) as pool:
                futures = [pool.submit(worker, j, str(root), fast, False) for j in jobs]
                rows = [f.result() for f in futures]
            assert all(r["status"] == "complete" for r in rows), rows
            results[name] = dict(seconds=time.monotonic() - start, runs=rows)
            print(name, results[name]["seconds"], flush=True)
        checks = []
        for j in jobs:
            a = OUT / "original_serial/jobs" / j["id"]
            b = OUT / optimized / "jobs" / j["id"]
            ra, rb = read(a / "result.json"), read(b / "result.json")
            assert ra["best_epoch"] == rb["best_epoch"], j["method"]
            assert [h["validation"] for h in ra["history"]] == [
                h["validation"] for h in rb["history"]
            ], j["method"]
            sa, sb = [
                torch.load(x / "resume.pt", map_location="cpu", weights_only=False)
                for x in (a, b)
            ]
            exact = all(torch.equal(v, sb["model"][k]) for k, v in sa["model"].items())
            assert exact, j["method"]
            assert exact_tree(sa["optimizer"], sb["optimizer"]), (
                j["method"],
                "optimizer",
            )
            assert exact_tree(sa["rng"], sb["rng"]), (j["method"], "rng")
            with np.load(a / "validation_predictions.npz") as pa, np.load(
                b / "validation_predictions.npz"
            ) as pb:
                assert all(np.array_equal(pa[k], pb[k]) for k in pa.files)
            checks.append(
                dict(
                    method=j["method"],
                    final_parameters_exact=exact,
                    validation_and_selection_exact=True,
                )
            )
        results["speedup"] = (
            results["original_serial"]["seconds"] / results[optimized]["seconds"]
        )
        results["checks"] = checks
        results["status"] = "passed"
        atomic_json(
            OUT.parent / ("benchmark_mps.json" if mps else "benchmark.json"), results
        )
        print("Verified speedup", results["speedup"], flush=True)


if __name__ == "__main__":
    main()
