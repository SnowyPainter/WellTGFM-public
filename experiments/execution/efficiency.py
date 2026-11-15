"""Isolated serial timing on a fixed, outcome-independent subset of the ablation panel."""

import concurrent.futures
import contextlib
import csv
import hashlib
import json
import multiprocessing
import random
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from experiments.common.runtime import atomic_json, gpu_guard
from experiments.robust.training import ROOT, read, train_job
from experiments.robust.evaluation import evaluate_job

OUT = ROOT / "experiments/execution/outputs/efficiency_v1"
UPSTREAM = ROOT / "experiments/robust/outputs/v2_tgn"


def measure(job):
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    from models.baselines.ctdg.tgat import TGAT

    TGAT.fast_history_enabled = True
    root = OUT / "runs" / job["id"]
    root.mkdir(parents=True, exist_ok=True)
    # CUDA context and generic BLAS warm-up are outside the measured interval.
    x = torch.ones((256, 256), device="cuda")
    for _ in range(3):
        x = x @ x
    del x
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    with (root / "run.log").open("a", buffering=1) as log, contextlib.redirect_stdout(
        log
    ), contextlib.redirect_stderr(log):
        start = time.perf_counter()
        result = train_job(job, root)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        peak = torch.cuda.max_memory_allocated()
        # Test scoring is excluded from the timing and does not change training/selection.
        evaluate_job(job, root)
    test = read(root / "jobs" / job["id"] / "test_budget5.json")
    record = dict(
        id=job["id"],
        target=job["target"],
        sources=job["sources"],
        method=job["method"],
        adaptation_seed=job["adaptation_seed"],
        end_to_end_seconds=elapsed,
        adaptation_seconds=result["seconds"],
        peak_allocated_bytes=peak,
        test_ap=test["metrics"]["ap"],
        selected_epoch=test["selected_epoch"],
        completed_at=time.time(),
    )
    atomic_json(root / "timing.json", record)
    return record


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    p = read(UPSTREAM / "protocol.json")
    ids = {v["job_id"] for v in p["stages"]["budget"]}
    pool = [j for j in p["jobs"] if j["id"] in ids and j["sources"]]
    selected = {
        t: min(
            {tuple(j["sources"]) for j in pool if j["target"] == t},
            key=lambda s: (len(s), tuple(sorted(s))),
        )
        for t in sorted({j["target"] for j in pool})
    }
    jobs = [
        {**j, "config": {**j["config"], "epochs": 5, "save_budgets": [5]}}
        for j in pool
        if tuple(j["sources"]) == selected[j["target"]]
    ]
    assert len(jobs) == 96
    random.Random(20261110).shuffle(jobs)
    paths = sorted(
        set(
            list((ROOT / "models").rglob("*.py"))
            + list((ROOT / "experiments/robust").glob("*.py"))
            + list((ROOT / "experiments/common").glob("*.py"))
            + [Path(__file__)]
        )
    )
    plan = dict(
        jobs=jobs,
        selection="Minimum source count, then lexicographic source names; no performance selection",
        implementation={
            str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in paths
        },
        serial=True,
        threads=2,
        precision="high",
        epochs=5,
        timing="End-to-end includes data/model loading, fitting, training, Validation and checkpoint I/O; excludes process startup, CUDA warm-up and Test. adaptation_seconds excludes initial data loading.",
        upstream_protocol_sha256=hashlib.sha256(
            (UPSTREAM / "protocol.json").read_bytes()
        ).hexdigest(),
    )
    if (OUT / "protocol.json").exists():
        assert read(OUT / "protocol.json") == plan
    else:
        atomic_json(OUT / "protocol.json", plan)
    try:
        with gpu_guard("cuda:0", 0):
            atomic_json(
                OUT / "environment.json",
                dict(
                    torch=torch.__version__,
                    cuda=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(0),
                ),
            )
            rows = []
            for j in jobs:
                root = OUT / "runs" / j["id"]
                done = root / "timing.json"
                if done.exists():
                    rows.append(read(done))
                    continue
                # Restart interrupted timing from scratch, never time a resumed run.
                if root.exists():
                    shutil.rmtree(root)
                atomic_json(
                    OUT / "progress.json",
                    dict(
                        status="running",
                        completed=len(rows),
                        total=96,
                        job=j,
                        updated_at=time.time(),
                    ),
                )
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=1, mp_context=multiprocessing.get_context("spawn")
                ) as ex:
                    row = ex.submit(measure, j).result()
                rows.append(row)
                with (OUT / "timings.csv").open("w") as f:
                    w = csv.DictWriter(f, fieldnames=list(row))
                    w.writeheader()
                    w.writerows(rows)
                summary = []
                for m in sorted({r["method"] for r in rows}):
                    rr = [r for r in rows if r["method"] == m]
                    if len(rr) != 12:
                        continue
                    summary.append(
                        dict(
                            method=m,
                            runs=len(rr),
                            seconds=float(
                                np.mean([r["end_to_end_seconds"] for r in rr])
                            ),
                            adaptation_seconds=float(
                                np.mean([r["adaptation_seconds"] for r in rr])
                            ),
                            test_ap=float(np.mean([r["test_ap"] for r in rr])),
                        )
                    )
                atomic_json(OUT / "summary.json", summary)
            atomic_json(
                OUT / "progress.json",
                dict(
                    status="complete",
                    completed=len(rows),
                    total=96,
                    updated_at=time.time(),
                ),
            )
    except Exception as error:
        atomic_json(
            OUT / "progress.json",
            dict(status="failed", error=repr(error), updated_at=time.time()),
        )
        raise


if __name__ == "__main__":
    main()
