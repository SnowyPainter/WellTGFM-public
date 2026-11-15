"""Independent CUDA processes; identical jobs, seeds, batches and checkpoint selection."""

import argparse, concurrent.futures, contextlib, fcntl, gc, multiprocessing, os, time, traceback
from pathlib import Path
import torch
from experiments.common.runtime import atomic_json, gpu_guard
from experiments.robust.training import ROOT, read, train_job
from experiments.robust.evaluation import evaluate_job
from experiments.robust.run import initialize
from experiments.robust.report import summarize
from experiments.robust.design import OUT


def worker(job, out_string, fast=True, evaluate=True):
    from models.baselines.ctdg.tgat import TGAT

    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    TGAT.fast_history_enabled = fast
    out = Path(out_string)
    root = out / "workers" / job["id"]
    root.mkdir(parents=True, exist_ok=True)
    (out / "jobs").mkdir(exist_ok=True)
    if not (root / "jobs").exists():
        (root / "jobs").symlink_to((out / "jobs").resolve(), target_is_directory=True)
    try:
        with (root / "run.log").open(
            "a", buffering=1
        ) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            start = time.monotonic()
            result = train_job(job, root)
            if evaluate:
                evaluate_job(job, out)
            record = dict(
                id=job["id"],
                status="complete",
                wall_seconds=time.monotonic() - start,
                pid=os.getpid(),
            )
    except Exception as error:
        record = dict(
            id=job["id"],
            status="failed",
            error=repr(error),
            traceback=traceback.format_exc(),
            pid=os.getpid(),
        )
    atomic_json(root / "execution.json", record)
    gc.collect()
    torch.cuda.empty_cache()
    return record


def complete(j, out):
    return all(
        (out / "jobs" / j["id"] / f"test_budget{b}.json").exists()
        for b in j["config"].get("save_budgets", [j["config"]["epochs"]])
    )


def execute(out, workers):
    p = initialize(out)
    index = {j["id"]: j for j in p["jobs"]}
    budget_ids = list(dict.fromkeys(v["job_id"] for v in p["stages"]["budget"]))
    previous = {
        v["job_id"]
        for stage, vs in p["stages"].items()
        if stage != "budget"
        for v in vs
    }
    assert all(
        complete(index[i], out) for i in previous
    ), "Earlier stages must be complete"
    with (out / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with gpu_guard("cuda:0", 0):
            done = {i for i, j in index.items() if complete(j, out)}
            pending = [i for i in budget_ids if i not in done]
            failures = {}
            retry = []
            started = time.time()

            def progress(active):
                states = []
                for i in active:
                    h = out / "workers" / i / "heartbeat.json"
                    states.append(
                        dict(job=index[i], heartbeat=read(h) if h.exists() else None)
                    )
                atomic_json(
                    out / "progress.json",
                    dict(
                        status="running",
                        stage="budget",
                        execution="parallel",
                        workers=workers,
                        stage_index=sum(i in done for i in budget_ids) + 1,
                        stage_jobs=len(budget_ids),
                        completed=len(done),
                        failed=len(failures),
                        total_jobs=len(index),
                        job=index[next(iter(active))] if active else None,
                        active_jobs=states,
                        started_at=started,
                        updated_at=time.time(),
                    ),
                )

            with concurrent.futures.ProcessPoolExecutor(
                max_workers=workers, mp_context=multiprocessing.get_context("spawn")
            ) as pool:
                active = {}
                iterator = iter(pending)

                def fill():
                    while len(active) < workers:
                        ident = next(iterator, None)
                        if ident is None:
                            break
                        active[pool.submit(worker, index[ident], str(out))] = ident

                fill()
                while active:
                    progress(active.values())
                    finished, _ = concurrent.futures.wait(
                        active,
                        timeout=10,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    for future in finished:
                        ident = active.pop(future)
                        try:
                            r = future.result()
                        except Exception as e:
                            r = dict(id=ident, status="failed", error=repr(e))
                        print(r, flush=True)
                        if r["status"] == "complete":
                            done.add(ident)
                        else:
                            retry.append(ident)
                            atomic_json(
                                out / "workers" / ident / "parallel_failure.json", r
                            )
                    if finished:
                        summarize(p, out)
                    fill()
            # Memory-pressure failures are retried with a single isolated worker from saved epochs.
            if retry:
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=1, mp_context=multiprocessing.get_context("spawn")
                ) as pool:
                    for ident in retry:
                        progress([ident])
                        r = pool.submit(worker, index[ident], str(out)).result()
                        print(r, flush=True)
                        if r["status"] == "complete":
                            done.add(ident)
                        else:
                            failures[ident] = r
            summarize(p, out)
            atomic_json(out / "failures.json", failures)
            atomic_json(
                out / "progress.json",
                dict(
                    status="completed_with_failures" if failures else "complete",
                    completed=len(done),
                    failed=len(failures),
                    total_jobs=len(index),
                    execution="parallel",
                    updated_at=time.time(),
                ),
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    execute(args.out.resolve(), args.workers)
