"""Exact output/gradient/RNG and validation replay checks on real CTDG/TKG data."""

import argparse, copy, time, gc
from pathlib import Path
import numpy as np
import torch
from experiments.common.runtime import (
    gpu_guard,
    capture_rng,
    restore_rng,
    evaluate,
    atomic_json,
)
from experiments.robust.training import (
    ROOT,
    DEFAULT,
    read,
    data_for,
    model_for,
    build,
    features,
    train_job,
)
from experiments.transfer.cross_representation import typed_negatives
from models.baselines.ctdg.tgat import TGAT

OUT = ROOT / "experiments/robust/outputs/v2_tgn/repairs/tgat_speed"


def main():
    torch.set_num_threads(2)
    protocol = read(ROOT / "experiments/robust/outputs/v2_tgn/protocol.json")
    records = []
    with gpu_guard("cuda:0", 0):
        for target in ("wikipedia", "reddit", "mooc", "lastfm"):
            job = next(
                j
                for j in protocol["jobs"]
                if j["target"] == target
                and j["method"] == "finetune"
                and j["sources"]
                and j["config"]["epochs"] == 50
            )
            data = data_for(job)
            model = model_for(job, DEFAULT, data)
            d = data[target]
            neg = d.negatives(7)
            native = list(d.batches(d.indices(0), 2048))[5]
            ids = native[:512]
            model.bind(d)
            model.begin_batch(native)
            for train in (False, True):
                model.train(train)
                rng = capture_rng("cuda:0")
                answers = []
                grads = []
                states = []
                times = []
                for fast in (False, True):
                    TGAT.fast_history_enabled = fast
                    restore_rng(rng, "cuda:0")
                    model.zero_grad(set_to_none=True)
                    torch.cuda.synchronize()
                    t = time.monotonic()
                    z = features(model, d, ids, neg)
                    z.square().mean().backward()
                    torch.cuda.synchronize()
                    times.append(time.monotonic() - t)
                    answers.append(z.detach().cpu())
                    grads.append(
                        {
                            n: p.grad.detach().cpu().clone()
                            for n, p in model.named_parameters()
                            if p.grad is not None
                        }
                    )
                    states.append(capture_rng("cuda:0"))
                assert torch.equal(*answers), (target, train, "output")
                assert all(torch.equal(grads[0][n], grads[1][n]) for n in grads[0]), (
                    target,
                    train,
                    "gradient",
                )
                assert torch.equal(
                    states[0]["cuda"], states[1]["cuda"]
                ) and torch.equal(states[0]["torch"], states[1]["torch"])
                records.append(
                    dict(
                        target=target,
                        train=train,
                        output_exact=True,
                        gradient_exact=True,
                        rng_exact=True,
                        seconds=times,
                    )
                )
            for name in ("ICEWS14", "GDELT") if target == "wikipedia" else ():
                d = data[name]
                model.bind(d)
                native = list(d.batches(d.indices(0), 1024))[3]
                ids = native[:128]
                neg = typed_negatives(d, 7)
                model.begin_batch(native)
                model.train()
                rng = capture_rng("cuda:0")
                answers = []
                grads = []
                for fast in (False, True):
                    TGAT.fast_history_enabled = fast
                    restore_rng(rng, "cuda:0")
                    model.zero_grad(set_to_none=True)
                    z = model.score(ids, neg[ids])
                    z.square().mean().backward()
                    answers.append(z.detach().cpu())
                    grads.append(
                        {
                            n: p.grad.detach().cpu().clone()
                            for n, p in model.named_parameters()
                            if p.grad is not None
                        }
                    )
                assert torch.equal(*answers), name
                # Typed embedding gradients use atomic scatter accumulation; the fast path is disabled.
                assert type(model.encoders[name]) is not TGAT
                for n in grads[0]:
                    torch.testing.assert_close(
                        grads[0][n], grads[1][n], rtol=1e-4, atol=1e-5
                    )
                records.append(
                    dict(
                        target=name,
                        train=True,
                        output_exact=True,
                        optimization_disabled=True,
                    )
                )
            del model
            gc.collect()
            torch.cuda.empty_cache()
        # Selected validation replay from two methods on all four Targets.
        for target in ("wikipedia", "reddit", "mooc", "lastfm"):
            for method in ("finetune", "tfra"):
                j = next(
                    j
                    for j in protocol["jobs"]
                    if j["target"] == target
                    and j["method"] == method
                    and j["sources"]
                    and j["config"]["epochs"] == 5
                    and (
                        ROOT
                        / "experiments/robust/outputs/v2_tgn/jobs"
                        / j["id"]
                        / "best.pt"
                    ).exists()
                )
                saved = ROOT / "experiments/robust/outputs/v2_tgn/jobs" / j["id"]
                data = data_for(j)
                d = data[target]
                model = model_for(j, j["config"], data)
                model.load_state_dict(
                    torch.load(
                        saved / "best.pt", map_location="cuda:0", weights_only=False
                    )["model"]
                )
                model.bind(d)
                negative = d.negatives(100007)
                answers = []
                metrics = []
                for fast in (False, True):
                    TGAT.fast_history_enabled = fast
                    m, s = evaluate(model, d, 1, negative, j["config"])
                    answers.append(s)
                    metrics.append(m)
                assert np.array_equal(*answers), (target, method, "validation scores")
                assert metrics[0] == metrics[1]
                reference = read(saved / "result.json")
                assert (
                    abs(
                        metrics[0]["ap"]
                        - max(h["validation"]["ap"] for h in reference["history"])
                    )
                    < 1e-8
                )
                records.append(
                    dict(
                        target=target,
                        method=method,
                        validation_scores_exact=True,
                        validation_ap=metrics[0]["ap"],
                    )
                )
                del model
                gc.collect()
                torch.cuda.empty_cache()
        # Frozen branch cache hit and state-write invalidation.
        j = next(
            j
            for j in protocol["jobs"]
            if j["method"] == "refine" and j["target"] == "wikipedia"
        )
        data = data_for(j)
        m = model_for(j, DEFAULT, data)
        d = data[j["target"]]
        m.bind(d)
        m.eval()
        ids = list(d.batches(d.indices(0), 2048))[2][:128]
        neg = d.negatives(7)
        m.begin_batch(ids)
        with torch.no_grad():
            TGAT.fast_history_enabled = False
            a = m.score(ids, neg[ids])
            TGAT.fast_history_enabled = True
            b = m.score(ids, neg[ids])
            c = m.score(ids, neg[ids])
            assert torch.equal(a, b) and torch.equal(b, c)
            state = copy.deepcopy(m.source.state_dict())
            m.source.load_state_dict(state)
            e = m.score(ids, neg[ids])
            assert torch.equal(a, e)
        records.append(dict(frozen_cache_exact=True, state_reload_invalidates=True))
        TGAT.fast_history_enabled = True
    atomic_json(OUT / "validation.json", dict(status="passed", records=records))
    print("Passed", len(records), "real-data equivalence checks", flush=True)


if __name__ == "__main__":
    main()
