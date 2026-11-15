"""One dataset representation per CTDG/TKG dataset, independent of model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
OUT = ROOT / "data/preprocessed"
CTDG = ("wikipedia", "reddit", "mooc")
TKG = ("ICEWS14", "ICEWS05-15", "GDELT")
DEFAULT_MAX_EVENTS = 100_000
SAMPLING_STRATEGY = "chronological_prefix_v1"


def _temporal_split(times: np.ndarray) -> np.ndarray:
    """Split a chronological prefix without putting one timestamp in two splits."""
    size = len(times)
    boundaries = []
    for fraction in (.70, .85):
        boundary = min(int(size * fraction), size)
        while boundary < size and boundary > 0 and times[boundary] == times[boundary - 1]:
            boundary += 1
        boundaries.append(boundary)
    split = np.full(size, 2, dtype=np.uint8)
    split[:boundaries[1]] = 1
    split[:boundaries[0]] = 0
    return split


def preprocess_ctdg(name: str, max_events: int = DEFAULT_MAX_EVENTS) -> Path:
    source = RAW / "ctdg" / f"{name}.csv"
    target = OUT / "ctdg" / name
    metadata_path = target / "metadata.json"
    if metadata_path.is_file():
        existing = json.loads(metadata_path.read_text())
        if (existing.get("max_events") == max_events
                and existing.get("sampling_strategy") == SAMPLING_STRATEGY):
            return target
    target.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(source, skiprows=1, header=None, dtype=np.float64)
    all_values = frame.to_numpy()
    if np.any(np.diff(all_values[:, 2]) < 0):
        raise ValueError(f"{source} is not chronological")
    values = all_values[:max_events]
    split = _temporal_split(values[:, 2])
    src = values[:, 0].astype(np.int32) + 1
    dst = values[:, 1].astype(np.int32) + int(values[:, 0].max()) + 2
    times = values[:, 2].astype(np.float64)
    labels = values[:, 3].astype(np.float32)
    features = values[:, 4:].astype(np.float32)
    edge_features = np.zeros((len(times) + 1, 172), dtype=np.float32)
    edge_features[1:, :features.shape[1]] = features
    node_features = np.zeros((max(src.max(), dst.max()) + 1, 172), dtype=np.float32)
    np.savez(target / "events.npz", src=src, dst=dst, time=times, label=labels,
             edge_id=np.arange(1, len(times) + 1, dtype=np.int32), split=split)
    np.save(target / "node_features.npy", node_features)
    np.save(target / "edge_features.npy", edge_features)
    metadata = {"family": "ctdg", "dataset": name, "num_events": len(times),
                "source_num_events": len(frame), "max_events": max_events,
                "sampling_strategy": SAMPLING_STRATEGY,
                "num_nodes": len(node_features) - 1, "feature_dim": 172,
                "split_counts": np.bincount(split, minlength=3).tolist(),
                "source": str(source.relative_to(ROOT))}
    (target / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"CTDG {name}: {len(times):,} events, splits {metadata['split_counts']}", flush=True)
    return target


def _icews_rows(folder: Path, split: str, entities: dict, relations: dict, timestamps: dict) -> np.ndarray:
    rows = []
    with (folder / f"{split}.txt").open() as source:
        for line in source:
            subject, relation, obj, stamp = line.rstrip("\n").split("\t")
            rows.append((entities[subject], relations[relation], entities[obj], timestamps[stamp]))
    return np.asarray(rows, dtype=np.int32)


def preprocess_tkg(name: str, max_events: int = DEFAULT_MAX_EVENTS) -> Path:
    source = RAW / "tkg" / name
    target = OUT / "tkg" / name
    metadata_path = target / "metadata.json"
    if metadata_path.is_file():
        existing = json.loads(metadata_path.read_text())
        if (existing.get("max_events") == max_events
                and existing.get("sampling_strategy") == SAMPLING_STRATEGY):
            return target
    target.mkdir(parents=True, exist_ok=True)
    if name == "GDELT":
        num_entities, num_relations = map(int, (source / "stat.txt").read_text().split()[:2])
        entities = {str(i): i for i in range(num_entities)}
        relations = {str(i): i for i in range(num_relations)}
        read_split = lambda split: np.loadtxt(source / f"{split}.txt", usecols=(0, 1, 2, 3), dtype=np.int32)
        time_step = 15
    else:
        entities = json.loads((source / "entity2id.json").read_text())
        relations = json.loads((source / "relation2id.json").read_text())
        timestamps = json.loads((source / "ts2id.json").read_text())
        read_split = lambda split: _icews_rows(source, split, entities, relations, timestamps)
        time_step = 1
    parts = [read_split(split) for split in ("train", "valid", "test")]
    for part in parts:
        if np.any(np.diff(part[:, 3]) < 0):
            raise ValueError(f"{source} is not chronological")
    if not (parts[0][-1, 3] < parts[1][0, 3] and parts[1][-1, 3] < parts[2][0, 3]):
        raise ValueError(f"{source} has overlapping train/valid/test times")
    source_counts = [len(part) for part in parts]
    all_facts = np.concatenate(parts)
    if len(all_facts) > max_events:
        facts = all_facts[:max_events]
        split = _temporal_split(facts[:, 3])
    else:
        facts = all_facts
        split = np.repeat(np.arange(3, dtype=np.uint8), source_counts)
    np.save(target / "facts.npy", facts)
    np.save(target / "split.npy", split)
    (target / "entity2id.json").write_text(json.dumps(entities))
    (target / "relation2id.json").write_text(json.dumps(relations))
    metadata = {"family": "tkg", "dataset": name, "num_facts": len(facts),
                "num_entities": len(entities), "num_relations": len(relations),
                "time_step": time_step,
                "split_counts": np.bincount(split, minlength=3).tolist(),
                "source_num_facts": sum(source_counts), "max_events": max_events,
                "source_split_counts": source_counts,
                "sampling_strategy": SAMPLING_STRATEGY,
                "source": str(source.relative_to(ROOT))}
    (target / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"TKG {name}: {len(facts):,} facts, splits {metadata['split_counts']}", flush=True)
    return target


def preprocess(family: str, name: str, max_events: int = DEFAULT_MAX_EVENTS) -> Path:
    if family == "ctdg" and name in CTDG:
        return preprocess_ctdg(name, max_events)
    if family == "tkg" and name in TKG:
        return preprocess_tkg(name, max_events)
    raise ValueError((family, name))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("ctdg", "tkg", "all"), default="all")
    parser.add_argument("--dataset")
    parser.add_argument("--max-events", type=int, default=DEFAULT_MAX_EVENTS)
    args = parser.parse_args()
    if args.dataset:
        family = "ctdg" if args.dataset in CTDG else "tkg"
        preprocess(family, args.dataset, args.max_events)
    else:
        for family, names in (("ctdg", CTDG), ("tkg", TKG)):
            if args.family in (family, "all"):
                for name in names:
                    preprocess(family, name, args.max_events)


if __name__ == "__main__":
    main()
