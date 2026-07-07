#!/usr/bin/env python3
"""Permanently rename ``states`` -> ``observation.state`` in a LeRobot dataset.

Updates:
  - meta/info.json feature key
  - meta/stats.json (from existing entry, fallback file, or recomputed from parquet)
  - meta/episodes/**/*.parquet columns prefixed with ``stats/states/``
  - data/**/*.parquet column ``states``
  - meta/modality.json ``original_key`` references (if present)

Example:
    uv run python rename_states_to_observation_state.py \\
        --dataset-root dataset/G1WholebodyLocomotionPickBetweenTablesTeleop-v0 \\
        --fallback-stats dataset/G1WholebodyLocomotionPickBetweenTablesTeleop-v0_old/meta/stats.json
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

OLD_KEY = "states"
NEW_KEY = "observation.state"


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _save_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=4)
        file.write("\n")


def _backup(path: Path) -> None:
    backup_path = path.with_suffix(path.suffix + ".bak")
    if backup_path.exists():
        return
    shutil.copy2(path, backup_path)


def rename_info_features(info_path: Path, *, dry_run: bool) -> bool:
    info = _load_json(info_path)
    features = info.get("features", {})
    if OLD_KEY not in features:
        if NEW_KEY in features:
            print(f"[info] already has {NEW_KEY!r} in {info_path}")
            return False
        raise KeyError(f"Feature {OLD_KEY!r} not found in {info_path}")

    feature = features.pop(OLD_KEY)
    shape = feature.get("shape")
    if shape == [-1]:
        print("[info] feature shape is [-1]; keeping as-is (runtime inferred from parquet).")

    features[NEW_KEY] = feature
    info["features"] = features

    if dry_run:
        print(f"[dry-run] would rename feature {OLD_KEY!r} -> {NEW_KEY!r} in {info_path}")
        return True

    _backup(info_path)
    _save_json(info_path, info)
    print(f"[info] renamed feature in {info_path}")
    return True


def _stats_from_fallback(fallback_path: Path) -> dict:
    stats = _load_json(fallback_path)
    if OLD_KEY not in stats:
        raise KeyError(f"{OLD_KEY!r} not found in fallback stats: {fallback_path}")
    return stats[OLD_KEY]


def _compute_stats_from_parquet(data_files: Iterable[Path]) -> dict:
    arrays: list[np.ndarray] = []
    for path in data_files:
        table = pq.read_table(path, columns=[OLD_KEY])
        column = table[OLD_KEY].to_pylist()
        arrays.append(np.asarray(column, dtype=np.float32))

    if not arrays:
        raise FileNotFoundError("No parquet data files found to compute stats.")

    values = np.concatenate(arrays, axis=0)
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "count": [int(values.shape[0])],
    }


def update_stats(
    stats_path: Path,
    *,
    fallback_stats_path: Path | None,
    data_files: list[Path],
    dry_run: bool,
) -> bool:
    stats = _load_json(stats_path) if stats_path.exists() else {}

    if NEW_KEY in stats:
        print(f"[stats] already has {NEW_KEY!r} in {stats_path}")
        return False

    if OLD_KEY in stats:
        state_stats = stats.pop(OLD_KEY)
        source = f"existing {stats_path}"
    elif fallback_stats_path is not None and fallback_stats_path.exists():
        state_stats = _stats_from_fallback(fallback_stats_path)
        source = f"fallback {fallback_stats_path}"
    else:
        state_stats = _compute_stats_from_parquet(data_files)
        source = "recomputed from parquet"

    stats[NEW_KEY] = state_stats

    if dry_run:
        print(f"[dry-run] would add {NEW_KEY!r} to {stats_path} (source: {source})")
        return True

    stats_path.parent.mkdir(parents=True, exist_ok=True)
    if stats_path.exists():
        _backup(stats_path)
    _save_json(stats_path, stats)
    print(f"[stats] wrote {NEW_KEY!r} to {stats_path} (source: {source})")
    return True


def _scalar_feature_keys(info_path: Path) -> set[str]:
    info = _load_json(info_path)
    return {
        key
        for key, feature in info.get("features", {}).items()
        if tuple(feature.get("shape", ())) == (1,) and feature.get("dtype") != "string"
    }


def fix_scalar_list_columns(data_dir: Path, info_path: Path, *, dry_run: bool) -> int:
    """Unwrap length-1 list columns that should be scalar per info.json."""
    scalar_keys = _scalar_feature_keys(info_path)
    if not scalar_keys:
        return 0

    updated = 0
    for path in sorted(data_dir.rglob("*.parquet")):
        schema = pq.read_schema(path)
        keys_to_fix = [
            key
            for key in scalar_keys
            if key in schema.names and pa.types.is_list(schema.field(key).type)
        ]
        if not keys_to_fix:
            continue

        if dry_run:
            print(f"[dry-run] would unwrap scalar list columns {keys_to_fix} in {path}")
            updated += 1
            continue

        table = pq.read_table(path)
        columns = {name: table[name] for name in table.column_names}
        for key in keys_to_fix:
            values = [row[0] if row is not None else None for row in columns[key].to_pylist()]
            columns[key] = pa.array(values, type=schema.field(key).type.value_type)
        _backup(path)
        pq.write_table(pa.table(columns), path)
        print(f"[data] unwrapped scalar list columns {keys_to_fix} in {path}")
        updated += 1

    return updated


def rename_data_parquet_columns(data_dir: Path, *, dry_run: bool) -> int:
    if not data_dir.exists():
        print(f"[data] skip missing directory: {data_dir}")
        return 0

    updated = 0
    for path in sorted(data_dir.rglob("*.parquet")):
        schema = pq.read_schema(path)
        if OLD_KEY not in schema.names:
            continue
        if NEW_KEY in schema.names:
            raise ValueError(f"{path} already contains both {OLD_KEY!r} and {NEW_KEY!r}")

        if dry_run:
            print(f"[dry-run] would rename parquet column in {path}")
            updated += 1
            continue

        table = pq.read_table(path)
        columns = {name: table[name] for name in table.column_names}
        columns[NEW_KEY] = columns.pop(OLD_KEY)
        _backup(path)
        pq.write_table(pa.table(columns), path)
        print(f"[data] renamed column in {path}")
        updated += 1

    return updated


def rename_episode_stats_columns(episodes_dir: Path, *, dry_run: bool) -> int:
    if not episodes_dir.exists():
        print(f"[episodes] skip missing directory: {episodes_dir}")
        return 0

    old_prefix = f"stats/{OLD_KEY}/"
    new_prefix = f"stats/{NEW_KEY}/"
    updated_files = 0

    for path in sorted(episodes_dir.rglob("*.parquet")):
        schema = pq.read_schema(path)
        rename_map = {
            name: name.replace(old_prefix, new_prefix, 1)
            for name in schema.names
            if name.startswith(old_prefix)
        }
        if not rename_map:
            continue

        if dry_run:
            print(f"[dry-run] would rename episode stats columns in {path}")
            updated_files += 1
            continue

        table = pq.read_table(path)
        columns = {}
        for name in table.column_names:
            columns[rename_map.get(name, name)] = table[name]
        _backup(path)
        pq.write_table(pa.table(columns), path)
        print(f"[episodes] renamed stats columns in {path}")
        updated_files += 1

    return updated_files


def update_modality(modality_path: Path, *, dry_run: bool) -> bool:
    if not modality_path.exists():
        print(f"[modality] skip missing file: {modality_path}")
        return False

    modality = _load_json(modality_path)
    changed = False

    for section in modality.values():
        if not isinstance(section, dict):
            continue
        for entry in section.values():
            if not isinstance(entry, dict):
                continue
            if entry.get("original_key") == OLD_KEY:
                entry["original_key"] = NEW_KEY
                changed = True

    if not changed:
        print(f"[modality] no {OLD_KEY!r} references in {modality_path}")
        return False

    if dry_run:
        print(f"[dry-run] would update modality references in {modality_path}")
        return True

    _backup(modality_path)
    _save_json(modality_path, modality)
    print(f"[modality] updated {modality_path}")
    return True


def verify(dataset_root: Path) -> None:
    info = _load_json(dataset_root / "meta/info.json")
    assert NEW_KEY in info["features"], "info.json missing observation.state"
    assert OLD_KEY not in info["features"], "info.json still contains states"

    stats_path = dataset_root / "meta/stats.json"
    if stats_path.exists():
        stats = _load_json(stats_path)
        assert NEW_KEY in stats, "stats.json missing observation.state"

    data_files = sorted((dataset_root / "data").rglob("*.parquet"))
    assert data_files, "no data parquet files found"
    schema = pq.read_schema(data_files[0])
    assert NEW_KEY in schema.names, "data parquet missing observation.state"
    assert OLD_KEY not in schema.names, "data parquet still contains states"

    print("[verify] ok")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("dataset/G1WholebodyLocomotionPickBetweenTablesTeleop-v0"),
        help="Root of the LeRobot dataset (contains meta/, data/, videos/).",
    )
    parser.add_argument(
        "--fallback-stats",
        type=Path,
        default=Path("dataset/G1WholebodyLocomotionPickBetweenTablesTeleop-v0_old/meta/stats.json"),
        help="Optional stats.json containing legacy `states` stats.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned changes without modifying files.",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip post-run verification.",
    )
    parser.add_argument(
        "--skip-scalar-fix",
        action="store_true",
        help="Do not unwrap length-1 list columns that should be scalar.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    meta_dir = dataset_root / "meta"
    info_path = meta_dir / "info.json"
    stats_path = meta_dir / "stats.json"
    modality_path = meta_dir / "modality.json"
    data_dir = dataset_root / "data"
    episodes_dir = meta_dir / "episodes"

    data_files = sorted(data_dir.rglob("*.parquet"))
    fallback_stats = args.fallback_stats.resolve() if args.fallback_stats else None

    print(f"Dataset: {dataset_root}")
    print(f"Mode: {'dry-run' if args.dry_run else 'apply'}")

    rename_info_features(info_path, dry_run=args.dry_run)
    update_stats(
        stats_path,
        fallback_stats_path=fallback_stats,
        data_files=data_files,
        dry_run=args.dry_run,
    )
    n_data = rename_data_parquet_columns(data_dir, dry_run=args.dry_run)
    n_scalar = 0
    if not args.skip_scalar_fix:
        n_scalar = fix_scalar_list_columns(data_dir, info_path, dry_run=args.dry_run)
    n_episodes = rename_episode_stats_columns(episodes_dir, dry_run=args.dry_run)
    update_modality(modality_path, dry_run=args.dry_run)

    print(
        "Summary: "
        f"data_files_updated={n_data}, "
        f"scalar_fix_files_updated={n_scalar}, "
        f"episode_files_updated={n_episodes}"
    )

    if not args.dry_run and not args.skip_verify:
        verify(dataset_root)


if __name__ == "__main__":
    main()
