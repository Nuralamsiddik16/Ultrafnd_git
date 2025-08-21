#!/usr/bin/env python3
"""Extract a small subset of the FakeSV dataset for quick experiments.

Usage:
    python scripts/mini_dataset.py --input-dir /path/to/FakeSV --output-dir ./mini_fake --num 50

The input directory must contain a ``data_complete.json`` file as described in
``FakeSVRawDataset``.  The script writes the first ``N`` entries to the output
folder and, if present, copies the corresponding files from ``video_comment``
and ``videos`` subdirectories.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Iterable, List, Dict, Any


def _load_records(path: Path) -> List[Dict[str, Any]]:
    """Load records from a JSON array or JSONL file."""
    if not path.exists():
        raise FileNotFoundError(f"data_complete.json not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            return json.load(f)
        return [json.loads(line) for line in f if line.strip()]


def _save_records(records: Iterable[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(list(records), f, ensure_ascii=False, indent=2)


def _copy_optional(src_dir: Path, dst_dir: Path, ids: Iterable[str], ext: str) -> None:
    if not src_dir.exists():
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    for vid in ids:
        src = src_dir / f"{vid}{ext}"
        if src.exists():
            shutil.copy2(src, dst_dir / src.name)


def main() -> None:
    p = argparse.ArgumentParser(description="Create a smaller FakeSV subset")
    p.add_argument("--input-dir", required=True, help="Path to the full dataset root")
    p.add_argument("--output-dir", required=True, help="Where to write the subset")
    p.add_argument("--num", type=int, default=100, help="Number of records to keep")
    args = p.parse_args()

    in_root = Path(args.input_dir)
    out_root = Path(args.output_dir)

    records = _load_records(in_root / "data_complete.json")
    subset = records[: args.num]
    _save_records(subset, out_root / "data_complete.json")

    ids = [r.get("video_id") or f"rec_{i}" for i, r in enumerate(subset)]
    _copy_optional(in_root / "video_comment", out_root / "video_comment", ids, ".json")
    _copy_optional(in_root / "videos", out_root / "videos", ids, ".mp4")

    print(f"Saved {len(subset)} records to {out_root}")


if __name__ == "__main__":
    main()
