#!/usr/bin/env python3
"""
DICOM Series Manifest Generator

Scans DICOM dataset directories and generates manifest CSV files
for conversion processing. Supports sharded output for distributed processing.

Features:
- Automatic detection of Images directories
- Series-level file counting
- Hash-based sharding for load balancing
- Support for multiple dataset versions

Environment Variables:
- OUT_DIR: Output directory for manifest files
- NUM_SHARDS: Number of shards to split manifest into (default: 16)
- MAX_DEPTH: Maximum directory depth to search for Images folder (default: 3)
- DATASET_VERSION: Dataset version to process (v1, v2, or all)

Directory Structure Expected:
- v1: /dataset_v1/{study}/Images/{series}/
- v2: /dataset_v2/Images/{hash}/Images/{series}/
"""

import csv
import hashlib
import json
import os
import time
from typing import Optional


def _hash(text: str) -> str:
    """Generate short hash from text."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


def _find_images_dir(base_dir: str, max_depth: int) -> Optional[str]:
    """Find 'Images' directory within max_depth levels."""
    base_depth = base_dir.rstrip(os.sep).count(os.sep)
    for root, dirs, _ in os.walk(base_dir):
        depth = root.count(os.sep) - base_depth
        if depth > max_depth:
            dirs[:] = []
            continue
        for d in dirs:
            if d.lower() == "images":
                return os.path.join(root, d)
    return None


def _count_files(path: str) -> int:
    """Count files in directory."""
    try:
        with os.scandir(path) as it:
            return sum(1 for entry in it if entry.is_file())
    except FileNotFoundError:
        return 0


def _series_dirs_v1(root: str, max_depth: int) -> list[str]:
    """Find series directories for v1 dataset structure."""
    series = []
    if not os.path.isdir(root):
        return series
    for name in sorted(os.listdir(root)):
        top_dir = os.path.join(root, name)
        if not os.path.isdir(top_dir):
            continue
        images_dir = _find_images_dir(top_dir, max_depth)
        if images_dir is None:
            continue
        subdirs = []
        try:
            with os.scandir(images_dir) as it:
                for entry in it:
                    if entry.is_dir():
                        subdirs.append(entry.path)
        except FileNotFoundError:
            continue
        for subdir in sorted(subdirs):
            series.append(subdir)
    return series


def _series_dirs_v2(root: str) -> list[str]:
    """Find series directories for v2 dataset structure.

    v2 structure: /dataset_v2/Images/{hash}/Images/{series_uid}/
    """
    series = []
    images_root = os.path.join(root, "Images")
    if not os.path.isdir(images_root):
        return series

    for hash_name in sorted(os.listdir(images_root)):
        hash_dir = os.path.join(images_root, hash_name)
        if not os.path.isdir(hash_dir):
            continue
        inner_images = os.path.join(hash_dir, "Images")
        if not os.path.isdir(inner_images):
            continue
        try:
            for series_name in sorted(os.listdir(inner_images)):
                series_dir = os.path.join(inner_images, series_name)
                if os.path.isdir(series_dir):
                    series.append(series_dir)
        except FileNotFoundError:
            continue
    return series


def _shard_id(series_hash: str, num_shards: int) -> int:
    """Calculate shard ID from hash."""
    return int(series_hash, 16) % num_shards


def main() -> None:
    """Main entry point."""
    # Configuration from environment
    out_dir = os.environ.get("OUT_DIR", "/data/manifests")
    num_shards = int(os.environ.get("NUM_SHARDS", "16"))
    max_depth = int(os.environ.get("MAX_DEPTH", "3"))
    dataset_version = os.environ.get("DATASET_VERSION", "all").lower()

    # Dataset roots - customize these for your environment
    dataset_roots = {}
    v1_root = os.environ.get("DATASET_V1_ROOT", "/data/cad_research/personal/zeron/VEGA_CT_DP/dataset_v1")
    v2_root = os.environ.get("DATASET_V2_ROOT", "/data/cad_research/personal/zeron/VEGA_CT_DP/dataset_v2")

    if dataset_version in ("v1", "all") and os.path.isdir(v1_root):
        dataset_roots["v1"] = v1_root
    if dataset_version in ("v2", "all") and os.path.isdir(v2_root):
        dataset_roots["v2"] = v2_root

    os.makedirs(out_dir, exist_ok=True)

    manifest_csv = os.path.join(out_dir, "series_all.csv")
    manifest_txt = os.path.join(out_dir, "series_all_paths.txt")
    shards_dir = os.path.join(out_dir, "shards")

    os.makedirs(shards_dir, exist_ok=True)

    # Initialize shard files
    shard_files = []
    shard_writers = []
    shard_counts = [0 for _ in range(num_shards)]

    for shard_idx in range(num_shards):
        shard_path = os.path.join(shards_dir, f"series_shard_{shard_idx:02d}.csv")
        shard_file = open(shard_path, "w", newline="")
        shard_files.append(shard_file)
        writer = csv.DictWriter(
            shard_file,
            fieldnames=["dataset", "series_dir", "file_count", "series_hash"],
        )
        writer.writeheader()
        shard_writers.append(writer)

    total = 0
    start_ts = time.time()

    with open(manifest_csv, "w", newline="") as mf, open(manifest_txt, "w") as mt:
        manifest_writer = csv.DictWriter(
            mf, fieldnames=["dataset", "series_dir", "file_count", "series_hash"]
        )
        manifest_writer.writeheader()

        for dataset, root in dataset_roots.items():
            # Use appropriate directory finder based on dataset version
            if dataset == "v2":
                series_dirs = _series_dirs_v2(root)
            else:
                series_dirs = _series_dirs_v1(root, max_depth)

            print(f"dataset={dataset} series_dirs_count={len(series_dirs)}")

            for series_dir in series_dirs:
                file_count = _count_files(series_dir)
                series_hash = _hash(series_dir)
                row = {
                    "dataset": dataset,
                    "series_dir": series_dir,
                    "file_count": file_count,
                    "series_hash": series_hash,
                }
                manifest_writer.writerow(row)
                mt.write(series_dir + "\n")
                shard_idx = _shard_id(series_hash, num_shards)
                shard_writers[shard_idx].writerow(row)
                shard_counts[shard_idx] += 1
                total += 1
                if total % 1000 == 0:
                    elapsed = time.time() - start_ts
                    print(f"progress={total} elapsed_sec={elapsed:.1f}")

    for f in shard_files:
        f.close()

    summary = {
        "total_series": total,
        "num_shards": num_shards,
        "shard_counts": shard_counts,
        "manifest_csv": manifest_csv,
        "manifest_txt": manifest_txt,
        "shards_dir": shards_dir,
        "datasets": list(dataset_roots.keys()),
    }

    summary_json = os.path.join(out_dir, "manifest_summary.json")
    with open(summary_json, "w") as f:
        json.dump(summary, f, ensure_ascii=True, indent=2)

    print(f"manifest_csv={manifest_csv}")
    print(f"manifest_txt={manifest_txt}")
    print(f"summary_json={summary_json}")
    print(f"total_series={total}")


if __name__ == "__main__":
    main()
