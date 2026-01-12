#!/usr/bin/env python3
"""
Generate manifest grouped by SeriesInstanceUID.

Scans DICOM directories and creates a manifest where each row represents
a unique SeriesInstanceUID, not a directory path. This properly handles
cases where multiple series exist in the same directory.

Features:
- Groups DICOM files by SeriesInstanceUID
- Extracts series metadata (description, kernel, orientation)
- Filters out non-CT modalities and LOCALIZER images
- Generates sharded CSV files for parallel processing

Environment Variables:
- DATASET_ROOT: Root directory containing DICOM files
- DATASET_NAME: Name identifier for the dataset (e.g., "v1", "v2")
- OUTPUT_DIR: Directory to write manifest files
- NUM_SHARDS: Number of shards to create (default: 16)
"""

import csv
import hashlib
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import pydicom

# Image types to skip (X-ray scout images)
SKIP_IMAGE_TYPES = {"LOCALIZER", "SCOUT", "TOPOGRAM"}

# Modalities to include (CT only)
INCLUDE_MODALITIES = {"CT"}


def _hash(text: str) -> str:
    """Generate short hash from text."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


def _is_localizer(ds: pydicom.dataset.Dataset) -> bool:
    """Check if DICOM is a localizer/scout image."""
    img_type = getattr(ds, "ImageType", [])
    if not img_type:
        return False
    for t in img_type:
        if isinstance(t, str) and t.upper() in SKIP_IMAGE_TYPES:
            return True
    return False


def _scan_directory(dir_path: str) -> list[dict[str, Any]]:
    """Scan a directory for DICOM files and extract series info."""
    series_files = defaultdict(list)
    series_meta = {}

    try:
        for entry in os.scandir(dir_path):
            if not entry.is_file():
                continue
            try:
                ds = pydicom.dcmread(entry.path, force=True, stop_before_pixels=True)

                # Skip non-CT modalities
                modality = getattr(ds, "Modality", "")
                if modality not in INCLUDE_MODALITIES:
                    continue

                # Skip LOCALIZER images
                if _is_localizer(ds):
                    continue

                series_uid = getattr(ds, "SeriesInstanceUID", None)
                if not series_uid:
                    continue

                series_files[series_uid].append(entry.path)

                # Store metadata from first file of each series
                if series_uid not in series_meta:
                    series_meta[series_uid] = {
                        "series_uid": series_uid,
                        "series_description": getattr(ds, "SeriesDescription", ""),
                        "image_type": list(getattr(ds, "ImageType", [])),
                        "kernel": str(getattr(ds, "ConvolutionKernel", "")),
                        "study_uid": getattr(ds, "StudyInstanceUID", ""),
                        "patient_id": getattr(ds, "PatientID", ""),
                        "modality": modality,
                    }
            except Exception:
                continue
    except Exception:
        return []

    results = []
    for series_uid, files in series_files.items():
        meta = series_meta[series_uid]
        results.append({
            "series_uid": series_uid,
            "series_dir": dir_path,
            "file_count": len(files),
            "series_description": meta["series_description"],
            "image_type": "|".join(meta["image_type"]),
            "kernel": meta["kernel"],
            "study_uid": meta["study_uid"],
            "patient_id": meta["patient_id"],
        })

    return results


def _find_dicom_dirs(root: str) -> list[str]:
    """Find all directories containing DICOM files."""
    dicom_dirs = set()

    for dirpath, _, filenames in os.walk(root):
        # Check if directory has any files (potential DICOM files)
        if filenames:
            dicom_dirs.add(dirpath)

    return sorted(dicom_dirs)


def main():
    dataset_root = os.environ.get("DATASET_ROOT", "")
    dataset_name = os.environ.get("DATASET_NAME", "unknown")
    output_dir = os.environ.get("OUTPUT_DIR", "/tmp/manifests")
    num_shards = int(os.environ.get("NUM_SHARDS", "16"))
    num_workers = int(os.environ.get("NUM_WORKERS", "16"))

    if not dataset_root:
        print("ERROR: DATASET_ROOT environment variable is required")
        sys.exit(1)

    print(f"dataset_root={dataset_root}")
    print(f"dataset_name={dataset_name}")
    print(f"output_dir={output_dir}")
    print(f"num_shards={num_shards}")
    print(f"num_workers={num_workers}")

    os.makedirs(output_dir, exist_ok=True)
    shards_dir = os.path.join(output_dir, "shards")
    os.makedirs(shards_dir, exist_ok=True)

    # Find all potential DICOM directories
    print("Scanning for DICOM directories...")
    dicom_dirs = _find_dicom_dirs(dataset_root)
    print(f"Found {len(dicom_dirs)} directories to scan")

    # Scan directories in parallel
    all_series = []
    completed = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_scan_directory, d): d for d in dicom_dirs}
        for future in as_completed(futures):
            completed += 1
            if completed % 100 == 0:
                print(f"Scanned {completed}/{len(dicom_dirs)} directories, found {len(all_series)} series")
            try:
                results = future.result()
                all_series.extend(results)
            except Exception as e:
                print(f"Error scanning directory: {e}")

    print(f"Total series found: {len(all_series)}")

    # Add dataset name and series_hash
    for item in all_series:
        item["dataset"] = dataset_name
        item["series_hash"] = _hash(item["series_uid"])

    # Write all series to master manifest
    master_csv = os.path.join(output_dir, "series_all.csv")
    fieldnames = [
        "dataset", "series_uid", "series_hash", "series_dir", "file_count",
        "series_description", "image_type", "kernel", "study_uid", "patient_id"
    ]

    with open(master_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in all_series:
            writer.writerow({k: item.get(k, "") for k in fieldnames})

    print(f"Master manifest: {master_csv}")

    # Create sharded manifests
    shards = defaultdict(list)
    for item in all_series:
        shard_id = int(item["series_hash"], 16) % num_shards
        shards[shard_id].append(item)

    for shard_id in range(num_shards):
        shard_file = os.path.join(shards_dir, f"series_shard_{shard_id:02d}.csv")
        shard_items = shards.get(shard_id, [])
        with open(shard_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for item in shard_items:
                writer.writerow({k: item.get(k, "") for k in fieldnames})
        print(f"Shard {shard_id:02d}: {len(shard_items)} series -> {shard_file}")

    # Summary
    print("\n=== Summary ===")
    print(f"Total series: {len(all_series)}")
    print(f"Shards created: {num_shards}")

    # Count by series description
    desc_counts = defaultdict(int)
    for item in all_series:
        desc_counts[item["series_description"]] += 1

    print("\nTop 10 series descriptions:")
    for desc, count in sorted(desc_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {desc}: {count}")


if __name__ == "__main__":
    main()
