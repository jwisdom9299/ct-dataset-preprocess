#!/usr/bin/env python3
"""
DICOM to Zarr Volume Converter

Converts DICOM series to Zarr volumes with SQLite/directory/zip storage.
Supports sharded processing for large-scale conversion jobs.

Features:
- LOCALIZER/SCOUT/TOPOGRAM image filtering (X-ray images excluded)
- DERIVED/MPR image filtering (keeps only ORIGINAL axial slices)
- Multiple storage backends (SQLite, DirectoryStore, ZipStore)
- Blosc compression with configurable settings
- Parallel processing support
- Comprehensive metadata extraction
- Slice-level verification

Environment Variables:
- INPUT_CSV: Path to CSV with series information
- OUT_ROOT: Output directory root
- STORE_TYPE: Storage type (sqlite, dir, zip)
- NUM_WORKERS: Number of parallel workers (default: 1)
- SHARD_ID: Current shard ID for distributed processing
- NUM_SHARDS: Total number of shards
- OVERWRITE: Set to "1" to overwrite existing files
- VERIFY_SLICES: Number of slices to verify (default: 3)
- SUMMARY_SUFFIX: Suffix for summary files
"""

import csv
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import time
from typing import Any

import numpy as np
import pydicom
import zarr
from numcodecs import Blosc

DEFAULT_SAMPLE_CSV = "/data/cad_research/personal/zeron/VEGA_CT_DP/manifests/series_all.csv"
DEFAULT_OUT_ROOT = "/data/cad_research/personal/zeron/VEGA_CT_DP/converted"

# Tags that should always be stored per-slice
ALWAYS_SLICE_TAGS = {
    "SOPInstanceUID",
    "InstanceNumber",
    "ImagePositionPatient",
}

# Tags to extract at series level
SERIES_TAG_CANDIDATES = [
    "PatientID",
    "PatientName",
    "PatientSex",
    "PatientAge",
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "StudyDate",
    "SeriesDate",
    "StudyDescription",
    "SeriesDescription",
    "Modality",
    "Manufacturer",
    "ManufacturerModelName",
    "BodyPartExamined",
    "ProtocolName",
    "ConvolutionKernel",
    "KVP",
    "FrameOfReferenceUID",
    "Rows",
    "Columns",
    "PixelSpacing",
    "SliceThickness",
    "SpacingBetweenSlices",
    "ImageOrientationPatient",
    "PhotometricInterpretation",
    "BitsAllocated",
    "BitsStored",
    "HighBit",
    "PixelRepresentation",
    "SamplesPerPixel",
    "RescaleSlope",
    "RescaleIntercept",
    "RescaleType",
    "WindowCenter",
    "WindowWidth",
    "PixelPaddingValue",
    "PixelPaddingRangeLimit",
]

# Tags to extract at slice level
SLICE_TAG_CANDIDATES = [
    "SOPInstanceUID",
    "InstanceNumber",
    "ImagePositionPatient",
    "SliceLocation",
    "AcquisitionTime",
    "ContentTime",
    "TriggerTime",
]

ALL_TAGS = sorted(set(SERIES_TAG_CANDIDATES + SLICE_TAG_CANDIDATES))

# Image types to skip (LOCALIZER/SCOUT images are X-ray, not CT slices)
SKIP_IMAGE_TYPES = {"LOCALIZER", "SCOUT", "TOPOGRAM"}

# Skip DERIVED images (MPR reformats) - keep only ORIGINAL slices
SKIP_DERIVED = True  # Set to False to include DERIVED images


def _to_json_value(value: Any) -> Any:
    """Convert DICOM value to JSON-serializable format."""
    if value is None:
        return None
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="ignore")
        except Exception:
            return str(value)
    if isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "real") and hasattr(value, "imag") and not isinstance(value, complex):
        return float(value)
    if isinstance(value, (list, tuple)):
        return [_to_json_value(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def _hash(text: str) -> str:
    """Generate short hash from text."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def _list_files(series_dir: str) -> list[str]:
    """List all files in a directory."""
    files = []
    try:
        with os.scandir(series_dir) as it:
            for entry in it:
                if entry.is_file():
                    files.append(entry.path)
    except FileNotFoundError:
        return []
    return sorted(files)


def _get_tag(ds: pydicom.dataset.Dataset, tag: str) -> Any:
    """Get DICOM tag value."""
    return _to_json_value(getattr(ds, tag, None))


def _is_localizer(ds: pydicom.dataset.Dataset) -> bool:
    """Check if DICOM is a localizer/scout image (X-ray, not CT slice)."""
    img_type = getattr(ds, "ImageType", [])
    if not img_type:
        return False
    for t in img_type:
        if isinstance(t, str) and t.upper() in SKIP_IMAGE_TYPES:
            return True
    return False


def _is_derived(ds: pydicom.dataset.Dataset) -> bool:
    """Check if DICOM is a DERIVED image (MPR reformat, not original slice).

    DERIVED images like Coronal/Sagittal MPR reformats have different orientations
    and shouldn't be mixed with original axial slices.
    """
    if not SKIP_DERIVED:
        return False
    img_type = getattr(ds, "ImageType", [])
    if not img_type:
        return False
    # First element of ImageType indicates ORIGINAL vs DERIVED
    if len(img_type) > 0 and isinstance(img_type[0], str):
        if img_type[0].upper() == "DERIVED":
            return True
    return False


def _sort_key_values(z_pos: float | None, inst_num: int | None, path: str) -> tuple:
    """Generate sort key for slice ordering."""
    if z_pos is not None:
        return (0, z_pos, inst_num or 0, path)
    return (1, inst_num or 0, path)


def _compute_spacing(z_positions: list[float]) -> float | None:
    """Compute median Z-spacing from positions."""
    if len(z_positions) < 2:
        return None
    diffs = [abs(z_positions[i + 1] - z_positions[i]) for i in range(len(z_positions) - 1)]
    diffs = [d for d in diffs if d > 0]
    if not diffs:
        return None
    return float(np.median(diffs))


def _store_path(base_dir: str, series_hash: str, store_type: str) -> str:
    """Get output path for Zarr store."""
    if store_type == "dir":
        return os.path.join(base_dir, f"{series_hash}.zarr")
    if store_type == "zip":
        return os.path.join(base_dir, f"{series_hash}.zarr.zip")
    if store_type == "sqlite":
        return os.path.join(base_dir, f"{series_hash}.zarr.sqlite")
    raise ValueError(f"Unsupported store_type={store_type}")


def _create_store(path: str, store_type: str):
    """Create Zarr store."""
    if store_type == "dir":
        return zarr.DirectoryStore(path)
    if store_type == "zip":
        return zarr.ZipStore(path, mode="w")
    if store_type == "sqlite":
        return zarr.SQLiteStore(path)
    raise ValueError(f"Unsupported store_type={store_type}")


def _path_stats(path: str) -> tuple[int, int]:
    """Get file size and count statistics."""
    if os.path.isdir(path):
        total = 0
        count = 0
        for root, _, files in os.walk(path):
            for name in files:
                count += 1
                try:
                    total += os.path.getsize(os.path.join(root, name))
                except FileNotFoundError:
                    continue
        return total, count
    if os.path.exists(path):
        return os.path.getsize(path), 1
    return 0, 0


def _write_json(path: str, payload: dict) -> None:
    """Write JSON file."""
    with open(path, "w") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)


def _write_slice_csv(path: str, rows: list[dict[str, Any]]) -> None:
    """Write slice-level CSV."""
    if not rows:
        return
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _resolve_series_hash(series_dir: str, row: dict[str, Any]) -> str:
    """Resolve series hash from row or generate from path."""
    series_hash = row.get("series_hash")
    if series_hash:
        return str(series_hash)
    return _hash(series_dir)


def _shard_for_row(series_hash: str, num_shards: int) -> int:
    """Determine shard ID for a series."""
    try:
        return int(series_hash, 16) % num_shards
    except Exception:
        return int(hashlib.sha1(series_hash.encode("utf-8")).hexdigest()[:8], 16) % num_shards


def _parse_int(value: str | None) -> int | None:
    """Parse integer from string."""
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _process_series(row: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Process a single DICOM series to Zarr volume."""
    series_dir = row.get("series_dir") or row.get("path") or ""
    dataset = row.get("dataset") or "unknown"
    series_hash = _resolve_series_hash(series_dir, row)
    file_count = row.get("file_count")

    if not series_dir:
        return {
            "dataset": dataset,
            "series_dir": series_dir,
            "series_hash": series_hash,
            "status": "failed",
            "reason": "no_series_dir",
        }

    store_type = config["store_type"]
    data_dir = config["data_dir"]
    meta_dir = config["meta_dir"]
    verify_slices = config["verify_slices"]
    overwrite = config["overwrite"]
    compressor_config = config["compressor_config"]

    series_out_path = _store_path(data_dir, series_hash, store_type)
    series_json_path = os.path.join(meta_dir, f"{series_hash}.series.json")
    slices_csv_path = os.path.join(meta_dir, f"{series_hash}.slices.csv")

    if not overwrite and (os.path.exists(series_out_path) or os.path.exists(series_json_path)):
        return {
            "dataset": dataset,
            "series_dir": series_dir,
            "series_hash": series_hash,
            "status": "skipped",
            "reason": "exists",
        }

    if overwrite and os.path.exists(series_out_path):
        if os.path.isdir(series_out_path):
            shutil.rmtree(series_out_path)
        else:
            os.remove(series_out_path)

    start_time = time.time()
    files = _list_files(series_dir)
    if not files:
        return {
            "dataset": dataset,
            "series_dir": series_dir,
            "series_hash": series_hash,
            "status": "failed",
            "reason": "no_files",
        }

    slice_items = []
    read_errors = 0
    localizer_skipped = 0
    derived_skipped = 0
    shape = None
    read_start = time.time()

    for path in files:
        try:
            ds = pydicom.dcmread(path, force=True)
        except Exception:
            read_errors += 1
            continue

        # Skip LOCALIZER/SCOUT images (X-ray, not CT slices)
        if _is_localizer(ds):
            localizer_skipped += 1
            continue

        # Skip DERIVED images (MPR reformats like Coronal/Sagittal)
        if _is_derived(ds):
            derived_skipped += 1
            continue

        try:
            arr = ds.pixel_array
        except Exception:
            read_errors += 1
            continue

        if arr.ndim != 2:
            read_errors += 1
            continue

        if shape is None:
            shape = arr.shape
        elif arr.shape != shape:
            read_errors += 1
            continue

        tags = {}
        for tag in ALL_TAGS:
            val = _get_tag(ds, tag)
            if val is not None:
                tags[tag] = val

        pos = getattr(ds, "ImagePositionPatient", None)
        z_pos = None
        if pos is not None and len(pos) >= 3:
            try:
                z_pos = float(pos[2])
            except Exception:
                z_pos = None

        inst_num = _parse_int(getattr(ds, "InstanceNumber", None))

        slice_items.append(
            {
                "path": path,
                "file_name": os.path.basename(path),
                "array": arr,
                "sort_key": _sort_key_values(z_pos, inst_num, path),
                "tags": tags,
                "z_pos": z_pos,
            }
        )

    read_time = time.time() - read_start
    if not slice_items:
        return {
            "dataset": dataset,
            "series_dir": series_dir,
            "series_hash": series_hash,
            "status": "failed",
            "reason": "read_failed",
            "read_errors": read_errors,
            "localizer_skipped": localizer_skipped,
            "derived_skipped": derived_skipped,
        }

    slice_items.sort(key=lambda x: x["sort_key"])

    z_positions = [item["z_pos"] for item in slice_items if item["z_pos"] is not None]

    volume = np.stack([item["array"] for item in slice_items], axis=0)
    z_dim, rows_dim, cols_dim = volume.shape
    if rows_dim == 0 or cols_dim == 0:
        return {
            "dataset": dataset,
            "series_dir": series_dir,
            "series_hash": series_hash,
            "status": "failed",
            "reason": "invalid_shape",
        }

    z_chunk = 16
    if z_dim >= 256:
        z_chunk = 64
    elif z_dim >= 64:
        z_chunk = 32

    y_chunk = min(256, rows_dim)
    x_chunk = min(256, cols_dim)

    write_start = time.time()
    compressor = Blosc(**compressor_config)
    store = _create_store(series_out_path, store_type)
    root = zarr.group(store=store, overwrite=True)
    arr = root.create_dataset(
        "volume",
        shape=volume.shape,
        chunks=(z_chunk, y_chunk, x_chunk),
        dtype=volume.dtype,
        compressor=compressor,
    )
    arr[:] = volume
    if hasattr(store, "close"):
        store.close()
    write_time = time.time() - write_start

    series_meta = {}
    slice_rows = []
    for tag in ALL_TAGS:
        values = [item["tags"].get(tag) for item in slice_items]
        all_same = all(v == values[0] for v in values)
        if all_same and tag not in ALWAYS_SLICE_TAGS:
            if values[0] is not None:
                series_meta[tag] = values[0]
            for item in slice_items:
                item["tags"].pop(tag, None)

    for order_idx, item in enumerate(slice_items):
        row_out = {
            "slice_index": order_idx,
            "file_name": item["file_name"],
        }
        row_out.update(item["tags"])
        slice_rows.append(row_out)

    spacing = _compute_spacing(z_positions)
    series_meta.update(
        {
            "dataset": dataset,
            "series_dir": series_dir,
            "series_hash": series_hash,
            "file_count": int(file_count) if file_count else len(files),
            "slice_count": z_dim,
            "volume_shape": list(volume.shape),
            "dtype": str(volume.dtype),
            "store_type": store_type,
            "chunks": [z_chunk, y_chunk, x_chunk],
            "spacing_z_median": spacing,
            "converted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "read_errors": read_errors,
            "localizer_skipped": localizer_skipped,
            "derived_skipped": derived_skipped,
        }
    )

    if verify_slices > 0:
        indices = np.linspace(0, z_dim - 1, num=min(verify_slices, z_dim), dtype=int)
        verify_ok = True
        store = _create_store(series_out_path, store_type)
        root = zarr.open_group(store=store, mode="r")
        for s_idx in indices:
            orig_md5 = hashlib.md5(volume[s_idx].tobytes()).hexdigest()
            load_md5 = hashlib.md5(root["volume"][s_idx].tobytes()).hexdigest()
            if orig_md5 != load_md5:
                verify_ok = False
                break
        if hasattr(store, "close"):
            store.close()
        series_meta["verify_ok"] = verify_ok
    else:
        series_meta["verify_ok"] = None

    _write_json(series_json_path, series_meta)
    _write_slice_csv(slices_csv_path, slice_rows)

    total_time = time.time() - start_time
    out_size, out_files = _path_stats(series_out_path)
    return {
        "dataset": dataset,
        "series_dir": series_dir,
        "series_hash": series_hash,
        "status": "ok",
        "slice_count": z_dim,
        "read_time_sec": round(read_time, 4),
        "write_time_sec": round(write_time, 4),
        "total_time_sec": round(total_time, 4),
        "output_bytes": out_size,
        "output_files": out_files,
        "read_errors": read_errors,
        "localizer_skipped": localizer_skipped,
        "derived_skipped": derived_skipped,
        "verify_ok": series_meta.get("verify_ok"),
    }


def _worker(args: tuple[dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    """Worker function for parallel processing."""
    row, config = args
    try:
        return _process_series(row, config)
    except Exception as exc:
        series_dir = row.get("series_dir") or row.get("path") or ""
        return {
            "dataset": row.get("dataset") or "unknown",
            "series_dir": series_dir,
            "series_hash": _resolve_series_hash(series_dir, row),
            "status": "failed",
            "reason": "exception",
            "error": str(exc),
        }


def _run_serial(rows: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    """Run conversion in serial mode."""
    summary_rows = []
    total = len(rows)
    for idx, row in enumerate(rows, start=1):
        summary_rows.append(_process_series(row, config))
        if idx % 10 == 0 or idx == total:
            print(f"progress={idx}/{total}")
    return summary_rows


def _run_parallel(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    num_workers: int,
) -> list[dict[str, Any]]:
    """Run conversion in parallel mode."""
    summary_rows = []
    total = len(rows)
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=num_workers) as pool:
        for idx, result in enumerate(
            pool.imap_unordered(_worker, [(row, config) for row in rows]), start=1
        ):
            summary_rows.append(result)
            if idx % 10 == 0 or idx == total:
                print(f"progress={idx}/{total}")
    return summary_rows


def main() -> None:
    """Main entry point."""
    input_csv = os.environ.get("INPUT_CSV") or os.environ.get("SAMPLE_CSV", DEFAULT_SAMPLE_CSV)
    out_root = os.environ.get("OUT_ROOT", DEFAULT_OUT_ROOT)
    store_type = os.environ.get("STORE_TYPE", "sqlite").lower().strip()
    limit = int(os.environ.get("LIMIT", "0"))
    verify_slices = int(os.environ.get("VERIFY_SLICES", "3"))
    overwrite = os.environ.get("OVERWRITE", "0") == "1"
    num_workers = int(os.environ.get("NUM_WORKERS", "1"))
    shard_id_raw = os.environ.get("SHARD_ID")
    num_shards_raw = os.environ.get("NUM_SHARDS")
    summary_suffix = os.environ.get("SUMMARY_SUFFIX", "").strip()

    shard_id = _parse_int(shard_id_raw)
    num_shards = _parse_int(num_shards_raw)

    store_name = f"zarr_{store_type}"
    data_dir = os.path.join(out_root, store_name, "data")
    meta_dir = os.path.join(out_root, store_name, "meta")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    with open(input_csv, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if shard_id is not None and num_shards and num_shards > 1:
        filtered = []
        for row in rows:
            series_dir = row.get("series_dir") or row.get("path") or ""
            series_hash = _resolve_series_hash(series_dir, row)
            if _shard_for_row(series_hash, num_shards) == shard_id:
                filtered.append(row)
        rows = filtered

    if limit > 0:
        rows = rows[:limit]

    print(f"input_csv={input_csv}")
    print(f"store_type={store_type}")
    print(f"series_count={len(rows)}")
    print(f"out_root={out_root}")
    print(f"num_workers={num_workers}")
    if shard_id is not None and num_shards:
        print(f"shard_id={shard_id} num_shards={num_shards}")

    compressor_config = {
        "cname": "zstd",
        "clevel": 3,
        "shuffle": int(Blosc.BITSHUFFLE),
    }

    config = {
        "store_type": store_type,
        "data_dir": data_dir,
        "meta_dir": meta_dir,
        "verify_slices": verify_slices,
        "overwrite": overwrite,
        "compressor_config": compressor_config,
    }

    if num_workers <= 1:
        summary_rows = _run_serial(rows, config)
    else:
        summary_rows = _run_parallel(rows, config, num_workers)

    safe_suffix = summary_suffix.replace("/", "_").replace(" ", "_")
    suffix = f"_{safe_suffix}" if safe_suffix else ""
    summary_csv = os.path.join(out_root, store_name, f"summary{suffix}.csv")
    summary_json = os.path.join(out_root, store_name, f"summary{suffix}.json")

    if summary_rows:
        fieldnames = sorted({k for row in summary_rows for k in row.keys()})
        with open(summary_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in summary_rows:
                writer.writerow(row)

        totals = {
            "series_total": len(summary_rows),
            "series_ok": sum(1 for row in summary_rows if row.get("status") == "ok"),
            "series_failed": sum(
                1 for row in summary_rows if row.get("status") == "failed"
            ),
            "series_skipped": sum(
                1 for row in summary_rows if row.get("status") == "skipped"
            ),
            "localizer_skipped_total": sum(
                int(row.get("localizer_skipped") or 0) for row in summary_rows
            ),
            "derived_skipped_total": sum(
                int(row.get("derived_skipped") or 0) for row in summary_rows
            ),
            "output_bytes_total": sum(int(row.get("output_bytes") or 0) for row in summary_rows),
            "output_files_total": sum(int(row.get("output_files") or 0) for row in summary_rows),
            "input_csv": input_csv,
            "store_type": store_type,
            "num_workers": num_workers,
            "shard_id": shard_id,
            "num_shards": num_shards,
        }
        _write_json(summary_json, totals)

    print(f"summary_csv={summary_csv}")
    print(f"summary_json={summary_json}")


if __name__ == "__main__":
    main()
