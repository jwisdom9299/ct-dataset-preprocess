#!/usr/bin/env python3
"""
Summary Aggregator

Aggregates per-shard summary CSV files into a single summary file
with statistics across all shards.

Environment Variables:
- OUT_ROOT: Output directory root containing zarr_{store_type}/ subdirectory
- STORE_TYPE: Storage type (sqlite, dir, zip) - default: sqlite
"""

import csv
import glob
import json
import os
import statistics
from typing import Optional


def _safe_float(value: Optional[str]) -> Optional[float]:
    """Safely convert string to float."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def _safe_int(value: Optional[str]) -> Optional[int]:
    """Safely convert string to int."""
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception:
        return None


def _stats(values: list[Optional[float]]) -> dict[str, Optional[float]]:
    """Calculate statistics for a list of values."""
    clean = [v for v in values if v is not None]
    if not clean:
        return {"mean": None, "p50": None, "min": None, "max": None}
    return {
        "mean": float(statistics.mean(clean)),
        "p50": float(statistics.median(clean)),
        "min": float(min(clean)),
        "max": float(max(clean)),
    }


def main() -> None:
    """Main entry point."""
    out_root = os.environ.get("OUT_ROOT", "/data/converted")
    store_type = os.environ.get("STORE_TYPE", "sqlite").lower().strip()
    store_name = f"zarr_{store_type}"

    summary_glob = os.path.join(out_root, store_name, "summary_shard_*.csv")
    summary_files = sorted(glob.glob(summary_glob))

    if not summary_files:
        # Try alternative pattern
        summary_glob = os.path.join(out_root, store_name, "summary*.csv")
        summary_files = sorted(glob.glob(summary_glob))
        # Exclude the aggregate file if it exists
        summary_files = [f for f in summary_files if "summary_all" not in f]

    print(f"Found {len(summary_files)} summary files")

    rows = []
    for path in summary_files:
        print(f"Reading: {path}")
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

    print(f"Total rows: {len(rows)}")

    summary_csv = os.path.join(out_root, store_name, "summary_all.csv")
    summary_json = os.path.join(out_root, store_name, "summary_all.json")

    if rows:
        fieldnames = sorted({k for row in rows for k in row.keys()})
        with open(summary_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    ok_rows = [row for row in rows if row.get("status") == "ok"]
    read_times = [_safe_float(row.get("read_time_sec")) for row in ok_rows]
    write_times = [_safe_float(row.get("write_time_sec")) for row in ok_rows]
    total_times = [_safe_float(row.get("total_time_sec")) for row in ok_rows]
    output_bytes = [_safe_int(row.get("output_bytes")) for row in ok_rows]
    slice_counts = [_safe_int(row.get("slice_count")) for row in ok_rows]

    verify_ok_count = sum(1 for row in ok_rows if str(row.get("verify_ok")).lower() == "true")
    verify_fail_count = sum(1 for row in ok_rows if str(row.get("verify_ok")).lower() == "false")

    totals = {
        "series_total": len(rows),
        "series_ok": sum(1 for row in rows if row.get("status") == "ok"),
        "series_failed": sum(1 for row in rows if row.get("status") == "failed"),
        "series_skipped": sum(1 for row in rows if row.get("status") == "skipped"),
        "verify_ok_count": verify_ok_count,
        "verify_fail_count": verify_fail_count,
        "localizer_skipped_total": sum(
            _safe_int(row.get("localizer_skipped")) or 0 for row in rows
        ),
        "output_bytes_total": sum(v or 0 for v in output_bytes),
        "output_gb_total": round(sum(v or 0 for v in output_bytes) / (1024**3), 2),
        "output_files_total": sum(_safe_int(row.get("output_files")) or 0 for row in ok_rows),
        "slice_count_stats": _stats([float(v) for v in slice_counts if v is not None]),
        "read_time_sec": _stats(read_times),
        "write_time_sec": _stats(write_times),
        "total_time_sec": _stats(total_times),
        "summary_files": summary_files,
        "store_type": store_type,
    }

    with open(summary_json, "w") as f:
        json.dump(totals, f, ensure_ascii=True, indent=2)

    print(f"summary_csv={summary_csv}")
    print(f"summary_json={summary_json}")
    print(f"series_total={totals['series_total']}")
    print(f"series_ok={totals['series_ok']}")
    print(f"series_failed={totals['series_failed']}")
    print(f"series_skipped={totals['series_skipped']}")
    print(f"verify_ok={verify_ok_count}/{totals['series_ok']}")
    print(f"output_gb_total={totals['output_gb_total']}")


if __name__ == "__main__":
    main()
