#!/usr/bin/env python3
"""
Verification Report Generator

Generates an HTML report comparing original DICOM images with
converted Zarr volumes for visual verification.

Features:
- Random sampling of converted series
- Side-by-side comparison (DICOM vs Zarr vs Difference)
- Metadata display
- Self-contained HTML with embedded images

Environment Variables:
- CONVERTED_ROOT: Root directory of converted data
- STORE_TYPE: Storage type (sqlite, dir, zip)
- OUTPUT_HTML: Output HTML file path
- NUM_SAMPLES: Number of samples to include (default: 10)
- NUM_SLICES: Number of slices per sample (default: 3)
"""

import base64
import csv
import io
import json
import os
import random
from typing import Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pydicom
import zarr


def load_summary(summary_csv: str) -> list[dict]:
    """Load summary CSV and filter for OK status."""
    rows = []
    with open(summary_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") == "ok":
                rows.append(row)
    return rows


def load_zarr_volume(zarr_path: str, store_type: str) -> Optional[np.ndarray]:
    """Load volume from Zarr store."""
    if not os.path.exists(zarr_path):
        return None

    if store_type == "sqlite":
        store = zarr.SQLiteStore(zarr_path)
    elif store_type == "zip":
        store = zarr.ZipStore(zarr_path, mode="r")
    else:
        store = zarr.DirectoryStore(zarr_path)

    root = zarr.open_group(store=store, mode="r")
    volume = root["volume"][:]
    if hasattr(store, "close"):
        store.close()
    return volume


def load_series_meta(meta_path: str) -> dict:
    """Load series metadata JSON."""
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path, "r") as f:
        return json.load(f)


def load_slices_csv(csv_path: str) -> list[dict]:
    """Load slices CSV."""
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        return list(reader)


def load_dicom_slice(series_dir: str, slice_idx: int, slices_info: list[dict]) -> Optional[np.ndarray]:
    """Load original DICOM slice."""
    if slice_idx >= len(slices_info):
        return None
    file_name = slices_info[slice_idx].get("file_name")
    if not file_name:
        return None
    dicom_path = os.path.join(series_dir, file_name)
    if not os.path.exists(dicom_path):
        return None
    try:
        ds = pydicom.dcmread(dicom_path, force=True)
        return ds.pixel_array
    except Exception:
        return None


def fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64


def generate_comparison(sample: dict, slice_indices: list[int],
                        data_dir: str, meta_dir: str, store_type: str) -> tuple:
    """Generate comparison images for a sample."""
    series_hash = sample.get("series_hash")
    series_dir = sample.get("series_dir")

    # Build paths
    if store_type == "sqlite":
        zarr_path = os.path.join(data_dir, f"{series_hash}.zarr.sqlite")
    elif store_type == "zip":
        zarr_path = os.path.join(data_dir, f"{series_hash}.zarr.zip")
    else:
        zarr_path = os.path.join(data_dir, f"{series_hash}.zarr")

    meta_path = os.path.join(meta_dir, f"{series_hash}.series.json")
    slices_path = os.path.join(meta_dir, f"{series_hash}.slices.csv")

    # Load data
    volume = load_zarr_volume(zarr_path, store_type)
    slices_info = load_slices_csv(slices_path)
    meta = load_series_meta(meta_path)

    if volume is None or not slices_info:
        return None, None

    images = []
    for slice_idx in slice_indices:
        slice_idx = min(slice_idx, len(slices_info) - 1, volume.shape[0] - 1)

        dicom_arr = load_dicom_slice(series_dir, slice_idx, slices_info)
        zarr_arr = volume[slice_idx]

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        # DICOM original
        if dicom_arr is not None:
            axes[0].imshow(dicom_arr, cmap="gray")
            axes[0].set_title(f"DICOM (slice {slice_idx})")
        else:
            axes[0].text(0.5, 0.5, "N/A", ha="center", va="center")
            axes[0].set_title("DICOM")
        axes[0].axis("off")

        # Zarr converted
        axes[1].imshow(zarr_arr, cmap="gray")
        axes[1].set_title(f"Zarr (slice {slice_idx})")
        axes[1].axis("off")

        # Difference
        if dicom_arr is not None and dicom_arr.shape == zarr_arr.shape:
            diff = np.abs(dicom_arr.astype(float) - zarr_arr.astype(float))
            axes[2].imshow(diff, cmap="hot")
            match = "OK" if diff.max() == 0 else f"max={diff.max():.0f}"
            axes[2].set_title(f"Diff ({match})")
        else:
            axes[2].text(0.5, 0.5, "N/A", ha="center", va="center")
            axes[2].set_title("Diff")
        axes[2].axis("off")

        plt.tight_layout()
        images.append(fig_to_base64(fig))

    return images, meta


def generate_html_report(samples: list[dict], num_slices: int,
                         data_dir: str, meta_dir: str, store_type: str) -> str:
    """Generate HTML report."""
    header = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>DICOM to Zarr Conversion Verification Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ color: #333; text-align: center; }}
        .summary {{ background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
        .sample {{ background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .sample h3 {{ margin-top: 0; color: #2196F3; }}
        .meta {{ background: #f9f9f9; padding: 10px; border-radius: 4px; font-size: 14px; margin-bottom: 15px; }}
        .meta code {{ background: #e0e0e0; padding: 2px 6px; border-radius: 3px; }}
        .slice-img {{ margin: 10px 0; text-align: center; }}
        .slice-img img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }}
        .info-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 10px; }}
        .info-item {{ background: #e3f2fd; padding: 8px; border-radius: 4px; }}
        .info-item strong {{ color: #1565C0; }}
    </style>
</head>
<body>
<div class="container">
    <h1>DICOM to Zarr Conversion Verification Report</h1>
    <div class="summary">
        <h2>Verification Summary</h2>
        <p>Comparing <strong>{len(samples)}</strong> randomly sampled DICOM series with their Zarr conversions.</p>
        <p>Each sample shows <strong>{num_slices}</strong> slices (first, middle, last).</p>
        <p>Storage type: <strong>{store_type}</strong></p>
    </div>
"""
    html_parts = [header]

    for i, sample in enumerate(samples):
        series_hash = sample.get("series_hash", "")
        slice_count = int(sample.get("slice_count", 0))

        # Slice indices to verify (first, middle, last)
        if slice_count >= 3:
            slice_indices = [0, slice_count // 2, slice_count - 1]
        else:
            slice_indices = list(range(slice_count))

        print(f"Processing sample {i+1}/{len(samples)}: {series_hash}")
        images, meta = generate_comparison(sample, slice_indices, data_dir, meta_dir, store_type)

        if images is None:
            continue

        html_parts.append(f"""
    <div class="sample">
        <h3>Sample {i+1}: {series_hash}</h3>
        <div class="info-grid">
            <div class="info-item"><strong>Dataset:</strong> {sample.get('dataset', 'N/A')}</div>
            <div class="info-item"><strong>Slices:</strong> {slice_count}</div>
            <div class="info-item"><strong>Verify OK:</strong> {sample.get('verify_ok', 'N/A')}</div>
            <div class="info-item"><strong>Size:</strong> {int(sample.get('output_bytes', 0)) / 1024 / 1024:.1f} MB</div>
        </div>
        <div class="meta">
            <strong>Metadata:</strong>
            Modality=<code>{meta.get('Modality', 'N/A')}</code>,
            Rows=<code>{meta.get('Rows', 'N/A')}</code>,
            Cols=<code>{meta.get('Columns', 'N/A')}</code>,
            SliceThickness=<code>{meta.get('SliceThickness', 'N/A')}</code>,
            spacing_z=<code>{meta.get('spacing_z_median', 'N/A')}</code>
        </div>
""")

        for img_base64 in images:
            html_parts.append(f"""
        <div class="slice-img">
            <img src="data:image/png;base64,{img_base64}" />
        </div>
""")

        html_parts.append("    </div>")

    html_parts.append("""
</div>
</body>
</html>
""")

    return "".join(html_parts)


def main():
    """Main entry point."""
    # Configuration
    converted_root = os.environ.get("CONVERTED_ROOT", "/data/converted")
    store_type = os.environ.get("STORE_TYPE", "sqlite").lower().strip()
    output_html = os.environ.get("OUTPUT_HTML", "/tmp/verify_report.html")
    num_samples = int(os.environ.get("NUM_SAMPLES", "10"))
    num_slices = int(os.environ.get("NUM_SLICES", "3"))

    store_name = f"zarr_{store_type}"
    summary_csv = os.path.join(converted_root, store_name, "summary_all.csv")
    data_dir = os.path.join(converted_root, store_name, "data")
    meta_dir = os.path.join(converted_root, store_name, "meta")

    print(f"Loading summary from: {summary_csv}")
    rows = load_summary(summary_csv)
    print(f"Found {len(rows)} OK series")

    # Random sample selection
    samples = random.sample(rows, min(num_samples, len(rows)))
    print(f"Selected {len(samples)} samples")

    # Generate HTML report
    print("Generating HTML report...")
    html = generate_html_report(samples, num_slices, data_dir, meta_dir, store_type)

    # Save file
    with open(output_html, "w") as f:
        f.write(html)

    print(f"Report saved to: {output_html}")
    print(f"File size: {os.path.getsize(output_html) / 1024:.1f} KB")


if __name__ == "__main__":
    main()
