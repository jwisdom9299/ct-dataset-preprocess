# CT Dataset Preprocessing Pipeline

DICOM to Zarr volume conversion pipeline for CT datasets. Designed for large-scale distributed processing with support for multiple storage backends.

## Features

- **DICOM to Zarr Conversion**: Convert DICOM series to chunked Zarr volumes
- **Multiple Storage Backends**: SQLite, DirectoryStore, ZipStore
- **Blosc Compression**: Efficient compression with zstd and bitshuffle
- **LOCALIZER Filtering**: Automatically excludes scout/localizer X-ray images
- **Distributed Processing**: Sharded manifest for parallel execution
- **Comprehensive Metadata**: Extracts and preserves DICOM tags
- **Verification**: MD5-based slice verification
- **Visual Reports**: HTML reports for manual verification

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Generate Manifest

Scan your DICOM dataset and generate manifest CSV files:

```bash
export DATASET_V1_ROOT=/path/to/dataset_v1
export DATASET_V2_ROOT=/path/to/dataset_v2
export OUT_DIR=/path/to/manifests
export NUM_SHARDS=16

python scripts/generate_manifest.py
```

Output:
- `series_all.csv`: Complete manifest
- `shards/series_shard_XX.csv`: Sharded manifests for distributed processing

### 2. Convert DICOM to Zarr

Convert a single shard:

```bash
export INPUT_CSV=/path/to/manifests/shards/series_shard_00.csv
export OUT_ROOT=/path/to/output
export STORE_TYPE=sqlite  # or dir, zip
export NUM_WORKERS=1
export SHARD_ID=0
export NUM_SHARDS=16
export SUMMARY_SUFFIX=shard_00

python scripts/convert_dicom_to_volume.py
```

### 3. Aggregate Results

After all shards complete:

```bash
export OUT_ROOT=/path/to/output
export STORE_TYPE=sqlite

python scripts/aggregate_summaries.py
```

### 4. Generate Verification Report

```bash
export CONVERTED_ROOT=/path/to/output
export STORE_TYPE=sqlite
export OUTPUT_HTML=/tmp/verify_report.html
export NUM_SAMPLES=10

python scripts/generate_verify_report.py
```

## Environment Variables

### convert_dicom_to_volume.py

| Variable | Description | Default |
|----------|-------------|---------|
| INPUT_CSV | Path to manifest CSV | Required |
| OUT_ROOT | Output directory root | Required |
| STORE_TYPE | Storage backend (sqlite/dir/zip) | sqlite |
| NUM_WORKERS | Parallel workers | 1 |
| SHARD_ID | Current shard ID | None |
| NUM_SHARDS | Total shards | None |
| OVERWRITE | Overwrite existing (0/1) | 0 |
| VERIFY_SLICES | Number of slices to verify | 3 |
| SUMMARY_SUFFIX | Suffix for summary files | "" |

### generate_manifest.py

| Variable | Description | Default |
|----------|-------------|---------|
| OUT_DIR | Output directory | Required |
| NUM_SHARDS | Number of shards | 16 |
| MAX_DEPTH | Max directory depth | 3 |
| DATASET_VERSION | v1/v2/all | all |
| DATASET_V1_ROOT | v1 dataset path | - |
| DATASET_V2_ROOT | v2 dataset path | - |

## Output Structure

```
output/
└── zarr_sqlite/
    ├── data/
    │   ├── {series_hash}.zarr.sqlite
    │   └── ...
    ├── meta/
    │   ├── {series_hash}.series.json
    │   ├── {series_hash}.slices.csv
    │   └── ...
    ├── summary_shard_00.csv
    ├── summary_shard_00.json
    └── summary_all.csv
```

## SkyPilot Integration

Example SkyPilot configuration for distributed processing:

```yaml
# configs/convert_full.yaml
resources:
  cloud: your-cloud
  instance_type: cpu-large
  image_id: pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

envs:
  MANIFEST_DIR: /data/manifests
  OUT_ROOT: /data/converted
  STORE_TYPE: sqlite
  NUM_SHARDS: "16"
  NUM_WORKERS: "1"
  OVERWRITE: "0"
  START_SHARD: "0"
  END_SHARD: "7"

run: |
  pip install -q pydicom zarr numcodecs numpy pylibjpeg pylibjpeg-libjpeg

  for shard_id in $(seq $START_SHARD $END_SHARD); do
    export INPUT_CSV="${MANIFEST_DIR}/shards/series_shard_$(printf '%02d' $shard_id).csv"
    export SHARD_ID=$shard_id
    export SUMMARY_SUFFIX="shard_$(printf '%02d' $shard_id)"
    python /path/to/convert_dicom_to_volume.py
  done
```

Run with:
```bash
sky exec cluster-name configs/convert_full.yaml
```

## LOCALIZER Filtering

The converter automatically filters out LOCALIZER/SCOUT/TOPOGRAM images which are X-ray scout images, not actual CT slices. These are identified by the DICOM ImageType tag.

Filtered image types:
- LOCALIZER
- SCOUT
- TOPOGRAM

The count of skipped localizer images is tracked in the summary output.

## Metadata Extraction

### Series-level Tags
- PatientID, PatientName, PatientSex, PatientAge
- StudyInstanceUID, SeriesInstanceUID
- StudyDate, SeriesDate
- Modality, Manufacturer, BodyPartExamined
- Rows, Columns, PixelSpacing, SliceThickness
- RescaleSlope, RescaleIntercept
- And more...

### Slice-level Tags
- SOPInstanceUID
- InstanceNumber
- ImagePositionPatient
- SliceLocation

## License

MIT License
