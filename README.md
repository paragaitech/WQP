# WQP-Centered Multi-State Water Quality Extractor

This version broadens the dataset and makes WQP the primary source of truth.

## Strategy
- Primary source: EPA Water Quality Portal (WQP)
- Optional enrichment: WQP station metadata
- Optional future enrichment: USGS merge where available
- States: CA, OR, WA, NV, AZ
- Period: 2015-01-01 to 2024-12-31
- Lower site threshold to retain more rows

## Main outputs
- `data/raw/wqp_long.csv`
- `data/interim/wqp_wide.csv`
- `data/interim/site_metadata.csv`
- `data/processed/final_dataset.csv`
- `data/processed/final_dataset.parquet`
- `data/processed/extraction_summary.json`

## Run
```bash
python src/build_dataset.py --config config/config.multistate.yaml
```
