# WQP-Centered Multi-State Water Quality Extractor

This version broadens the dataset and makes WQP the primary source of truth.

## Project overview
This project builds a multi-state stream water quality dataset centered on the
EPA Water Quality Portal (WQP). It pulls monitoring results for CA, OR, WA, NV,
and AZ from 2015-01-01 through 2024-12-31, focusing on pH, turbidity, and
dissolved oxygen.

The main pipeline converts raw WQP results into a daily site-level dataset,
adds station metadata, creates time features, lag features, rolling averages,
and a turbidity compliance flag, then saves CSV and parquet outputs for
analysis and modeling.

An optional USGS enrichment step can add daily discharge, water temperature, and
gage height from USGS NWIS for WQP sites that map to USGS site numbers. These
features are merged back into the base dataset by site and date, with support
for a nearest-prior carry-forward window.

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

## Current processed dataset
- Base final dataset: 144,849 rows across 2,236 sites
- USGS-enriched dataset: 144,849 rows
- USGS-backed sites before filtering: 273
- USGS-backed sites after filtering: 108

## Notebooks
The notebooks support the workflow interactively:
- `notebooks/01_extract_wqp.ipynb`: WQP extraction
- `notebooks/02_extract_usgs.ipynb`: USGS extraction
- `notebooks/01_validate_usgs_enrichment.ipynb`: USGS enrichment validation
- `notebooks/03_merge_and_features.ipynb`: merge and feature creation
- `notebooks/04_modeling_template.ipynb`: modeling template
- `notebooks/wqp_eda_preprocessing_modelling.ipynb`: full EDA,
  preprocessing, and modeling notebook
- `notebooks/wqp_eda_preprocessing_modelling.html`: exported HTML version of
  the full EDA, preprocessing, and modeling notebook

## Run
```bash
python src/build_dataset.py --config config/config.multistate.yaml
```

## Optional USGS enrichment
```bash
python src/enrich_with_usgs.py --config config/config.enrichment.yaml
```
