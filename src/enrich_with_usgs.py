from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import yaml

from usgs_enrichment import (
    extract_usgs_site_no,
    batch_fetch_usgs_daily,
    aggregate_usgs_daily,
    enrich_with_usgs_exact,
    enrich_with_usgs_nearest_prior,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_dataset(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path, parse_dates=["date"])


def save_df(df: pd.DataFrame, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix == ".parquet":
        df.to_parquet(p, index=False)
    elif p.suffix == ".csv":
        df.to_csv(p, index=False)
    else:
        raise ValueError(f"Unsupported output format: {p.suffix}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg["paths"]
    usgs_cfg = cfg["usgs"]
    proc_cfg = cfg["processing"]

    base_df = read_dataset(paths["base_dataset_path"]).copy()
    base_df["date"] = pd.to_datetime(base_df["date"], errors="coerce")
    base_df["usgs_site_no"] = base_df["site_id"].apply(extract_usgs_site_no)

    start_date = str(base_df["date"].min().date())
    end_date = str(base_df["date"].max().date())
    usgs_sites = sorted(base_df["usgs_site_no"].dropna().astype(str).unique().tolist())

    LOGGER.info("Base rows: %s", len(base_df))
    LOGGER.info("Unique base sites: %s", base_df["site_id"].nunique())
    LOGGER.info("USGS-backed sites in base data: %s", len(usgs_sites))
    LOGGER.info("USGS enrichment date range: %s to %s", start_date, end_date)

    usgs_long = batch_fetch_usgs_daily(
        site_list=usgs_sites,
        parameter_map=usgs_cfg["parameter_map"],
        start_date=start_date,
        end_date=end_date,
        site_status=usgs_cfg.get("site_status", "all"),
        pause_seconds=float(usgs_cfg.get("request_pause_seconds", 0.0)),
    )

    if usgs_long.empty:
        LOGGER.warning("No USGS data returned. Saving base dataset with USGS site column only.")
        enriched = base_df.copy()
        save_df(enriched, paths["output_csv_path"])
        save_df(enriched, paths["output_parquet_path"])
        summary = {
            "base_rows": int(len(base_df)),
            "enriched_rows": int(len(enriched)),
            "usgs_backed_sites": int(len(usgs_sites)),
            "usgs_long_rows": 0,
            "usgs_exact_nonnull": {},
            "usgs_nearest_nonnull": {},
            "note": "No USGS data returned",
        }
        Path(paths["output_summary_path"]).parent.mkdir(parents=True, exist_ok=True)
        with open(paths["output_summary_path"], "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return

    site_counts = usgs_long.groupby("usgs_site_no").size()
    keep_sites = site_counts[site_counts >= int(proc_cfg.get("min_usgs_rows_per_site", 5))].index.tolist()
    usgs_long = usgs_long[usgs_long["usgs_site_no"].isin(keep_sites)].copy()
    usgs_wide = aggregate_usgs_daily(usgs_long)

    exact_df = enrich_with_usgs_exact(base_df, usgs_wide)
    exact_nonnull = {c: int(exact_df[c].notna().sum()) for c in ["discharge", "water_temp", "gage_height"] if c in exact_df.columns}

    LOGGER.info("Base rows with USGS site number: %s", base_df["usgs_site_no"].notna().sum())
    LOGGER.info("Base rows without USGS site number: %s", base_df["usgs_site_no"].isna().sum())
    LOGGER.info("USGS wide rows: %s", len(usgs_wide))
    LOGGER.info("Carry-forward tolerance (days): %s", usgs_cfg.get("carry_forward_days", 7))

    enriched = enrich_with_usgs_nearest_prior(
        base_df,
        usgs_wide,
        tolerance_days=int(usgs_cfg.get("carry_forward_days", 7)),
    )
    nearest_nonnull = {c: int(enriched[c].notna().sum()) for c in ["discharge", "water_temp", "gage_height"] if c in enriched.columns}

    save_df(enriched, paths["output_csv_path"])
    save_df(enriched, paths["output_parquet_path"])

    summary = {
        "base_rows": int(len(base_df)),
        "enriched_rows": int(len(enriched)),
        "unique_base_sites": int(base_df["site_id"].nunique()),
        "usgs_backed_sites_before_filter": int(len(usgs_sites)),
        "usgs_backed_sites_after_filter": int(len(keep_sites)),
        "usgs_long_rows_after_filter": int(len(usgs_long)),
        "usgs_exact_nonnull": exact_nonnull,
        "usgs_nearest_nonnull": nearest_nonnull,
        "carry_forward_days": int(usgs_cfg.get("carry_forward_days", 7)),
    }
    Path(paths["output_summary_path"]).parent.mkdir(parents=True, exist_ok=True)
    with open(paths["output_summary_path"], "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    LOGGER.info("Enrichment complete.")
    LOGGER.info("Exact non-null coverage: %s", exact_nonnull)
    LOGGER.info("Nearest-prior non-null coverage: %s", nearest_nonnull)


if __name__ == "__main__":
    main()
