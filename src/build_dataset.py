from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import yaml

from features import (
    add_compliance_flag,
    add_lag_features,
    add_rolling_mean_features,
    add_time_features,
    filter_usable_sites,
)
from wqp import aggregate_wqp_daily, pull_characteristics_multistate, pull_station_metadata_multistate

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_required_floor(cfg: dict) -> int:
    mins = cfg["research_question_minimums"]
    max_min = max(mins.values())
    return int(round(max_min * 1.30)) if cfg["targets"].get("require_30pct_above_minimum", True) else int(max_min)


def save_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".csv":
        df.to_csv(path, index=False)
    elif path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        raise ValueError(f"Unsupported extension: {path.suffix}")


def build_dataset(cfg: dict, project_root: Path) -> pd.DataFrame:
    state_codes = cfg["project"]["state_codes"]
    start_date = cfg["project"]["start_date"]
    end_date = cfg["project"]["end_date"]
    site_type = cfg["project"]["site_type"]

    required_floor = compute_required_floor(cfg)
    target_rows = max(required_floor, cfg["targets"]["final_min_rows"])
    LOGGER.info("Required minimum floor with 30%% buffer: %s rows", required_floor)
    LOGGER.info("Configured target rows: %s", target_rows)

    wqp_long = pull_characteristics_multistate(
        statecodes=state_codes,
        start_date=start_date,
        end_date=end_date,
        site_type=site_type,
        characteristics=cfg["wqp"]["characteristics"],
        synonym_map=cfg["wqp"]["characteristic_synonyms"],
    )
    if wqp_long.empty:
        raise RuntimeError("No WQP data returned. Check states, date range, and characteristic names.")

    save_df(wqp_long, project_root / "data" / "raw" / "wqp_long.csv")

    wqp_wide = aggregate_wqp_daily(wqp_long)
    if wqp_wide.empty:
        raise RuntimeError("WQP aggregation produced no rows.")

    save_df(wqp_wide, project_root / "data" / "interim" / "wqp_wide.csv")

    if cfg["project"].get("include_station_metadata", True):
        station_meta = pull_station_metadata_multistate(statecodes=state_codes, site_type=site_type)
        save_df(station_meta, project_root / "data" / "interim" / "site_metadata.csv")
        final_df = wqp_wide.merge(station_meta, on="site_id", how="left")
    else:
        final_df = wqp_wide.copy()

    final_df = add_time_features(final_df, date_col="date")
    final_df = add_lag_features(
        final_df,
        group_col="site_id",
        target_cols=["ph", "turbidity", "dissolved_oxygen"],
        lags=cfg["features"]["create_lags"],
    )
    final_df = add_rolling_mean_features(
        final_df,
        group_col="site_id",
        target_cols=["ph", "turbidity", "dissolved_oxygen"],
        windows=cfg["features"]["create_rolling_means"],
    )
    final_df = add_compliance_flag(
        final_df,
        turbidity_threshold=cfg["features"]["compliance_thresholds"]["turbidity_gt"],
    )
    final_df = filter_usable_sites(
        final_df,
        min_rows_per_site=cfg["quality_filters"]["min_rows_per_site"],
        min_nonnull_targets_per_site=cfg["quality_filters"]["min_nonnull_targets_per_site"],
    )

    save_df(final_df, project_root / "data" / "processed" / "final_dataset.csv")
    save_df(final_df, project_root / "data" / "processed" / "final_dataset.parquet")

    summary = {
        "state_codes": state_codes,
        "start_date": start_date,
        "end_date": end_date,
        "required_floor_with_30pct_buffer": required_floor,
        "configured_target_rows": target_rows,
        "final_row_count": int(len(final_df)),
        "final_site_count": int(final_df["site_id"].nunique()) if not final_df.empty else 0,
        "columns": list(final_df.columns),
        "nonnull_counts": final_df.notna().sum().to_dict() if not final_df.empty else {},
    }
    with open(project_root / "data" / "processed" / "extraction_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    LOGGER.info("Final dataset rows: %s", len(final_df))
    LOGGER.info("Final unique sites: %s", final_df["site_id"].nunique() if not final_df.empty else 0)

    if len(final_df) < required_floor:
        LOGGER.warning(
            "Final dataset (%s rows) is below the buffered minimum requirement (%s rows). "
            "Consider widening geography, adding years, or relaxing filters.",
            len(final_df), required_floor
        )

    return final_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    project_root = Path(args.config).resolve().parents[1]
    final_df = build_dataset(cfg, project_root)
    LOGGER.info("Done. Final dataset shape: %s", final_df.shape)


if __name__ == "__main__":
    main()
