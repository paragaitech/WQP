from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Optional

import pandas as pd
import requests
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)
USGS_DV_URL = "https://waterservices.usgs.gov/nwis/dv/"


def normalize_usgs_date(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = str(value).strip()
    for fmt in ("%Y-%m-%d", "%m-%d-%Y"):
        try:
            dt = datetime.strptime(value, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    raise ValueError(f"Invalid USGS date '{value}'. Use YYYY-MM-DD or MM-DD-YYYY.")


def extract_usgs_site_no(site_id: str) -> Optional[str]:
    if pd.isna(site_id):
        return None
    site_id = str(site_id)
    if site_id.startswith("USGS-"):
        return site_id.split("USGS-")[-1]
    return None


def fetch_usgs_daily(site_no: str, parameter_cd: str, start_date: str, end_date: str, site_status: str = "all") -> pd.DataFrame:
    params = {
        "format": "json",
        "sites": site_no,
        "parameterCd": parameter_cd,
        "startDT": normalize_usgs_date(start_date),
        "endDT": normalize_usgs_date(end_date),
        "siteStatus": site_status,
    }
    response = requests.get(USGS_DV_URL, params=params, timeout=180)
    if response.status_code >= 400:
        LOGGER.warning("USGS request failed: %s", response.url)
        LOGGER.warning("USGS response (first 300 chars): %s", response.text[:300])
    response.raise_for_status()

    payload = response.json()
    rows = []
    for ts in payload.get("value", {}).get("timeSeries", []):
        variable_name = ts.get("variable", {}).get("variableName", "")
        for values_block in ts.get("values", []):
            for point in values_block.get("value", []):
                rows.append({
                    "usgs_site_no": site_no,
                    "date": pd.to_datetime(point.get("dateTime"), errors="coerce").date(),
                    "parameter_cd": parameter_cd,
                    "variable_name": variable_name,
                    "value": pd.to_numeric(point.get("value"), errors="coerce"),
                })
    return pd.DataFrame(rows)


def batch_fetch_usgs_daily(site_list, parameter_map: dict[str, str], start_date: str, end_date: str, site_status: str = "all", pause_seconds: float = 0.0) -> pd.DataFrame:
    frames = []
    for site_no in tqdm(list(site_list), desc="USGS enrichment"):
        for parameter_cd, feature_name in parameter_map.items():
            try:
                df = fetch_usgs_daily(site_no, parameter_cd, start_date, end_date, site_status=site_status)
                if not df.empty:
                    df["feature_name"] = feature_name
                    frames.append(df)
            except Exception as exc:
                LOGGER.warning("USGS pull failed for site=%s pcode=%s: %s", site_no, parameter_cd, exc)
            if pause_seconds:
                time.sleep(pause_seconds)

    if not frames:
        return pd.DataFrame(columns=["usgs_site_no", "date", "feature_name", "value"])

    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["date", "value"])
    return out[["usgs_site_no", "date", "feature_name", "value"]]


def aggregate_usgs_daily(long_df: pd.DataFrame) -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame(columns=["usgs_site_no", "date"])
    grouped = long_df.groupby(["usgs_site_no", "date", "feature_name"], as_index=False)["value"].mean()
    wide = grouped.pivot(index=["usgs_site_no", "date"], columns="feature_name", values="value").reset_index()
    wide.columns.name = None
    return wide


def enrich_with_usgs_exact(base_df: pd.DataFrame, usgs_wide: pd.DataFrame) -> pd.DataFrame:
    out = base_df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    temp_usgs = usgs_wide.copy()
    temp_usgs["date"] = pd.to_datetime(temp_usgs["date"], errors="coerce")
    return out.merge(temp_usgs, on=["usgs_site_no", "date"], how="left")


# def enrich_with_usgs_nearest_prior(base_df: pd.DataFrame, usgs_wide: pd.DataFrame, tolerance_days: int = 7) -> pd.DataFrame:
#     left = base_df.copy()
#     right = usgs_wide.copy()

#     left["date"] = pd.to_datetime(left["date"], errors="coerce")
#     right["date"] = pd.to_datetime(right["date"], errors="coerce")

#     left = left.sort_values(["usgs_site_no", "date"])
#     right = right.sort_values(["usgs_site_no", "date"])

#     enriched = pd.merge_asof(
#         left,
#         right,
#         on="date",
#         by="usgs_site_no",
#         direction="backward",
#         tolerance=pd.Timedelta(days=tolerance_days),
#     )
#     return enriched

# def enrich_with_usgs_nearest_prior(base_df: pd.DataFrame, usgs_wide: pd.DataFrame, tolerance_days: int = 7) -> pd.DataFrame:
#     left = base_df.copy()
#     right = usgs_wide.copy()

#     left["date"] = pd.to_datetime(left["date"], errors="coerce")
#     right["date"] = pd.to_datetime(right["date"], errors="coerce")

#     # Separate rows that cannot participate in USGS merge
#     left_no_usgs = left[left["usgs_site_no"].isna()].copy()
#     left_yes_usgs = left[left["usgs_site_no"].notna()].copy()

#     # Drop rows with invalid dates before merge_asof
#     left_yes_usgs = left_yes_usgs.dropna(subset=["date", "usgs_site_no"]).copy()
#     right = right.dropna(subset=["date", "usgs_site_no"]).copy()

#     # Make sure join keys are strings and consistently typed
#     left_yes_usgs["usgs_site_no"] = left_yes_usgs["usgs_site_no"].astype(str)
#     right["usgs_site_no"] = right["usgs_site_no"].astype(str)

#     # Strict sorting required by merge_asof
#     left_yes_usgs = left_yes_usgs.sort_values(["usgs_site_no", "date"]).reset_index(drop=True)
#     right = right.sort_values(["usgs_site_no", "date"]).reset_index(drop=True)

#     enriched_yes_usgs = pd.merge_asof(
#         left_yes_usgs,
#         right,
#         on="date",
#         by="usgs_site_no",
#         direction="backward",
#         tolerance=pd.Timedelta(days=tolerance_days),
#     )

#     # Reattach rows that had no USGS site number
#     enriched = pd.concat([enriched_yes_usgs, left_no_usgs], ignore_index=True, sort=False)

#     # Restore original order as much as possible
#     if "site_id" in enriched.columns and "date" in enriched.columns:
#         enriched = enriched.sort_values(["site_id", "date"]).reset_index(drop=True)

#     return enriched

def enrich_with_usgs_nearest_prior(base_df: pd.DataFrame, usgs_wide: pd.DataFrame, tolerance_days: int = 7) -> pd.DataFrame:
    """
    Robust nearest-prior enrichment performed site-by-site to avoid pandas
    merge_asof sorting issues on grouped merges.
    """
    left = base_df.copy()
    right = usgs_wide.copy()

    left["date"] = pd.to_datetime(left["date"], errors="coerce")
    right["date"] = pd.to_datetime(right["date"], errors="coerce")

    # Keep rows with no USGS site number unchanged
    left_no_usgs = left[left["usgs_site_no"].isna()].copy()
    left_yes_usgs = left[left["usgs_site_no"].notna()].copy()

    # Clean keys
    left_yes_usgs = left_yes_usgs.dropna(subset=["date", "usgs_site_no"]).copy()
    right = right.dropna(subset=["date", "usgs_site_no"]).copy()

    left_yes_usgs["usgs_site_no"] = left_yes_usgs["usgs_site_no"].astype(str)
    right["usgs_site_no"] = right["usgs_site_no"].astype(str)

    enriched_parts = []

    right_groups = {site: g.sort_values("date").reset_index(drop=True)
                    for site, g in right.groupby("usgs_site_no")}

    for site_no, left_group in left_yes_usgs.groupby("usgs_site_no"):
        left_group = left_group.sort_values("date").reset_index(drop=True)

        if site_no not in right_groups:
            enriched_parts.append(left_group)
            continue

        right_group = right_groups[site_no]

        merged = pd.merge_asof(
            left_group,
            right_group,
            on="date",
            direction="backward",
            tolerance=pd.Timedelta(days=tolerance_days),
        )

        # Put the grouping key back explicitly
        merged["usgs_site_no"] = site_no
        enriched_parts.append(merged)

    if enriched_parts:
        enriched_yes_usgs = pd.concat(enriched_parts, ignore_index=True, sort=False)
    else:
        enriched_yes_usgs = left_yes_usgs.copy()

    enriched = pd.concat([enriched_yes_usgs, left_no_usgs], ignore_index=True, sort=False)

    if "site_id" in enriched.columns and "date" in enriched.columns:
        enriched = enriched.sort_values(["site_id", "date"]).reset_index(drop=True)

    return enriched

