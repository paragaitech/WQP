from __future__ import annotations

import pandas as pd


def add_time_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out["month"] = out[date_col].dt.month
    out["quarter"] = out[date_col].dt.quarter
    out["day_of_year"] = out[date_col].dt.dayofyear
    out["year"] = out[date_col].dt.year
    out["is_warm_season"] = out["month"].isin([4, 5, 6, 7, 8, 9]).astype(int)
    return out


def add_lag_features(df: pd.DataFrame, group_col: str, target_cols: list[str], lags: list[int]) -> pd.DataFrame:
    out = df.sort_values([group_col, "date"]).copy()
    for col in target_cols:
        if col not in out.columns:
            continue
        for lag in lags:
            out[f"{col}_lag{lag}"] = out.groupby(group_col)[col].shift(lag)
    return out


def add_rolling_mean_features(df: pd.DataFrame, group_col: str, target_cols: list[str], windows: list[int]) -> pd.DataFrame:
    out = df.sort_values([group_col, "date"]).copy()
    for col in target_cols:
        if col not in out.columns:
            continue
        for window in windows:
            out[f"{col}_rollmean{window}"] = (
                out.groupby(group_col)[col]
                .transform(lambda s: s.rolling(window, min_periods=max(2, window // 2)).mean())
            )
    return out


def add_compliance_flag(df: pd.DataFrame, turbidity_threshold: float = 5.0) -> pd.DataFrame:
    out = df.copy()
    if "turbidity" in out.columns:
        out["compliance_flag"] = (out["turbidity"] > turbidity_threshold).astype("Int64")
    else:
        out["compliance_flag"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    return out


def filter_usable_sites(
    df: pd.DataFrame,
    min_rows_per_site: int = 10,
    min_nonnull_targets_per_site: int = 1,
    target_cols: list[str] | None = None,
) -> pd.DataFrame:
    target_cols = target_cols or ["ph", "turbidity", "dissolved_oxygen"]

    usable_sites = []
    for site_id, g in df.groupby("site_id"):
        nonnull_targets = sum(col in g.columns and g[col].notna().sum() > 0 for col in target_cols)
        if len(g) >= min_rows_per_site and nonnull_targets >= min_nonnull_targets_per_site:
            usable_sites.append(site_id)

    return df[df["site_id"].isin(usable_sites)].copy()
