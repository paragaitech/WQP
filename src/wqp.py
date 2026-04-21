from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Optional

import pandas as pd
import requests

LOGGER = logging.getLogger(__name__)

WQP_RESULT_URL = "https://www.waterqualitydata.us/data/Result/search"
WQP_STATION_URL = "https://www.waterqualitydata.us/data/Station/search"

STATE_FIPS = {
    "AL": "01", "AK": "02", "AZ": "04", "AR": "05", "CA": "06", "CO": "08",
    "CT": "09", "DE": "10", "FL": "12", "GA": "13", "HI": "15", "ID": "16",
    "IL": "17", "IN": "18", "IA": "19", "KS": "20", "KY": "21", "LA": "22",
    "ME": "23", "MD": "24", "MA": "25", "MI": "26", "MN": "27", "MS": "28",
    "MO": "29", "MT": "30", "NE": "31", "NV": "32", "NH": "33", "NJ": "34",
    "NM": "35", "NY": "36", "NC": "37", "ND": "38", "OH": "39", "OK": "40",
    "OR": "41", "PA": "42", "RI": "44", "SC": "45", "SD": "46", "TN": "47",
    "TX": "48", "UT": "49", "VT": "50", "VA": "51", "WA": "53", "WV": "54",
    "WI": "55", "WY": "56",
}


@dataclass
class WQPQuery:
    statecode: str
    characteristic_name: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    site_type: str = "Stream"
    mime_type: str = "csv"


def normalize_statecode(value: str) -> str:
    value = str(value).strip().upper()
    if value.startswith("US:"):
        suffix = value.split("US:")[-1]
        if suffix.isdigit() and len(suffix) == 2:
            return f"US:{suffix}"
        if suffix in STATE_FIPS:
            return f"US:{STATE_FIPS[suffix]}"
        raise ValueError(f"Unrecognized statecode: {value}")
    if value.isdigit() and len(value) == 2:
        return f"US:{value}"
    if value in STATE_FIPS:
        return f"US:{STATE_FIPS[value]}"
    raise ValueError(f"Invalid statecode: {value}")


def normalize_wqp_date(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = str(value).strip()
    for fmt in ("%Y-%m-%d", "%m-%d-%Y"):
        try:
            dt = datetime.strptime(value, fmt)
            return dt.strftime("%m-%d-%Y")
        except ValueError:
            continue
    raise ValueError(f"Invalid WQP date: {value}")


def fetch_wqp_results(query: WQPQuery) -> pd.DataFrame:
    params = {
        "statecode": normalize_statecode(query.statecode),
        "startDateLo": normalize_wqp_date(query.start_date),
        "startDateHi": normalize_wqp_date(query.end_date),
        "siteType": query.site_type,
        "mimeType": query.mime_type,
        "sorted": "no",
    }
    if query.characteristic_name:
        params["characteristicName"] = query.characteristic_name

    response = requests.get(WQP_RESULT_URL, params=params, timeout=180)
    if response.status_code >= 400:
        LOGGER.warning("WQP request failed: %s", response.url)
        LOGGER.warning("WQP response (first 500 chars): %s", response.text[:500])
    response.raise_for_status()
    if not response.text.strip():
        return pd.DataFrame()
    return pd.read_csv(io.StringIO(response.text), low_memory=False)


def fetch_wqp_stations(statecode: str, site_type: str = "Stream", mime_type: str = "csv") -> pd.DataFrame:
    params = {
        "statecode": normalize_statecode(statecode),
        "siteType": site_type,
        "mimeType": mime_type,
        "sorted": "no",
    }
    response = requests.get(WQP_STATION_URL, params=params, timeout=180)
    if response.status_code >= 400:
        LOGGER.warning("WQP station request failed: %s", response.url)
        LOGGER.warning("WQP station response (first 500 chars): %s", response.text[:500])
    response.raise_for_status()
    if not response.text.strip():
        return pd.DataFrame()
    return pd.read_csv(io.StringIO(response.text), low_memory=False)


def standardize_wqp_results(df: pd.DataFrame, canonical_parameter: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["site_id", "date", "parameter", "value", "unit"])

    col_map = {
        "MonitoringLocationIdentifier": "site_id",
        "ActivityStartDate": "activity_date",
        "CharacteristicName": "characteristic_name",
        "ResultMeasureValue": "value_raw",
        "ResultMeasure/MeasureUnitCode": "unit",
        "ActivityIdentifier": "activity_id",
        "ResultStatusIdentifier": "result_status",
    }
    existing = {k: v for k, v in col_map.items() if k in df.columns}
    out = df.rename(columns=existing).copy()

    required = ["site_id", "activity_date", "value_raw"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise KeyError(f"Missing required WQP columns: {missing}")

    out["date"] = pd.to_datetime(out["activity_date"], errors="coerce").dt.date
    out["value"] = pd.to_numeric(out["value_raw"], errors="coerce")
    out["parameter"] = canonical_parameter
    out["unit"] = out.get("unit", pd.Series(index=out.index, dtype="object"))
    out = out.dropna(subset=["site_id", "date", "value"])
    return out[["site_id", "date", "parameter", "value", "unit"]]


def aggregate_wqp_daily(long_df: pd.DataFrame) -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame(columns=["site_id", "date"])
    grouped = (
        long_df.groupby(["site_id", "date", "parameter"], as_index=False)["value"]
        .mean()
    )
    wide = grouped.pivot(index=["site_id", "date"], columns="parameter", values="value").reset_index()
    wide.columns.name = None
    return wide


def standardize_station_metadata(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["site_id"])

    rename_map = {
        "MonitoringLocationIdentifier": "site_id",
        "MonitoringLocationName": "site_name",
        "MonitoringLocationLatitude": "latitude",
        "MonitoringLocationLongitude": "longitude",
        "MonitoringLocationTypeName": "site_type_name",
        "HUCEightDigitCode": "huc8",
        "StateCode": "state_code",
        "CountyCode": "county_code",
        "OrganizationFormalName": "organization_name",
    }
    existing = {k: v for k, v in rename_map.items() if k in df.columns}
    out = df.rename(columns=existing).copy()

    keep_cols = [c for c in [
        "site_id", "site_name", "latitude", "longitude", "site_type_name",
        "huc8", "state_code", "county_code", "organization_name"
    ] if c in out.columns]

    out = out[keep_cols].drop_duplicates(subset=["site_id"])
    return out


def pull_characteristics_multistate(
    statecodes: Iterable[str],
    start_date: str,
    end_date: str,
    site_type: str,
    characteristics: Iterable[str],
    synonym_map: dict[str, list[str]],
) -> pd.DataFrame:
    frames = []

    for statecode in statecodes:
        for canonical in characteristics:
            synonyms = synonym_map.get(canonical, [canonical])
            pulled = False

            for term in synonyms:
                LOGGER.info(
                    "Pulling WQP characteristic '%s' for canonical '%s' in state '%s'",
                    term, canonical, statecode
                )
                try:
                    df = fetch_wqp_results(
                        WQPQuery(
                            statecode=statecode,
                            characteristic_name=term,
                            start_date=start_date,
                            end_date=end_date,
                            site_type=site_type,
                        )
                    )
                except requests.HTTPError as exc:
                    LOGGER.warning(
                        "WQP pull failed for state=%s characteristic='%s' canonical='%s': %s",
                        statecode, term, canonical, exc
                    )
                    continue

                if not df.empty:
                    frames.append(
                        standardize_wqp_results(
                            df,
                            canonical_parameter=canonical.lower().replace(" ", "_")
                        )
                    )
                    pulled = True
                    break

            if not pulled:
                LOGGER.warning(
                    "No WQP data returned for state '%s' canonical characteristic '%s'",
                    statecode, canonical
                )

    if not frames:
        return pd.DataFrame(columns=["site_id", "date", "parameter", "value", "unit"])

    return pd.concat(frames, ignore_index=True)


def pull_station_metadata_multistate(statecodes: Iterable[str], site_type: str = "Stream") -> pd.DataFrame:
    frames = []
    for statecode in statecodes:
        LOGGER.info("Pulling WQP station metadata for state '%s'", statecode)
        try:
            df = fetch_wqp_stations(statecode=statecode, site_type=site_type)
        except requests.HTTPError as exc:
            LOGGER.warning("WQP station pull failed for state=%s: %s", statecode, exc)
            continue
        if not df.empty:
            frames.append(standardize_station_metadata(df))

    if not frames:
        return pd.DataFrame(columns=["site_id"])
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["site_id"])
