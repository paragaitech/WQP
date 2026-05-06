# Data Description and Data Dictionary

## Data Description

This project uses a multi-state stream water-quality dataset built primarily
from the EPA Water Quality Portal (WQP), with optional hydrologic enrichment
from the U.S. Geological Survey (USGS) National Water Information System (NWIS)
daily values service. The dataset focuses on stream monitoring observations in
California, Oregon, Washington, Nevada, and Arizona from 2015-01-01 through
2024-12-31.

The main water-quality variables are pH, turbidity, and dissolved oxygen.
Station metadata, temporal features, lag features, rolling means, and optional
USGS hydrologic variables are added to support forecasting, explainability,
anomaly detection, compliance-risk classification, and model comparison.

The data files are included in this project repository so an evaluator can
access them through the provided GitHub repository link after the project is
pushed. The primary evaluator-ready files are:

- `data/raw/wqp_long.csv`
- `data/interim/wqp_wide.csv`
- `data/interim/site_metadata.csv`
- `data/processed/final_dataset.csv`
- `data/processed/final_dataset.parquet`
- `data/processed/final_dataset_enriched.csv`
- `data/processed/final_dataset_enriched.parquet`
- `data/processed/extraction_summary.json`
- `data/processed/usgs_enrichment_summary.json`

## Data Source(s) and Access

The dataset was generated from public web services:

| Source | Use in Project | Access URL |
|---|---|---|
| EPA Water Quality Portal (WQP) result service | Primary source for pH, turbidity, and dissolved oxygen observations | <https://www.waterqualitydata.us/data/Result/search> |
| EPA Water Quality Portal (WQP) station service | Source for monitoring-location metadata | <https://www.waterqualitydata.us/data/Station/search> |
| WQP Web Services Guide | Documentation for WQP service access and query structure | <https://www.waterqualitydata.us/webservices_documentation/> |
| WQP Description | Background on WQP, WQX, and NWIS integration | <https://www.waterqualitydata.us/wqp_description/> |
| USGS NWIS Daily Values service | Optional enrichment for discharge, water temperature, and gage height | <https://waterservices.usgs.gov/nwis/dv/> |
| USGS Daily Values service documentation | Documentation for USGS daily values query parameters and outputs | <https://nwis.waterservices.usgs.gov/docs/dv-service/> |

### APA 7 References

U.S. Geological Survey. (n.d.). *Daily values*. Water Services.
Retrieved May 6, 2026, from
<https://nwis.waterservices.usgs.gov/docs/dv-service/>

U.S. Geological Survey. (n.d.). *Daily values service details*. Water
Services. Retrieved May 6, 2026, from
<https://nwis.waterservices.usgs.gov/docs/dv-service/daily-values-service-details/>

Water Quality Portal. (n.d.). *WQP web services guide*. Retrieved May 6,
2026, from <https://www.waterqualitydata.us/webservices_documentation/>

Water Quality Portal. (n.d.). *What is the Water Quality Portal (WQP)*.
Retrieved May 6, 2026, from
<https://www.waterqualitydata.us/wqp_description/>

## Dataset Overview

The final enriched dataset is a daily, site-level analytical dataset. Each row
represents one monitoring site on one observation date. The dataset combines
observed water-quality measurements, station metadata, time-derived features,
lagged water-quality values, rolling water-quality averages, compliance
screening indicators, and optional USGS hydrologic variables where available.

- Number of records (rows): 144,849
- Number of variables (columns): 38
- Time period: 2015-01-01 to 2024-12-31
- Geographic scope: CA, OR, WA, NV, and AZ
- Unit of analysis: Monitoring site-date observation
- Site count: 2,236 unique monitoring sites
- Primary processed file: `data/processed/final_dataset_enriched.parquet`
- CSV equivalent: `data/processed/final_dataset_enriched.csv`

### Target Variable(s)

The target variables are water-quality outcomes used in forecasting,
explainability, anomaly detection, and compliance-risk analysis:

- `turbidity`: Water clarity/suspended-particle measurement. Used as a primary
  forecasting target, anomaly indicator, and compliance-risk screening input.
- `ph`: Acidity/alkalinity measurement. Used as a forecasting and driver
  analysis target.
- `dissolved_oxygen`: Oxygen concentration in water. Used as a forecasting and
  driver analysis target.
- `compliance_flag`: Binary turbidity screening indicator where `1` means
  turbidity exceeded the project threshold and `0` means it did not.

## Data Dictionary (Mandatory)

Table 1  
Data dictionary

| Variable Name | Definition | Datatype | Allowed Values/Range | Missing Values Handling | Notes |
|---|---|---|---|---|---|
| `site_id` | Unique monitoring site identifier from WQP/source records. | String/object | Valid WQP or USGS-style site identifier | Required for modeling; rows without site IDs should be excluded. | Used for grouping, joins, lag creation, rolling features, and site-level time ordering. |
| `date` | Observation or sampling date at daily granularity. | Datetime | 2015-01-01 to 2024-12-31 in this dataset | Required; invalid dates should be removed. | Used for chronological sorting, train-test splitting, lags, rolling means, and sequence models. |
| `dissolved_oxygen` | Dissolved oxygen concentration in water. | Float | Generally non-negative; implausible negative values treated as invalid | Missing target rows are excluded for dissolved oxygen-specific supervised modeling. | Main target for forecasting and driver analysis. |
| `ph` | Acidity or alkalinity measurement. | Float | Chemically valid range is 0 to 14 | Values outside 0-14 should be set to missing; target-specific missing rows excluded for pH models. | Main target for forecasting and driver analysis. |
| `turbidity` | Measure of water clarity and suspended particles. | Float | Usually >= 0 NTU; extreme spikes reviewed or capped depending on preprocessing mode | Negative values should be set to missing; target-specific missing rows excluded for turbidity models. | Primary target for forecasting, anomaly detection, and compliance-risk screening. |
| `site_name` | Descriptive monitoring location name. | String/object | Valid station/location name | Retained for metadata; not required for numeric modeling. | Useful for auditing and reporting. |
| `site_type_name` | Type or context of monitoring location. | String/object | WQP site type values such as stream, canal, reservoir, or similar location types | Retained as metadata; encode if used in categorical modeling. | Helps describe monitoring context. |
| `huc8` | Eight-digit hydrologic unit code. | Float | Valid HUC8 watershed identifier where available | Missing values retained unless watershed-level analysis requires them. | Useful for watershed grouping and spatial interpretation. |
| `state_code` | Numeric state FIPS code. | Integer | 4 = AZ, 6 = CA, 32 = NV, 41 = OR, 53 = WA | Required for state-level filtering; missing values should be reviewed. | Can be mapped to state abbreviations for modeling and reporting. |
| `county_code` | County FIPS code from station metadata. | Float | Valid county code where available | Missing values retained unless county-level analysis requires them. | Metadata field for location context. |
| `organization_name` | Organization responsible for the monitoring record or station. | String/object | Valid organization name | Retained for metadata; not usually used directly in numeric models. | Useful for source auditing. |
| `month` | Calendar month extracted from `date`. | Integer | 1 to 12 | Recomputed from valid date if missing. | Seasonal predictor. |
| `quarter` | Calendar quarter extracted from `date`. | Integer | 1 to 4 | Recomputed from valid date if missing. | Seasonal predictor. |
| `day_of_year` | Day number within the calendar year. | Integer | 1 to 366 | Recomputed from valid date if missing. | Captures annual seasonality. |
| `year` | Calendar year extracted from `date`. | Integer | 2015 to 2024 | Recomputed from valid date if missing. | Used for temporal trends and splitting. |
| `is_warm_season` | Indicator for warm-season months. | Integer | 0 or 1 | Recomputed from `month` if missing. | Project feature representing seasonal water-quality conditions. |
| `ph_lag1` | Previous observation pH for the same site. | Float | Same practical range as `ph` | Structural missingness expected for early site records; impute or drop depending on model. | Temporal predictor. |
| `ph_lag3` | pH from three prior observations for the same site. | Float | Same practical range as `ph` | Structural missingness expected for sites with insufficient history. | Temporal predictor. |
| `ph_lag7` | pH from seven prior observations for the same site. | Float | Same practical range as `ph` | Structural missingness expected for sites with insufficient history. | Longer-history temporal predictor. |
| `turbidity_lag1` | Previous observation turbidity for the same site. | Float | Usually >= 0 | Structural missingness expected for early site records. | Important turbidity forecasting predictor. |
| `turbidity_lag3` | Turbidity from three prior observations for the same site. | Float | Usually >= 0 | Structural missingness expected for sites with insufficient history. | Temporal predictor. |
| `turbidity_lag7` | Turbidity from seven prior observations for the same site. | Float | Usually >= 0 | Structural missingness expected for sites with insufficient history. | Longer-history temporal predictor. |
| `dissolved_oxygen_lag1` | Previous observation dissolved oxygen for the same site. | Float | Generally non-negative | Structural missingness expected for early site records. | Important dissolved oxygen forecasting predictor. |
| `dissolved_oxygen_lag3` | Dissolved oxygen from three prior observations for the same site. | Float | Generally non-negative | Structural missingness expected for sites with insufficient history. | Temporal predictor. |
| `dissolved_oxygen_lag7` | Dissolved oxygen from seven prior observations for the same site. | Float | Generally non-negative | Structural missingness expected for sites with insufficient history. | Longer-history temporal predictor. |
| `ph_rollmean7` | Rolling mean of recent pH values for the same site using a 7-observation window. | Float | Same practical range as `ph` | Structural missingness expected until enough prior observations exist. | Strong driver for pH forecasting. |
| `ph_rollmean14` | Rolling mean of recent pH values for the same site using a 14-observation window. | Float | Same practical range as `ph` | Structural missingness expected until enough prior observations exist. | Longer-window pH temporal feature. |
| `turbidity_rollmean7` | Rolling mean of recent turbidity values for the same site using a 7-observation window. | Float | Usually >= 0 | Structural missingness expected until enough prior observations exist. | Strong predictor for turbidity behavior. |
| `turbidity_rollmean14` | Rolling mean of recent turbidity values for the same site using a 14-observation window. | Float | Usually >= 0 | Structural missingness expected until enough prior observations exist. | Longer-window turbidity temporal feature. |
| `dissolved_oxygen_rollmean7` | Rolling mean of recent dissolved oxygen values for the same site using a 7-observation window. | Float | Generally non-negative | Structural missingness expected until enough prior observations exist. | Strong predictor for dissolved oxygen behavior. |
| `dissolved_oxygen_rollmean14` | Rolling mean of recent dissolved oxygen values for the same site using a 14-observation window. | Float | Generally non-negative | Structural missingness expected until enough prior observations exist. | Longer-window dissolved oxygen temporal feature. |
| `compliance_flag` | Binary turbidity threshold flag created by the project pipeline. | Nullable integer | 0 or 1 | Created from turbidity when available; missing if turbidity is unavailable in alternate workflows. | In the current pipeline, threshold is turbidity > 5.0. Exclude from predictors when it would leak the target. |
| `usgs_site_no_x` | Intermediate USGS site number field from merge output. | String/object | Valid USGS site number where available | Mostly missing; retained as provenance from merge process. | Redundant/intermediate field; prefer `usgs_site_no` for analysis. |
| `usgs_site_no_y` | Intermediate USGS site number field from merge output. | String/object | Valid USGS site number where available | Mostly missing; retained as provenance from merge process. | Redundant/intermediate field; prefer `usgs_site_no` for analysis. |
| `discharge` | USGS daily stream discharge enrichment. | Float | Generally >= 0, typically cubic feet per second depending on USGS parameter metadata | Missing expected where no USGS match exists; may be imputed for hydrologic modeling tracks. | USGS parameter code `00060`. |
| `gage_height` | USGS daily gage height enrichment. | Float | Site-specific valid range, generally feet depending on USGS parameter metadata | Missing expected where no USGS match exists; may be imputed for hydrologic modeling tracks. | USGS parameter code `00065`. |
| `water_temp` | USGS daily water temperature enrichment. | Float | Plausible environmental water temperature range; units follow USGS parameter metadata | Missing expected where no USGS match exists; may be imputed for hydrologic modeling tracks. | USGS parameter code `00010`. |
| `usgs_site_no` | Cleaned USGS site number extracted from WQP site IDs where available. | String/object | Valid USGS station number | Missing where WQP site does not map to a USGS station. | Join key for USGS hydrologic enrichment. |

## Notes on Missing Values

Missing values are expected in this dataset. Some missingness is structural:
early records at each monitoring site cannot have lag or rolling-window values.
Other missingness comes from partial overlap between WQP monitoring sites and
USGS daily hydrologic stations. Modeling notebooks handle missing values
differently by task, including target-specific row exclusion, median imputation,
site-month hydrologic imputation, and optional outlier capping.
