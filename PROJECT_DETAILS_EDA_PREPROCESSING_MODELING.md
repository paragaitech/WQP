# Project Details: EDA, Data Preprocessing, and Modeling

This document summarizes the exploratory data analysis, preprocessing strategy,
and modeling workflow reviewed from
`/Users/prane/Downloads/wqp_eda_preprocessing_modelling.ipynb`.

## Notebook Purpose

The notebook supports the project **AI-Driven Water Quality Prediction and
Monitoring: A Unified Framework for Compliance Intelligence**. It extends the
dataset-building pipeline by preparing the processed WQP/USGS dataset for
exploratory analysis, feature cleanup, forecasting, driver analysis, anomaly
detection, compliance-risk classification, and model comparison.

The modeling dataset used in the notebook is the enriched final dataset, with
144,849 observations and 38 initial columns before notebook-specific
preprocessing.

## Analytical Scope

The notebook is organized around five research questions:

| Research Question | Focus | Evaluation |
|---|---|---|
| RQ1 | Forecast next-period turbidity, pH, and dissolved oxygen | RMSE, MAE, R2 |
| RQ2 | Identify hydrologic, seasonal, lagged, and rolling-window drivers | OLS significance, feature importance, SHAP-ready outputs |
| RQ3 | Detect abnormal water-quality behavior | Precision, Recall, F1, ROC-AUC |
| RQ4 | Predict future state-aware compliance-risk events | Precision, Recall, F1, ROC-AUC |
| RQ5 | Compare statistical, ML, and DL model families | Cross-model forecasting performance |

## Data Used

The notebook uses the enriched WQP dataset, which includes:

- Core water-quality targets: `turbidity`, `ph`, and `dissolved_oxygen`
- Site metadata: `site_id`, `site_name`, `site_type_name`, `huc8`,
  `state_code`, `county_code`, and `organization_name`
- Time features: `month`, `quarter`, `day_of_year`, `year`, and
  `is_warm_season`
- Lag features: 1-, 3-, and 7-observation lags for pH, turbidity, and
  dissolved oxygen
- Rolling means: 7- and 14-observation rolling means for pH, turbidity, and
  dissolved oxygen
- Compliance flag: a basic turbidity threshold flag
- Optional USGS enrichment fields: `discharge`, `gage_height`, `water_temp`,
  and USGS site identifiers

## EDA Before Preprocessing

The raw enriched dataset contained substantial missingness and extreme values.
The highest missing-value rates were concentrated in USGS hydrologic fields:

| Variable | Missing % Before Preprocessing |
|---|---:|
| `gage_height` | 99.49 |
| `water_temp` | 96.50 |
| `discharge` | 95.54 |
| `usgs_site_no` | 91.36 |
| `ph_lag7` | 40.11 |
| `dissolved_oxygen_lag7` | 38.89 |
| `ph_rollmean14` | 38.69 |
| `turbidity` | 27.04 |

The notebook identified three major data-quality issues:

- Hydrologic fields were sparse because only some WQP sites map cleanly to
  USGS stations.
- Lag and rolling-window features introduced structural missingness because
  early records at each site do not have enough prior observations.
- Some water-quality measurements contained physically implausible or extreme
  values, including negative turbidity, negative dissolved oxygen, and pH
  values outside the chemically valid 0-14 range.

Before preprocessing, descriptive statistics showed strong skew and invalid
extremes:

| Variable | Mean | Median | Max | Key Issue |
|---|---:|---:|---:|---|
| `turbidity` | 21.31 | 3.10 | 24,800.00 | Large right tail and event-driven spikes |
| `dissolved_oxygen` | 12.98 | 9.96 | 1,040.00 | Invalid negative and extreme high values |
| `ph` | 7.53 | 7.50 | 860.00 | Values outside plausible chemical range |

Raw pairwise correlations among pH, turbidity, and dissolved oxygen were weak.
The notebook notes that the strongest useful relationships appeared between
each target and its own lagged or rolling features, indicating strong temporal
dependence.

## Preprocessing Controls

The notebook defines three main preprocessing controls:

| Control | Setting Used | Purpose |
|---|---|---|
| `IMPUTE_HYDROLOGY` | `True` | Impute missing hydrologic predictors instead of dropping most enriched rows |
| `OUTLIER_MODE` | `"cap"` | Winsorize extreme values using IQR bounds |
| `STATE_AWARE_FLAG` | `True` | Create state-aware turbidity compliance-risk screening flags |

## Data Preprocessing Steps

The preprocessing workflow applies these transformations:

1. State codes are mapped to state abbreviations for AZ, CA, NV, OR, and WA.
2. Invalid pH values below 0 or above 14 are set to missing.
3. Negative dissolved oxygen values are set to missing.
4. Negative turbidity values are set to missing.
5. Calendar features are recalculated from the date field.
6. Missing hydrologic values are imputed using site-month mean, then site mean,
   then global mean.
7. State-aware compliance-risk flags are created from turbidity and estimated
   site background turbidity.
8. Extreme values are capped using IQR-based lower and upper bounds.
9. Data is sorted chronologically by state, site, and date before splitting or
   sequence modeling.

## State-Aware Compliance Screening

The notebook creates `background_turbidity_est` using prior site-level
turbidity behavior. It then applies state-aware screening rules:

| State | Screening Rule Used |
|---|---|
| OR | Flag if turbidity is more than 10% above background |
| WA | Flag if turbidity is more than 5 NTU above background when background is <= 50, otherwise more than 10% above background |
| CA | Fallback flag if turbidity is more than 20% above background |
| AZ | Fallback flag if turbidity is more than 20% above background |
| NV | Fallback flag if turbidity is more than 20% above background |

The future version of this flag is used for compliance-risk classification in
RQ4, while same-row compliance variables and same-row water-quality targets are
excluded from RQ4 predictors to reduce leakage.

## EDA After Preprocessing

After preprocessing, the dataset contained 144,849 rows and 42 columns. The
hydrologic fields were imputed, while structural missingness remained in lag,
rolling, and target columns.

| Variable | Missing % After Preprocessing |
|---|---:|
| `usgs_site_no` | 91.36 |
| `ph_lag7` | 40.11 |
| `dissolved_oxygen_lag7` | 38.89 |
| `ph_rollmean14` | 38.69 |
| `turbidity_lag7` | 34.95 |
| `ph` | 32.52 |
| `dissolved_oxygen` | 31.21 |
| `turbidity` | 29.92 |
| `discharge` | 0.00 |
| `gage_height` | 0.00 |
| `water_temp` | 0.00 |

The capped and cleaned descriptive statistics became more physically plausible:

| Variable | Mean | Median | Max |
|---|---:|---:|---:|
| `turbidity` | 6.42 | 3.40 | 20.73 |
| `dissolved_oxygen` | 10.04 | 9.96 | 15.77 |
| `ph` | 7.52 | 7.50 | 9.19 |

Post-preprocessing correlations among the three raw targets remained weak,
supporting the need for nonlinear models and engineered temporal features.

## Dataset Tracks

The notebook creates two parallel analysis tracks:

| Track | Shape | Purpose |
|---|---:|---|
| Track A | 144,849 rows x 42 columns | Forecasting and anomaly detection, prioritizing row preservation |
| Track B | 144,849 rows x 42 columns | Driver analysis and compliance modeling with hydrologic predictors |

Track A uses broad numeric forecasting features while excluding compliance
flags from forecasting predictors. Track B focuses on hydrologic, seasonal,
lagged, and rolling-window variables for explanation and compliance-risk
classification.

## Modeling Methods

The notebook evaluates a mix of statistical, machine learning, and deep
learning methods:

- Linear Regression as an interpretable statistical baseline
- Random Forest for nonlinear tabular modeling and feature importance
- XGBoost for boosted nonlinear prediction
- LightGBM and CatBoost as optional gradient-boosting models
- Weighted ensemble forecasts from tree-based models
- LSTM for 14-observation sequence forecasting
- TCN-style causal Conv1D model for sequence forecasting
- OLS and Random Forest feature importance for driver analysis
- Isolation Forest and Autoencoder for anomaly detection
- Logistic Regression, Random Forest, XGBoost, and LightGBM for
  compliance-risk classification
- Optional TabNet experiments, which were not run because `pytorch-tabnet` was
  unavailable in the notebook environment

All supervised tabular modeling sections use chronological 80/20 train-test
splits. Missing predictors are generally median-imputed, and neural models use
MinMax scaling. Sequence models use 14 prior observations per site to predict
the next target value.

## RQ1 Forecasting Results

RQ1 evaluates next-period forecasting for turbidity, dissolved oxygen, and pH.

| Target | Best Tabular Model | RMSE | MAE | R2 |
|---|---|---:|---:|---:|
| `turbidity` | XGBoost | 2.023 | 1.350 | 0.907 |
| `dissolved_oxygen` | Random Forest | 1.198 | 0.780 | 0.683 |
| `ph` | Weighted Ensemble | 0.327 | 0.235 | 0.621 |

Linear Regression underperformed the nonlinear models across all targets,
especially pH and dissolved oxygen. Tree-based models were strongest for
turbidity, which behaves as a nonlinear and event-driven parameter.

## Sequence Forecasting Results

The LSTM and TCN experiments use 14-observation sequences within each site.

| Target | LSTM R2 | TCN R2 | Stronger Sequence Model |
|---|---:|---:|---|
| `turbidity` | 0.510 | 0.492 | LSTM |
| `dissolved_oxygen` | 0.837 | 0.860 | TCN |
| `ph` | 0.674 | 0.678 | TCN |

Sequence models performed especially well for dissolved oxygen and pH,
indicating strong temporal continuity. They were weaker for turbidity than
tree-based models, likely because turbidity is more episodic and event-driven.

## RQ2 Driver Analysis Results

RQ2 examines which variables explain short-term water-quality behavior.

| Target | Best RQ2 Model | RMSE | MAE | R2 |
|---|---|---:|---:|---:|
| `turbidity` | XGBoost | 2.305 | 1.482 | 0.879 |
| `dissolved_oxygen` | XGBoost | 1.160 | 0.757 | 0.703 |
| `ph` | XGBoost | 0.331 | 0.235 | 0.611 |

Key driver findings:

- Turbidity was most strongly explained by the current compliance flags,
  turbidity rolling means, and recent turbidity lags.
- Dissolved oxygen was dominated by `dissolved_oxygen_rollmean7` and
  `dissolved_oxygen_lag1`, with seasonal variables also significant.
- pH was dominated by `ph_rollmean7`, followed by pH lags and seasonal
  indicators.
- OLS p-values confirmed that rolling means, lag terms, and warm-season effects
  were statistically meaningful for multiple targets.

The notebook positions SHAP analysis as the next interpretability step for
formal model explanation.

## RQ3 Anomaly Detection Results

RQ3 creates an anomaly label from rule-based abnormal conditions:

- Turbidity above the 98th percentile
- Dissolved oxygen below the 2nd percentile
- pH below 6.5 or above 8.5

The evaluated models were Threshold Baseline, Isolation Forest, and Autoencoder.

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| Autoencoder | 0.090 | 0.072 | 0.080 | 0.512 |
| Isolation Forest | 0.118 | 0.007 | 0.014 | 0.502 |
| Threshold Baseline | 0.000 | 0.000 | 0.000 | 0.500 |

The Autoencoder performed best overall, but absolute performance remained
modest. The notebook identifies anomaly detection as challenging because of
class imbalance, noisy environmental data, and weak surrogate anomaly labels.

## RQ4 Compliance Classification Results

RQ4 predicts the next-period `compliance_flag_stateaware` value. The target is
created by shifting the current site-level state-aware compliance flag one
observation forward.

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| XGBoost | 0.477 | 0.712 | 0.572 | 0.874 |
| LightGBM | 0.471 | 0.709 | 0.566 | 0.874 |
| Random Forest | 0.401 | 0.813 | 0.537 | 0.877 |
| Logistic Regression | 0.238 | 0.665 | 0.351 | 0.698 |

Ensemble machine learning models substantially outperformed Logistic
Regression. Random Forest achieved the highest ROC-AUC and recall, while
XGBoost produced the strongest F1-score and best precision-recall balance.

## RQ5 Model Comparison Results

RQ5 compares forecasting models across all targets.

| Target | Best Overall Model | RMSE | MAE | R2 |
|---|---|---:|---:|---:|
| `dissolved_oxygen` | TCN | 1.078 | 0.645 | 0.860 |
| `ph` | TCN | 0.326 | 0.215 | 0.678 |
| `turbidity` | XGBoost | 2.023 | 1.350 | 0.907 |

The comparison showed that no single model was universally best:

- TCN and LSTM were strongest for dissolved oxygen and pH, where temporal
  continuity is high.
- XGBoost, LightGBM, Random Forest, and weighted ensembles were strongest for
  turbidity, where nonlinear event-driven variation matters more.
- Linear Regression consistently underperformed, confirming that nonlinear
  methods are needed for this dataset.
- Weighted ensembles were competitive and stable, especially for turbidity and
  pH.

## Preliminary Findings

The notebook supports several interim conclusions:

- Advanced ML and DL models generally outperform traditional statistical
  baselines for water-quality forecasting.
- Engineered temporal features are central to model performance.
- pH and dissolved oxygen show strong temporal structure, making them good
  candidates for sequence models.
- Turbidity is more nonlinear and event-driven, making boosted tree models more
  effective.
- Compliance-risk prediction has promising early-warning potential, especially
  with XGBoost, LightGBM, and Random Forest.
- Anomaly detection remains the weakest modeling layer and needs better labels,
  threshold calibration, and class-imbalance handling.

## Limitations and Risks

The notebook identifies these main limitations:

- USGS hydrologic features have limited station overlap with WQP records.
- Lag and rolling-window features create unavoidable structural missingness.
- Some raw water-quality values are physically implausible and require careful
  cleanup.
- Anomaly labels are surrogate labels rather than confirmed field-validated
  events.
- Compliance-risk labels are screening indicators, not formal regulatory
  violation labels.
- Optional TabNet experiments were not completed because the required package
  was unavailable.
- Additional SHAP analysis, hyperparameter tuning, statistical testing, and
  final validation remain future work.

## Recommended Next Steps

1. Finalize the preprocessing decisions for hydrologic imputation and outlier
   capping.
2. Add SHAP-based interpretation for RQ2 driver analysis.
3. Tune the strongest models for each target instead of using broad default
   settings.
4. Improve RQ3 anomaly labels using domain-informed thresholds or external
   event records.
5. Validate RQ4 compliance-risk logic against formal state-specific regulatory
   definitions where available.
6. Add paper-ready plots and summary tables for the final report.
7. Consider moving stable notebook logic into reusable scripts once the
   modeling workflow is finalized.
