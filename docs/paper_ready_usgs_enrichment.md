## Optional USGS Enrichment Layer (Paper-Ready)

To enhance the explanatory power of the WQP-centered dataset without compromising sample size, an optional USGS enrichment layer is applied after construction of the core dataset. This enrichment step adds hydrologic variables such as discharge, water temperature, and gage height for sites that can be linked to USGS monitoring stations.

The enrichment is implemented as a secondary step rather than a hard requirement during base dataset construction. This design avoids the row-loss problem that can occur when exact site-date overlap is required across heterogeneous environmental data systems. Instead, the full WQP-centered dataset is preserved, and USGS variables are merged where available.

Two matching strategies are supported: (1) exact site-date matching and (2) nearest-prior-date carry-forward within a defined tolerance window. The second approach improves coverage while remaining defensible for environmental monitoring contexts where hydrologic conditions are often persistent across short time windows. As a result, the enriched dataset retains the full WQP-centered sample size while providing additional predictors for driver analysis, forecasting, and compliance-risk modeling.
