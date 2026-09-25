# Climate-Linked Mortality Prediction — Zindi Challenge

Predicting whether a recorded death is climate-sensitive, using health, demographic, geographic, and climate-related data (temperature, rainfall, NDVI, elevation). Built end-to-end on Databricks Free Edition using a medallion architecture (bronze → silver → gold) with MLflow experiment tracking.

**Result:** 0.8000 weighted score (60% F1 + 40% ROC-AUC) on the official Zindi leaderboard, rank 492.
Challenge: [Zindi — Climate-Linked Mortality Prediction](https://zindi.africa)

## Pipeline

**Bronze layer** — raw ingestion of Train/Test/climate_features CSVs into Delta tables, no transformation.

**Silver layer** — joined climate features to death records on ID, parsed dates, checked for nulls (none found) and class balance (65% climate-sensitive / 35% not).

**Gold layer** — encoded categorical features (zone, gender), dropped non-numeric columns not directly usable (deathdate, location), registered as a Databricks Feature Table (`gold_features`) via the Feature Engineering client.

**Modeling** — compared four approaches, each tracked as a separate MLflow run:
| Model | F1 | ROC-AUC | Weighted score |
|---|---|---|---|
| Logistic Regression (unscaled) | 0.8060 | 0.7768 | 0.7943 |
| Logistic Regression (scaled) | 0.7789 | 0.8167 | 0.7940 |
| XGBoost (manual params) | 0.7873 | 0.8089 | 0.7959 |
| XGBoost (Optuna, single-split tuning) | 0.7852 | 0.8154 | 0.7973 (local) → 0.788 (test) |

**A note on methodology:** the single-split Optuna search initially looked like the best model locally (0.8052 weighted score), but scored *worse* on the actual Zindi test set (0.788) — a clear sign of overfitting to one validation split. Re-running the search with 5-fold stratified cross-validation gave a more honest estimate (0.7986 mean CV score), which lined up closely with the real test performance. This mismatch — and catching it — was one of the more useful parts of the project.

**Final submitted model:** manually-tuned XGBoost with class weighting for imbalance, achieving **0.8000** on the official leaderboard.

## Tech stack
Databricks (PySpark, Delta Lake, Feature Engineering client), MLflow, scikit-learn, XGBoost, Optuna, pandas

## What I'd improve with more time
- Feature interactions (e.g., temperature/rainfall × age)
- LightGBM comparison
- Geospatial features beyond raw lat/long (e.g., distance to nearest health facility, if available)
- A larger, CV-based hyperparameter search from the start rather than as a correction
