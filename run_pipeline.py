# run_pipeline.py

import pandas as pd
import numpy as np

from config import HORIZON_DAYS, RANDOM_SEED
from features.build_features import build_features
from targets.build_target import build_weekly_target
from models.linear_model import fit_ridge, predict
from evaluation.walk_forward import walk_forward_splits

# --------------------------------------------------
# 1. Load data
# --------------------------------------------------

BETAS_PATH   = "data/RollingBetas_smoothed_daily.csv"
RETURNS_PATH = "StockReturns_Daily2.xlsx"
SECTOR_PATH  = "data/sector_map.csv"

betas = pd.read_csv(BETAS_PATH)
returns = pd.read_excel("StockReturns_Daily2.xlsx", sheet_name="StockReturns")
sector_map = pd.read_csv(SECTOR_PATH)

# --------------------------------------------------
# 2. Clean & standardize columns
# --------------------------------------------------

betas = betas.rename(columns={
    "Date": "date",
    "ticker": "stock",
    "Macro_FX_PC1": "beta_fx_pc1",
    "TSFL Commodities": "beta_commodities",
    "TSFL Credit": "beta_credit",
    "TSFL Crowded": "beta_crowded",
    "TSFL Emerging Markets": "beta_em",
    "TSFL Equity US": "beta_equity_us",
    "TSFL Fixed Income Carry": "beta_fi_carry",
    "TSFL Interest Rates": "beta_interest_rates",
    "TSFL Low Risk": "beta_low_risk",
    "TSFL Momentum": "beta_momentum",
    "TSFL Quality": "beta_quality",
    "TSFL Value": "beta_value",
    "TSFL Real Estate": "beta_real_estate",
    "TSFL Small Cap": "beta_size",
    "TSFL US Inflation": "beta_us_inflation",

})

betas["date"] = pd.to_datetime(betas["date"])
returns["date"] = pd.to_datetime(returns["date"])

# --------------------------------------------------
# 3. Build features (X)
# --------------------------------------------------

X = build_features(betas)

# --------------------------------------------------
# 4. Build target (y)
# --------------------------------------------------

y = build_weekly_target(returns, sector_map)

# --------------------------------------------------
# 5. Merge panel
# --------------------------------------------------

data = (
    X.merge(y, on=["date", "stock"], how="inner")
      .sort_values(["date", "stock"])
      .reset_index(drop=True)
)

# Drop rows with missing values (from rolling / diff)
data = data.dropna()

# --------------------------------------------------
# 6. Sanity checks (MANDATORY)
# --------------------------------------------------

# same number of stocks every day
assert data.groupby("date")["stock"].nunique().min() == 55

# no duplicated rows
assert not data.duplicated(subset=["date", "stock"]).any()

# --------------------------------------------------
# 7. Define feature columns
# --------------------------------------------------

feature_cols = [
    c for c in data.columns
    if c.startswith("beta_") or c.startswith("d_beta_")
]

# --------------------------------------------------
# 8. Walk-forward training & prediction
# --------------------------------------------------

dates = sorted(data["date"].unique())
predictions = []

for train_dates, test_date in walk_forward_splits(
    dates,
    min_train_size=500,   # ~2 years
    step=1
):
    train = data[data["date"].isin(train_dates)]
    test  = data[data["date"] == test_date]

    X_train = train[feature_cols]
    y_train = train["y"]

    X_test = test[feature_cols]

    # ---- Fit baseline model
    model = fit_ridge(X_train, y_train, alpha=1.0)

    # ---- Predict cross-sectional scores
    test = test.copy()
    test["score"] = predict(model, X_test)

    predictions.append(
        test[["date", "stock", "score"]]
    )

# --------------------------------------------------
# 9. Collect predictions
# --------------------------------------------------

predictions = pd.concat(predictions).reset_index(drop=True)

print("Walk-forward completed.")
print("Prediction dates:", predictions["date"].nunique())
print(predictions.head())
