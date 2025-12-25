# run_pipeline.py

import pandas as pd
import numpy as np

from config import HORIZON_DAYS, RANDOM_SEED
from features.build_features import build_features
from targets.build_target import build_weekly_target
from models.linear_model import fit_ridge, predict
from evaluation.walk_forward import walk_forward_splits

# --------------------------------------------------
# 1. Load paths
# --------------------------------------------------

BETAS_PATH   = "data/RollingBetas_smoothed_daily.csv"
RETURNS_PATH = "StockReturns_Daily2.xlsx"
SECTOR_PATH  = "data/sector_map.csv"

# --------------------------------------------------
# 2. Load + standardize BETAS (wide)
# --------------------------------------------------

betas = pd.read_csv(BETAS_PATH)

betas = betas.rename(columns={
    "Date": "date",
    "ticker": "stock",
    "Stock": "stock",

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

# --------------------------------------------------
# 3. Load + reshape RETURNS (wide → long)
# --------------------------------------------------

returns = pd.read_excel(RETURNS_PATH)
returns = returns.rename(columns={"Date": "date"})
returns["date"] = pd.to_datetime(returns["date"])

returns_long = (
    returns
    .melt(id_vars="date", var_name="stock", value_name="return")
    .dropna()
)

# --------------------------------------------------
# 4. Align returns to beta availability (start only)
# --------------------------------------------------

start_date = betas["date"].min()
returns_long = returns_long[returns_long["date"] >= start_date]

# --------------------------------------------------
# 5. Load sector map
# --------------------------------------------------

sector_map = pd.read_csv(SECTOR_PATH)
sector_map = sector_map.rename(columns={"Stock": "stock"})

# --------------------------------------------------
# 6. Build features and target
# --------------------------------------------------

X = build_features(betas)
y = build_weekly_target(returns_long, sector_map)

# --------------------------------------------------
# 7. Merge panel
# --------------------------------------------------

data = (
    X.merge(y, on=["date", "stock"], how="inner")
     .sort_values(["date", "stock"])
     .reset_index(drop=True)
)

# --------------------------------------------------
# 8. Enforce strict 55-stock universe (drop 1 edge date)
# --------------------------------------------------

counts = data.groupby("date")["stock"].nunique()
valid_dates = counts[counts == 55].index
data = data[data["date"].isin(valid_dates)]

assert data.groupby("date")["stock"].nunique().min() == 55
assert not data.duplicated(subset=["date", "stock"]).any()

# --------------------------------------------------
# 9. Feature columns
# --------------------------------------------------

feature_cols = [c for c in data.columns if c.startswith("beta_") or c.startswith("d_beta_")]

# --------------------------------------------------
# 10. Walk-forward training & prediction
# --------------------------------------------------

dates = sorted(data["date"].unique())
predictions = []

for train_dates, test_date in walk_forward_splits(dates, min_train_size=500):

    train = data[data["date"].isin(train_dates)]
    test  = data[data["date"] == test_date]

    # Drop rows with missing features in TRAIN
    train_clean = train.dropna(subset=feature_cols + ["y"])

    model = fit_ridge(
        train_clean[feature_cols],
        train_clean["y"],
        alpha=1.0
    )

    # Predict only where features are available
    test_clean = test.dropna(subset=feature_cols)

    test = test.loc[test_clean.index].copy()
    test["score"] = predict(model, test_clean[feature_cols])

    predictions.append(test[["date", "stock", "score"]])

# --------------------------------------------------
# 11. Collect predictions
# --------------------------------------------------

predictions = pd.concat(predictions).reset_index(drop=True)

print("Walk-forward completed.")
print("Prediction dates:", predictions["date"].nunique())
print(predictions.head())

# --------------------------------------------------
# 12. Information Coefficient evaluation
# --------------------------------------------------

from evaluation.ic import compute_daily_ic, summarize_ic

ic_df = compute_daily_ic(
    predictions=predictions,
    targets=y,
    method="spearman"
)

ic_stats = summarize_ic(ic_df)

print("\nInformation Coefficient (Spearman):")
for k, v in ic_stats.items():
    print(f"{k}: {v:.4f}")
