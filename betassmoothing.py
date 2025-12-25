import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# ============================================================
# INPUT FILES
# ============================================================

TSFL_FACTORS_XLSX = "tsfl_2s_factors_daily.xlsx"
TSFL_SHEET_NAME  = "TSFL_2S_Factors"

STOCK_RETURNS_XLSX = "StockReturns_Daily2.xlsx"
STOCK_SHEET_NAME   = "StockReturns"

OUT_BETAS_CSV   = "data/RollingBetas_smoothed_daily.csv"
OUT_ML_CSV      = "data/ML_dataset_betas_weekly_target.csv"
OUT_PCA_VAR_CSV = "PCA_macro_explained_variance.csv"
OUT_BETAS_XLSX  = "RollingBetas_presentation.xlsx"
OUT_LAMBDA_CSV  = "lambda_selection_diagnostics.csv"
OUT_BETA_NORM_CSV = "beta_magnitude_diagnostics.csv"

# ============================================================
# PARAMETERS
# ============================================================

WINDOW_DAYS   = 60
GAMMA_SMOOTH  = 1.0
MIN_FRAC_OK   = 0.80
CLIP_BETAS    = 5.0
TARGET_HORIZON_DAYS = 5

LAMBDA_GRID = [0.003, 0.005, 0.01, 0.1, 0.5]

# ============================================================
# MACRO PCA BLOCKS
# ============================================================

MACRO_BLOCKS = {
    "Macro_FX": [
        "TSFL USD FX",
        "TSFL Currency FX",
        "TSFL Foreign Exchange Carry",
    ],
}

N_PCA_COMPONENTS = 1   # PC1 only

# ============================================================
# HELPERS
# ============================================================

def ridge_solution(X, y, beta_prev, lam, gamma):
    k = X.shape[1]
    A = X.T @ X + (lam + gamma) * np.eye(k)
    b = X.T @ y + gamma * beta_prev
    return np.linalg.solve(A, b)

def select_lambda(X, y, beta_prev):
    best_lam = None
    best_err = np.inf
    for lam in LAMBDA_GRID:
        beta = ridge_solution(X, y, beta_prev, lam, GAMMA_SMOOTH)
        err = np.mean((y - X @ beta) ** 2)
        if err < best_err:
            best_err = err
            best_lam = lam
    return best_lam

def safe_standardize(X, eps=1e-12):
    """Columnwise standardization with protection against 0 std."""
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd = np.where(sd < eps, 1.0, sd)
    return (X - mu) / sd

# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # LOAD TSFL FACTORS
    # --------------------------------------------------------

    factors = pd.read_excel(TSFL_FACTORS_XLSX, sheet_name=TSFL_SHEET_NAME)
    factors["Date"] = pd.to_datetime(factors["Date"])
    factors = factors.set_index("Date").sort_index()
    factors = factors.ffill().dropna()

    factor_names = list(factors.columns)

    # Validate macro block factors exist
    macro_union = []
    for block, facs in MACRO_BLOCKS.items():
        for f in facs:
            if f not in factor_names:
                raise ValueError(f"Macro block '{block}' factor not found in TSFL factors: '{f}'")
        macro_union.extend(facs)

    macro_union = list(dict.fromkeys(macro_union))  # unique preserve order

    # Build transformed factor list:
    #   - keep everything not in any macro block
    #   - add one PC name per block
    base_factors = [f for f in factor_names if f not in macro_union]
    pc_factors = [f"{block}_PC1" for block in MACRO_BLOCKS.keys()]
    factor_names_used = base_factors + pc_factors

    # Indices for fast slicing from the original X (before PCA replacement)
    base_idx = [factor_names.index(f) for f in base_factors]
    block_idx = {block: [factor_names.index(f) for f in facs] for block, facs in MACRO_BLOCKS.items()}

    # --------------------------------------------------------
    # LOAD STOCK RETURNS (WIDE → LONG)
    # --------------------------------------------------------

    stocks_wide = pd.read_excel(STOCK_RETURNS_XLSX, sheet_name=STOCK_SHEET_NAME)
    stocks_wide["Date"] = pd.to_datetime(stocks_wide["Date"])

    stocks = (
        stocks_wide
        .melt(id_vars=["Date"], var_name="ticker", value_name="ret")
        .dropna()
        .sort_values(["ticker", "Date"])
    )

    # --------------------------------------------------------
    # CONTAINERS
    # --------------------------------------------------------

    results = []
    pca_variance_records = []
    lambda_records = []
    beta_norm_records = []

    # --------------------------------------------------------
    # ROLLING ESTIMATION
    # --------------------------------------------------------

    for ticker, sdf in stocks.groupby("ticker"):

        df = sdf.set_index("Date").join(factors, how="inner")
        if len(df) < WINDOW_DAYS:
            continue

        y_all = df["ret"].values
        X_all = df[factor_names].values
        dates = df.index

        # IMPORTANT CHANGE:
        # beta_prev lives in the *transformed* space (base_factors + PCs),
        # because that's what you actually estimate after PCA replacement.
        beta_prev_used = np.zeros(len(factor_names_used))

        for t in range(WINDOW_DAYS, len(df)):

            y = y_all[t - WINDOW_DAYS:t]
            X = X_all[t - WINDOW_DAYS:t]

            if np.mean(np.isfinite(y)) < MIN_FRAC_OK:
                continue

            # -------------------------------
            # STANDARDIZE FACTORS (KEY STEP)
            # -------------------------------
            X = safe_standardize(X)

            # -------------------------------
            # PCA (MACRO BLOCKS) + BUILD X_used
            # Instead of delete/insert (which breaks with >1 block),
            # we build X_used from scratch: [base] + [PCs...]
            # -------------------------------

            X_base = X[:, base_idx]
            X_used = X_base.copy()

            for block, idx in block_idx.items():
                X_block = X[:, idx]  # already standardized columns

                pca = PCA(n_components=N_PCA_COMPONENTS)
                Z = pca.fit_transform(X_block)  # shape (window, 1)

                pca_variance_records.append({
                    "Date": dates[t],
                    "ticker": ticker,
                    "block": block,
                    "component": 1,
                    "explained_variance_ratio": float(pca.explained_variance_ratio_[0]),
                })

                # Append PC1 as the single replacement factor for the block
                X_used = np.hstack([X_used, Z])

            # -------------------------------
            # LAMBDA SELECTION
            # -------------------------------

            lam_star = select_lambda(X_used, y, beta_prev_used)

            beta_used = ridge_solution(
                X_used, y, beta_prev_used, lam_star, GAMMA_SMOOTH
            )

            # -------------------------------
            # DIAGNOSTICS
            # -------------------------------

            lambda_records.append({
                "Date": dates[t],
                "ticker": ticker,
                "lambda": lam_star,
            })

            beta_norm_records.append({
                "Date": dates[t],
                "ticker": ticker,
                "l2_norm": float(np.linalg.norm(beta_used)),
                "max_abs": float(np.max(np.abs(beta_used))),
            })

            # -------------------------------
            # CLIP + UPDATE PREVIOUS BETAS (TRANSFORMED SPACE)
            # -------------------------------

            if CLIP_BETAS is not None:
                beta_used = np.clip(beta_used, -CLIP_BETAS, CLIP_BETAS)

            beta_prev_used = beta_used

            # -------------------------------
            # SAVE BETAS IN TRANSFORMED SPACE
            # This is the key change:
            # We output base factors + Macro_FX_PC1 (etc),
            # and we DO NOT output the three original FX factors.
            # -------------------------------

            for f, b in zip(factor_names_used, beta_used):
                results.append({
                    "Date": dates[t],
                    "ticker": ticker,
                    "factor": f,
                    "beta": float(b),
                })

    # --------------------------------------------------------
    # SAVE OUTPUTS
    # --------------------------------------------------------

    betas = pd.DataFrame(results)
    betas.to_csv(OUT_BETAS_CSV, index=False)
# --- WIDE version (one row per Date x ticker) ---
    betas_wide = (
        betas
        .pivot(index=["Date", "ticker"], columns="factor", values="beta")
        .reset_index()
        .sort_values(["ticker", "Date"])
    )

    betas_wide.to_csv(OUT_BETAS_CSV, index=False)

    pd.DataFrame(pca_variance_records).to_csv(OUT_PCA_VAR_CSV, index=False)
    pd.DataFrame(lambda_records).to_csv(OUT_LAMBDA_CSV, index=False)

    # ... inside main(), in SAVE OUTPUTS, after writing OUT_LAMBDA_CSV ...

    lambda_df = pd.DataFrame(lambda_records)
    lambda_df["Date"] = pd.to_datetime(lambda_df["Date"])
    lambda_df = lambda_df.sort_values(["Date", "ticker"])

    # Shares per (Date, lambda)
    shares = (
        lambda_df.groupby(["Date", "lambda"])
        .size()
        .groupby(level=0)
        .apply(lambda s: s / s.sum())
        .unstack(fill_value=0)
        .sort_index()
    )

    # For each date, look at the maximum share among lambdas (dominance measure)
    max_share = shares.max(axis=1)

    # Bin into your requested ranges
    bands = pd.DataFrame(index=shares.index)
    bands[">=50%"]     = (max_share >= 0.50).astype(float)
    bands["50-30%"]    = ((max_share < 0.50) & (max_share >= 0.30)).astype(float)
    bands["30-15%"]    = ((max_share < 0.30) & (max_share >= 0.15)).astype(float)
    bands["<15%"]      = (max_share < 0.15).astype(float)

    # Plot (values are 0/1). Optional: smooth with rolling mean for readability
    smooth = 20  # days; change or set to 1 for no smoothing
    bands_smooth = bands.rolling(smooth, min_periods=1).mean()
    bands_smooth = bands_smooth.copy()
    bands_smooth.index = pd.to_datetime(bands_smooth.index.get_level_values(0))
    plt.figure()
    for col in bands_smooth.columns:
        plt.plot(bands_smooth.index, bands_smooth[col], label=col)

    plt.xlabel("Date")
    plt.ylabel(f"Fraction of days in band (rolling {smooth})")
    plt.title("How concentrated is λ selection over time?")
    plt.legend()
    plt.tight_layout()
    plt.show()

    pd.DataFrame(beta_norm_records).to_csv(OUT_BETA_NORM_CSV, index=False)

    # --------------------------------------------------------
    # PRESENTATION EXCEL (SMALL SUBSET)
    # --------------------------------------------------------

    subset = sorted(betas["ticker"].unique())[:8]

    pretty = (
        betas
        .query("ticker in @subset")
        .assign(col=lambda x: x["ticker"].astype(str) + "_" + x["factor"].astype(str))
        .pivot(index="Date", columns="col", values="beta")
        .sort_index()
    )

    pretty.to_excel(OUT_BETAS_XLSX)

    # --------------------------------------------------------
    # ML DATASET
    # --------------------------------------------------------

    stocks["future_ret"] = stocks.groupby("ticker")["ret"].shift(-TARGET_HORIZON_DAYS)

    ml = (
        betas
        .pivot(index=["Date", "ticker"], columns="factor", values="beta")
        .reset_index()
        .merge(
            stocks[["Date", "ticker", "future_ret"]],
            on=["Date", "ticker"],
            how="left",
        )
        .dropna()
    )

    ml.to_csv(OUT_ML_CSV, index=False)

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
