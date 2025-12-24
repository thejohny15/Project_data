import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# ============================================================
# INPUT FILES
# ============================================================

TSFL_FACTORS_XLSX = "tsfl_2s_factors_daily.xlsx"
TSFL_SHEET_NAME  = "TSFL_2S_Factors"

STOCK_RETURNS_XLSX = "StockReturns_Daily2.xlsx"
STOCK_SHEET_NAME   = "StockReturns"

OUT_BETAS_CSV   = "RollingBetas_smoothed_daily.csv"
OUT_ML_CSV      = "ML_dataset_betas_weekly_target.csv"
OUT_PCA_VAR_CSV = "PCA_macro_explained_variance.csv"
OUT_BETAS_XLSX  = "RollingBetas_presentation.xlsx"
OUT_LAMBDA_CSV  = "lambda_selection_diagnostics.csv"
OUT_BETA_NORM_CSV = "beta_magnitude_diagnostics.csv"

# ============================================================
# PARAMETERS
# ============================================================

WINDOW_DAYS   = 120
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

    macro_block_indices = {
        block: [factor_names.index(f) for f in facs]
        for block, facs in MACRO_BLOCKS.items()
    }

    # --------------------------------------------------------
    # LOAD STOCK RETURNS (WIDE → LONG)
    # --------------------------------------------------------

    stocks_wide = pd.read_excel(STOCK_RETURNS_XLSX, sheet_name=STOCK_SHEET_NAME)
    stocks_wide["Date"] = pd.to_datetime(stocks_wide["Date"])

    stocks = (
        stocks_wide
        .melt(id_vars=["Date"], var_name="permno", value_name="ret")
        .dropna()
        .sort_values(["permno", "Date"])
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

    for permno, sdf in stocks.groupby("permno"):

        df = sdf.set_index("Date").join(factors, how="inner")
        if len(df) < WINDOW_DAYS:
            continue

        y_all = df["ret"].values
        X_all = df[factor_names].values
        dates = df.index

        beta_prev = np.zeros(len(factor_names))

        for t in range(WINDOW_DAYS, len(df)):

            y = y_all[t - WINDOW_DAYS:t]
            X = X_all[t - WINDOW_DAYS:t]

            if np.mean(np.isfinite(y)) < MIN_FRAC_OK:
                continue

            # -------------------------------
            # STANDARDIZE FACTORS (KEY STEP)
            # -------------------------------

            X = (X - X.mean(axis=0)) / X.std(axis=0)

            # -------------------------------
            # PCA (MACRO BLOCKS)
            # -------------------------------

            X_used = X.copy()
            beta_prev_used = beta_prev.copy()
            pca_maps = {}

            for block, idx in macro_block_indices.items():

                X_block = X[:, idx]

                pca = PCA(n_components=N_PCA_COMPONENTS)
                Z = pca.fit_transform(X_block)

                pca_variance_records.append({
                    "Date": dates[t],
                    "permno": permno,
                    "block": block,
                    "component": 1,
                    "explained_variance_ratio": pca.explained_variance_ratio_[0],
                })

                X_used = np.delete(X_used, idx, axis=1)
                X_used = np.hstack([X_used, Z])

                beta_prev_used = np.delete(beta_prev_used, idx)
                beta_prev_used = np.concatenate([beta_prev_used, np.zeros(1)])

                pca_maps[block] = (idx, pca.components_)

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
                "permno": permno,
                "lambda": lam_star,
            })

            beta_norm_records.append({
                "Date": dates[t],
                "permno": permno,
                "l2_norm": np.linalg.norm(beta_used),
                "max_abs": np.max(np.abs(beta_used)),
            })

            # -------------------------------
            # RECONSTRUCT FULL BETAS
            # -------------------------------

            beta_full = beta_used.copy()

            for block, (idx, comps) in pca_maps.items():
                beta_pc = beta_full[-1]
                beta_block = beta_pc * comps[0]
                beta_full = np.insert(beta_full[:-1], idx, beta_block)

            if CLIP_BETAS is not None:
                beta_full = np.clip(beta_full, -CLIP_BETAS, CLIP_BETAS)

            beta_prev = beta_full

            for f, b in zip(factor_names, beta_full):
                results.append({
                    "Date": dates[t],
                    "permno": permno,
                    "factor": f,
                    "beta": b,
                })

    # --------------------------------------------------------
    # SAVE OUTPUTS
    # --------------------------------------------------------

    betas = pd.DataFrame(results)
    betas.to_csv(OUT_BETAS_CSV, index=False)

    pd.DataFrame(pca_variance_records).to_csv(OUT_PCA_VAR_CSV, index=False)
    pd.DataFrame(lambda_records).to_csv(OUT_LAMBDA_CSV, index=False)
    pd.DataFrame(beta_norm_records).to_csv(OUT_BETA_NORM_CSV, index=False)

    # --------------------------------------------------------
    # PRESENTATION EXCEL (SMALL SUBSET)
    # --------------------------------------------------------

    subset = sorted(betas["permno"].unique())[:8]

    pretty = (
        betas
        .query("permno in @subset")
        .assign(col=lambda x: x["permno"].astype(str) + "_" + x["factor"].astype(str))
        .pivot(index="Date", columns="col", values="beta")
        .sort_index()
    )

    pretty.to_excel(OUT_BETAS_XLSX)

    # --------------------------------------------------------
    # ML DATASET
    # --------------------------------------------------------

    stocks["future_ret"] = stocks.groupby("permno")["ret"].shift(-TARGET_HORIZON_DAYS)

    ml = (
        betas
        .pivot(index=["Date", "permno"], columns="factor", values="beta")
        .reset_index()
        .merge(
            stocks[["Date", "permno", "future_ret"]],
            on=["Date", "permno"],
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
