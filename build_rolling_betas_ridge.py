import numpy as np
import pandas as pd
from loguru import logger

# If sklearn isn't installed in your environment, tell me and I’ll give a no-sklearn PCA.
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ======================================================
# CONFIG
# ======================================================

TSFL_FACTORS_XLSX = "tsfl_2s_factors_weekly.xlsx"
TSFL_SHEET_NAME   = "TSFL_2S_Factors"

STOCK_RETURNS_XLSX = "StockReturns2Weekly.xlsx"
STOCK_SHEET_NAME   = "StockReturns"

ROLLING_WINDOW = 52
MIN_VALID_FRAC = 0.80

# Ridge lambda tuning
LAMBDA_GRID = [0.0, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1, 0.2, 0.5, 1.0, 5.0]
MAX_STOCKS_FOR_TUNING = 55  # set None for all stocks, but it'll be slow

# Output
CLIP_BETAS = 5.0
OUTPUT_PREFIX = "RollingBetas"

# Macro blocks (edit names to match your exact TSFL column names)
MACRO_BLOCKS = {
    "Macro_FX": [
        "TSFL USD FX",
        "TSFL Currency FX",
        "TSFL Foreign Exchange Carry",
    ],
    "Macro_RatesCredit": [
        "TSFL Interest Rates",
        "TSFL Fixed Income Carry",
        "TSFL US Inflation",
        "TSFL Credit",
    ],
}

# ======================================================
# Utilities
# ======================================================

def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = pd.to_datetime(df.index).normalize()
    return df.sort_index()

def load_and_align_option_a():
    """
    Option A: master calendar = stocks.index (S&P500 trading days).
    Reindex factors to stock dates (keep NaNs in factors).
    """
    logger.info("Loading TSFL factors from {}", TSFL_FACTORS_XLSX)
    tsfl = pd.read_excel(TSFL_FACTORS_XLSX, sheet_name=TSFL_SHEET_NAME, index_col=0, parse_dates=True)

    logger.info("Loading stock returns from {}", STOCK_RETURNS_XLSX)
    stocks = pd.read_excel(STOCK_RETURNS_XLSX, sheet_name=STOCK_SHEET_NAME, index_col=0, parse_dates=True)

    tsfl.columns = tsfl.columns.astype(str)
    stocks.columns = stocks.columns.astype(str)

    tsfl = _normalize_index(tsfl)
    stocks = _normalize_index(stocks)

    # Drop days where all stocks are NaN
    stocks = stocks.loc[~stocks.isna().all(axis=1)].copy()

    # Master calendar = stocks.index
    tsfl = tsfl.reindex(stocks.index)

    if not tsfl.index.equals(stocks.index):
        raise ValueError("Alignment failed: TSFL index != stocks index")

    logger.info("Aligned shapes: TSFL {}, Stocks {}", tsfl.shape, stocks.shape)
    return tsfl, stocks

def triage_factors(tsfl_raw: pd.DataFrame, top_n: int = 15):
    nan_frac = tsfl_raw.isna().mean().sort_values(ascending=False)
    stds = tsfl_raw.std(skipna=True).sort_values()

    print("\n=== TRIAGE: Top NaN factors ===")
    print(nan_frac.head(top_n))

    print("\n=== TRIAGE: Smallest-std factors ===")
    print(stds.head(top_n))

def pca_block(tsfl_raw: pd.DataFrame, cols: list[str], block_name: str) -> pd.Series | None:
    """
    Build 1st principal component from cols within the block.
    - Works on aligned raw factor returns
    - Standardizes within block (so PCA captures correlation structure)
    - Fills NaNs with 0 after standardization (equivalent to 'missing = average' in Z-space)
      If a column has too many NaNs, consider dropping it earlier.
    """
    cols_present = [c for c in cols if c in tsfl_raw.columns]
    if len(cols_present) == 0:
        logger.warning("Macro block {}: none of the columns exist in tsfl_raw", block_name)
        return None
    if len(cols_present) == 1:
        # If only one factor in the block, just return it as-is (renamed)
        c = cols_present[0]
        logger.info("Macro block {} has one column only: using it directly ({})", block_name, c)
        s = tsfl_raw[c].copy()
        s.name = f"{block_name}_SINGLE"
        return s

    X = tsfl_raw[cols_present].copy()

    # Standardize within block for PCA
    scaler = StandardScaler(with_mean=True, with_std=True)
    Z = scaler.fit_transform(X.fillna(X.mean()))  # quick robust fill for PCA fit

    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(Z).flatten()
    explained = float(pca.explained_variance_ratio_[0])

    logger.info("{} PC1 explains {:.1%} of variance (cols={})", block_name, explained, cols_present)

    pc = pd.Series(pc1, index=tsfl_raw.index, name=f"{block_name}_PC1")
    return pc

def apply_macro_pca(tsfl_raw: pd.DataFrame, macro_blocks: dict) -> pd.DataFrame:
    """
    Replace raw macro columns with PC1 per macro block.
    """
    tsfl_raw = tsfl_raw.copy()

    pcs = []
    used_macro_cols = []

    for block_name, cols in macro_blocks.items():
        pc = pca_block(tsfl_raw, cols, block_name)
        if pc is not None:
            pcs.append(pc)
        used_macro_cols.extend([c for c in cols if c in tsfl_raw.columns])

    # Drop the original macro columns that were used
    used_macro_cols = sorted(set(used_macro_cols))
    if used_macro_cols:
        logger.info("Dropping original macro columns replaced by PCA: {}", used_macro_cols)
        tsfl_raw = tsfl_raw.drop(columns=used_macro_cols)

    # Add PCs
    if pcs:
        tsfl_raw = pd.concat([tsfl_raw] + pcs, axis=1)

    logger.info("After macro PCA: {} factors", tsfl_raw.shape[1])
    return tsfl_raw

def standardize_factors(tsfl: pd.DataFrame):
    """
    Standardize final factor set for ridge.
    """
    means = tsfl.mean(skipna=True)
    stds  = tsfl.std(skipna=True)

    # Drop factors with zero std (constant)
    valid = stds > 0
    if not valid.all():
        dropped = stds.index[~valid].tolist()
        logger.warning("Dropping constant/zero-std factors: {}", dropped)
        tsfl = tsfl.loc[:, valid]
        means = means[valid]
        stds  = stds[valid]

    tsfl_std = (tsfl - means) / stds
    return tsfl_std, means, stds

def ridge_fit_beta_std(X_w: np.ndarray, y_w: np.ndarray, lam: float):
    """
    Ridge with intercept unpenalized. X_w must be standardized already.
    Returns intercept, beta_std.
    """
    X_design = np.column_stack([np.ones(len(X_w)), X_w])
    XTX = X_design.T @ X_design
    XTy = X_design.T @ y_w

    I = np.eye(XTX.shape[0])
    I[0, 0] = 0.0

    coef = np.linalg.solve(XTX + lam * I, XTy)
    return float(coef[0]), coef[1:]

def choose_ridge_lambda(tsfl_std: pd.DataFrame, stocks: pd.DataFrame, window: int, lambda_grid, max_stocks: int | None):
    """
    Walk-forward OOS tuning:
      fit on [t-window+1..t], predict r_{t+1}.
    """
    stock_list = list(stocks.columns) if max_stocks is None else list(stocks.columns)[:max_stocks]

    F = tsfl_std.to_numpy(dtype=float)
    R = stocks[stock_list].to_numpy(dtype=float)

    T, k = F.shape
    min_obs = max(int(MIN_VALID_FRAC * window), 10)

    t_start = window - 1
    t_end = T - 2

    logger.info("Tuning lambda on {} stocks, window={}, grid size={}", len(stock_list), window, len(lambda_grid))

    best_lam = None
    best_mse = np.inf

    for lam in lambda_grid:
        se_sum = 0.0
        n_sum = 0

        for j in range(R.shape[1]):
            y = R[:, j]

            for t in range(t_start, t_end + 1):
                y_win = y[t - window + 1 : t + 1]
                X_win = F[t - window + 1 : t + 1, :]
                X_next = F[t + 1, :]

                valid_train = (~np.isnan(y_win)) & (~np.isnan(X_win).any(axis=1))
                if valid_train.sum() < min_obs:
                    continue
                if np.isnan(y[t + 1]) or np.isnan(X_next).any():
                    continue

                y_w = y_win[valid_train]
                X_w = X_win[valid_train, :]

                intercept, beta_std = ridge_fit_beta_std(X_w, y_w, lam)
                y_pred = intercept + float(beta_std @ X_next)

                err = y[t + 1] - y_pred
                se_sum += err * err
                n_sum += 1

        mse = se_sum / n_sum if n_sum > 0 else np.inf
        logger.info("lambda={:<8g}  OOS_MSE={:<12g}  n={}", lam, mse, n_sum)

        if mse < best_mse:
            best_mse = mse
            best_lam = lam

    logger.success("Chosen lambda = {} (min OOS MSE = {})", best_lam, best_mse)
    return best_lam

def rolling_betas_for_stock_raw(stock_ret: pd.Series, tsfl_std: pd.DataFrame, factor_stds: pd.Series, window: int, lam: float):
    """
    Rolling ridge betas for one stock. Returns RAW betas.
    """
    y = stock_ret.to_numpy(dtype=float)
    X = tsfl_std.to_numpy(dtype=float)

    T, k = X.shape
    betas_std = np.full((T, k), np.nan)

    min_obs = max(int(MIN_VALID_FRAC * window), 10)

    for t in range(window - 1, T):
        y_win = y[t - window + 1 : t + 1]
        X_win = X[t - window + 1 : t + 1, :]

        valid = (~np.isnan(y_win)) & (~np.isnan(X_win).any(axis=1))
        if valid.sum() < min_obs:
            continue

        y_w = y_win[valid]
        X_w = X_win[valid, :]

        _, beta_std = ridge_fit_beta_std(X_w, y_w, lam)
        betas_std[t, :] = beta_std

    beta_std_df = pd.DataFrame(betas_std, index=tsfl_std.index, columns=tsfl_std.columns)

    # Unscale to raw units (critical)
    beta_raw_df = beta_std_df.divide(factor_stds, axis=1)

    if CLIP_BETAS is not None:
        beta_raw_df = beta_raw_df.clip(-CLIP_BETAS, CLIP_BETAS)

    return beta_raw_df

def compute_all_betas_panel(tsfl_std: pd.DataFrame, stocks: pd.DataFrame, factor_stds: pd.Series, window: int, lam: float):
    betas_by_stock = {}
    for stk in stocks.columns:
        logger.info("Rolling betas for {}", stk)
        betas_by_stock[stk] = rolling_betas_for_stock_raw(stocks[stk], tsfl_std, factor_stds, window, lam)
    panel = pd.concat(betas_by_stock, axis=1)
    logger.info("Beta panel shape: {}", panel.shape)
    return panel

# ======================================================
# MAIN
# ======================================================

if __name__ == "__main__":
    # 1) Load + align (Option A)
    tsfl_raw, stocks = load_and_align_option_a()

    # 2) Triage (optional but helpful)
    triage_factors(tsfl_raw, top_n=15)

    # 3) Macro PCA compression (manual grouping + PCA inside group)
    tsfl_raw = apply_macro_pca(tsfl_raw, MACRO_BLOCKS)

    # 4) Standardize final factor set
    tsfl_std, factor_means, factor_stds = standardize_factors(tsfl_raw)

    # Ensure alignment
    if not tsfl_std.index.equals(stocks.index):
        raise ValueError("Post-processing alignment failed: factors index != stocks index")

    # 5) Choose ridge lambda automatically
    chosen_lam = choose_ridge_lambda(
        tsfl_std=tsfl_std,
        stocks=stocks,
        window=ROLLING_WINDOW,
        lambda_grid=LAMBDA_GRID,
        max_stocks=MAX_STOCKS_FOR_TUNING
    )

    if chosen_lam is None:
        raise ValueError("Failed to choose a valid lambda from the grid")

    # 6) Compute rolling betas (RAW units)
    betas_panel = compute_all_betas_panel(
        tsfl_std=tsfl_std,
        stocks=stocks,
        factor_stds=factor_stds,
        window=ROLLING_WINDOW,
        lam=chosen_lam
    )

    # 7) Save
    out_xlsx = f"{OUTPUT_PREFIX}_{ROLLING_WINDOW}d_macroPCA_ridge_lambda3_{chosen_lam:g}.xlsx"
    sheet = f"Betas_{ROLLING_WINDOW}d"
    logger.info("Saving to {}", out_xlsx)
    betas_panel.to_excel(out_xlsx, sheet_name=sheet)
    logger.success("Done.")
