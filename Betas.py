import numpy as np
import pandas as pd
from loguru import logger

# ======================================================
# CONFIG – adjust these filenames/sheet names if needed
# ======================================================

# TSFL factor returns (daily) – pick the file you already have
TSFL_FACTORS_XLSX = "tsfl_2s_factors.xlsx"          
TSFL_SHEET_NAME   = "Sheet1"                          

# Stock daily returns (from previous script)
STOCK_RETURNS_XLSX = "StockReturns.xlsx"
STOCK_SHEET_NAME   = "StockReturns"

ROLLING_WINDOW = 60   # 60 trading days


# ======================================================
# 1. Load TSFL factors and stock returns, align dates
# ======================================================

def load_and_align_data():
    logger.info("Loading TSFL factor returns from {}", TSFL_FACTORS_XLSX)
    tsfl = pd.read_excel(
        TSFL_FACTORS_XLSX,
        sheet_name=TSFL_SHEET_NAME,
        index_col=0,
        parse_dates=True,
    )

    logger.info("Loading stock returns from {}", STOCK_RETURNS_XLSX)
    stocks = pd.read_excel(
        STOCK_RETURNS_XLSX,
        sheet_name=STOCK_SHEET_NAME,
        index_col=0,
        parse_dates=True,
    )

    # Make sure columns are clean strings
    tsfl.columns = tsfl.columns.astype(str)
    stocks.columns = stocks.columns.astype(str)

    # Keep only overlapping dates
    common_index = tsfl.index.intersection(stocks.index)
    tsfl = tsfl.loc[common_index].sort_index()
    stocks = stocks.loc[common_index].sort_index()

    logger.info("Aligned TSFL shape:  {}", tsfl.shape)
    logger.info("Aligned stocks shape: {}", stocks.shape)

    # Drop rows where all TSFL factors are NaN or all stocks are NaN
    mask_tsfl_ok = ~tsfl.isna().all(axis=1)
    mask_stk_ok  = ~stocks.isna().all(axis=1)
    mask = mask_tsfl_ok & mask_stk_ok

    tsfl = tsfl.loc[mask]
    stocks = stocks.loc[mask]

    logger.info("After dropping all-NaN rows:")
    logger.info("  TSFL shape:  {}", tsfl.shape)
    logger.info("  Stocks shape: {}", stocks.shape)

    return tsfl, stocks


# ======================================================
# 2. Rolling 60-day OLS beta estimation for one stock
# ======================================================

def rolling_betas_for_stock(stock_ret: pd.Series,
                            factor_ret: pd.DataFrame,
                            window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """
    Compute rolling OLS betas of one stock vs all TSFL factors.

    stock_ret: Series of daily returns for one stock (aligned with factor_ret index)
    factor_ret: DataFrame of daily TSFL factor returns (columns = factors)
    window: number of days in each OLS window (60)
    """
    y = stock_ret.values
    X_full = factor_ret.values
    T, k = X_full.shape

    betas = np.full((T, k), np.nan)

    for t in range(window - 1, T):
        # window [t-window+1, ..., t]
        y_win = y[t - window + 1 : t + 1]
        X_win = X_full[t - window + 1 : t + 1]

        # drop rows with any NaN in y or X
        valid = ~np.isnan(y_win) & ~np.isnan(X_win).any(axis=1)
        if valid.sum() < max(int(0.8 * window), 10):
            # require at least 80% of the window to be valid
            continue

        y_w = y_win[valid]
        X_w = X_win[valid]

        # design matrix with intercept
        X_design = np.column_stack([np.ones(len(X_w)), X_w])

        beta_hat, *_ = np.linalg.lstsq(X_design, y_w, rcond=None)
        # beta_hat[0] is intercept, beta_hat[1:] are factor betas
        betas[t, :] = beta_hat[1:]

    beta_df = pd.DataFrame(
        betas,
        index=factor_ret.index,
        columns=factor_ret.columns,
    )
    return beta_df


# ======================================================
# 3. Loop over all stocks and build a big beta panel
# ======================================================

def compute_all_betas(tsfl: pd.DataFrame,
                      stocks: pd.DataFrame,
                      window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """
    Returns a DataFrame with MultiIndex columns:
    (stock, factor), index = dates, values = rolling beta(stock, factor, date)
    """
    betas_by_stock = {}

    for stk in stocks.columns:
        logger.info("Computing rolling betas for stock {}", stk)
        beta_df = rolling_betas_for_stock(stocks[stk], tsfl, window=window)
        betas_by_stock[stk] = beta_df

    # columns: MultiIndex (stock, factor)
    betas_panel = pd.concat(betas_by_stock, axis=1)  # keys=stock names
    logger.info("Full beta panel shape: {}", betas_panel.shape)
    return betas_panel


# ======================================================
# 4. Save to Excel
# ======================================================

def save_betas_to_excel(betas_panel: pd.DataFrame,
                        filename: str = "RollingBetas_60d.xlsx"):
    logger.info("Saving rolling betas to {}", filename)
    betas_panel.to_excel(filename, sheet_name="Betas_60d")
    logger.success("Saved successfully as {}", filename)


# ======================================================
# MAIN
# ======================================================

if __name__ == "__main__":
    tsfl_factors, stock_returns = load_and_align_data()
    betas_panel = compute_all_betas(tsfl_factors, stock_returns, window=ROLLING_WINDOW)
    save_betas_to_excel(betas_panel)
