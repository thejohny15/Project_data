import yfinance as yf
import pandas as pd
import numpy as np
from loguru import logger

# ========================================================
# 1. Universe: 55 stocks, 5 per major GICS sector
# ========================================================
TICKERS = [
    # Technology
    "AAPL", "MSFT", "NVDA", "AVGO", "ADBE",
    
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS",
    
    # Industrials
    "BA", "CAT", "GE", "UNP", "HON",
    
    # Healthcare
    "JNJ", "UNH", "PFE", "MRK", "ABT",
    
    # Energy
    "XOM", "CVX", "COP", "OXY", "SLB",
    
    # Consumer Discretionary
    "AMZN", "TSLA", "HD", "MCD", "NKE",
    
    # Consumer Staples
    "KO", "WMT", "COST", "PG", "PEP",
    
    # Utilities
    "DUK", "NEE", "D", "SO", "AEP",
    
    # Materials
    "LIN", "FCX", "NEM", "DD", "VMC",
    
    # Communication Services
    "GOOGL", "META", "NFLX", "CMCSA", "VZ",
    
    # Real Estate
    "PLD", "AMT", "SPG", "EQIX", "PSA",
]


# ========================================================
# 2. Download prices safely using the new yfinance behavior
# ========================================================
def download_prices(tickers, start="2014-01-01"):
    logger.info(f"Starting download for {len(tickers)} tickers…")

    all_data = {}
    failed = []

    for t in tickers:
        logger.info(f"Downloading {t}…")

        try:
            # Explicitly adjust so "Close" is adjusted price
            df = yf.download(t, start=start, progress=False, auto_adjust=True)

            if df is None or df.empty:
                logger.warning(f"No data for {t}, marking as failed.")
                failed.append(t)
                continue

            if "Close" not in df.columns:
                logger.error(f"{t}: 'Close' column missing — skipping.")
                failed.append(t)
                continue

            close = df["Close"]

            # yfinance sometimes returns a DataFrame with MultiIndex columns
            # close can be a Series or a 1-col DataFrame; handle both:
            if isinstance(close, pd.DataFrame):
                # take the first column as a Series
                close = close.squeeze()

            # ensure it's a proper Series with name = ticker
            if isinstance(close, pd.Series):
                close = close.astype(float)
                close.name = t
            else:
                logger.error(f"{t}: Unexpected data type after squeeze — skipping.")
                failed.append(t)
                continue

            all_data[t] = close

        except Exception as e:
            logger.exception(f"Error downloading {t}: {e}")
            failed.append(t)

    return all_data, failed


# ========================================================
# 3. Convert prices → daily simple returns
# ========================================================
def build_return_panel(all_data: dict, freq: str = "D"):
    logger.info("Aligning price series…")
    price_df = pd.DataFrame(all_data).sort_index()
    logger.info(f"Raw price panel shape: {price_df.shape}")

    # Daily simple returns from prices
    daily_returns = price_df.pct_change()

    if freq.upper() in ["W", "W-FRI", "WEEKLY"]:
        # Weekly returns = compounded daily returns within each week (Fri-anchored)
        returns = (1.0 + daily_returns).resample("W-FRI").prod() - 1.0
    else:
        # Daily returns
        returns = daily_returns

    # Drop periods where ALL stocks are NaN
    returns = returns.dropna(how="all")

    logger.info(f"Final return panel shape: {returns.shape}")
    return returns

# ========================================================
# 4. Save to Excel
# ========================================================
def save_to_excel(returns, filename="StockReturns2.xlsx"):
    logger.info(f"Saving return panel to {filename}…")
    returns.to_excel(filename, sheet_name="StockReturns")
    logger.success(f"Saved successfully as {filename}")


# ========================================================
# MAIN
# ========================================================
if __name__ == "__main__":
    all_data, failed = download_prices(TICKERS, start="2014-01-01")
    if failed:
        logger.warning("Some tickers failed to download:")
        for f in failed:
            logger.warning(f"- {f}")
        logger.warning("You can rerun the script if needed.")

    returns2 = build_return_panel(all_data, freq="D")
    returns = build_return_panel(all_data, freq="W-FRI")
    save_to_excel(returns, filename="StockReturns2Weekly.xlsx")
    save_to_excel(returns2, filename="StockReturns_Daily2.xlsx")
