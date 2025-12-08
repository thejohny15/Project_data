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
def download_prices(tickers, start="2015-01-01"):
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
def build_return_panel(all_data: dict):
    logger.info("Aligning price series…")
    price_df = pd.DataFrame(all_data)
    (price_df == 0).any()
    zero_price_mask = (price_df == 0).any()
    print("Any zero prices per ticker:")
    print(zero_price_mask)


    logger.info(f"Raw price panel shape: {price_df.shape}")

    # SIMPLE RETURNS = pct change
    returns = price_df.pct_change()

    # drop days where ALL stocks have NaN (market holidays)
    returns = returns.dropna(how="all")

    logger.info(f"Final return panel shape: {returns.shape}")

    return returns


# ========================================================
# 4. Save to Excel
# ========================================================
def save_to_excel(returns, filename="StockReturns.xlsx"):
    logger.info(f"Saving return panel to {filename}…")
    returns.to_excel(filename, sheet_name="StockReturns")
    logger.success(f"Saved successfully as {filename}")


# ========================================================
# MAIN
# ========================================================
if __name__ == "__main__":
    all_data, failed = download_prices(TICKERS, start="2015-01-01")

    if failed:
        logger.warning("Some tickers failed to download:")
        for f in failed:
            logger.warning(f"- {f}")
        logger.warning("You can rerun the script if needed.")

    returns = build_return_panel(all_data)
    save_to_excel(returns)

