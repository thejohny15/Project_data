"""
TSFL 2-Stage (2S) Factor Construction – Yahoo Finance version.

Problem with TSFL trend following history too short and did not find a replacement.
Removing TSFL Most Crowded Long would buy me an extra 100 data points, but I want to keep it.
Going for daily data
Have a check at the construction of intrest rate factor and fixed income factor
Because TSFL factors are correlated, we estimate factor exposures with a ridge penalty to stabilize betas and avoid multicollinearity problems.
"""

from __future__ import annotations
from datetime import date, timedelta

from datetime import date
from typing import Sequence, Final

import argparse
import time

import numpy as np
import pandas as pd
import statsmodels.api as sm
from loguru import logger
import yfinance as yf
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Configuration / Static Mappings
# ---------------------------------------------------------------------------

#TSFL factor names 
from typing import Final

TICKER_NAME_MAP: Final[dict[str, str]] = {
    # Equity / broad markets
    "ACWI": "TSFL Equity US",                      # MSCI ACWI ETF (proxy for MXCXDMHR)

    # Rates / bonds
    "IEF": "TSFL Interest Rates",                  # 7–10Y US Treasuries (proxy for global gov 7–10Y)
    "LQD": "TSFL US IG",                           # US investment grade credit (LUACTRUU)
    "HYG": "TSFL US HY",                           # US high yield credit (LF98TRUU)
    "IEAC.L": "TSFL EU IG",                        # EUR IG corporates (Euro IG proxy)
    "IHYG.L": "TSFL EU HY",                        # EUR HY corporates (Euro HY proxy)
    "EMB": "TSFL EM Credit",                       # EM USD sovereign/corp (EMUSTRUU proxy)

    # Commodities
    "DBC": "TSFL Commodities",   # Invesco DB Commodity Index Tracking Fund

    # Emerging markets equity
    "EEM": "TSFL EM Equity",                       # MSCI EM ETF (M1EF proxy)

    # FX
    "CEW": "TSFL Currency FX",                     # EM currency basket (MXEF0CX0 proxy)
    "DX-Y.NYB": "TSFL USD FX",                         # US Dollar Index (DXY Curncy)

    # Inflation
    "TIP": "TSFL US Inflation",                    # US TIPS ETF (BCIT5T proxy)

    # Style factors – MSCI World
    "^105868-USD-STRD": "TSFL Value",              # MSCI WORLD VALUE index 
    "^105867-USD-STRD": "TSFL Growth",             # MSCI WORLD GROWTH index 
    "IWMO.L": "TSFL Momentum",   # iShares Edge MSCI World Momentum Factor UCITS ETF
    "IWV": "TSFL Small Cap",                      # iShares Russell 3000 ETF as global small-cap proxy (SGESR3KX)
    "IWQU.L": "TSFL Quality",                      # iShares Edge MSCI World Quality Factor UCITS (M1WOQU proxy) :contentReference[oaicite:4]{index=4}

    # Alternative premia / hedge-fund style
    "ACWV": "TSFL Low Risk",                       # Global minimum-volatility equity (SAW1LRGV proxy) 
    "UUP": "TSFL Foreign Exchange Carry",   # Invesco DB US Dollar Index Bullish NEED TO DOUBLE CHECK THE MEANING
    "BKLN": "TSFL Fixed Income Carry",             # Senior loan / credit carry proxy (SGIXBC6E proxy)
    "GVIP": "TSFL Most Crowded Long",              # GS Hedge Industry VIP ETF (GSTHHVIP proxy) 
    "HDG": "TSFL Most Crowded Short",              # Hedge-fund replication proxy (GSCBMSAL proxy)

    # Size (small vs large)
    "IWM": "TSFL Size",                            # US small caps as size proxy (SGEPSBW)

    # Real estate
    "REET": "TSFL Real Estate",                    # Global REIT ETF (NDUWREIT proxy)
}


# Order for generic secondary residualisation (vs full core panel)
SECONDARY_FACTOR_ORDER: Final[list[str]] = [
    "TSFL USD FX",
    "TSFL US Inflation",
    "TSFL Small Cap",
    "TSFL Quality",
    "TSFL Low Risk",
    "TSFL Foreign Exchange Carry",
    "TSFL Fixed Income Carry",
    "TSFL Real Estate",
    "TSFL Currency FX",
]

ANCHOR_FACTORS: Final[list[str]] = ["TSFL Equity US", "TSFL Interest Rates"]
CREDIT_FACTORS: Final[list[str]] = ["TSFL US IG", "TSFL US HY", "TSFL EU IG", "TSFL EU HY"]

EM_FACTOR_EQUITY = "TSFL EM Equity"
EM_FACTOR_CREDIT = "TSFL EM Credit"
COMMODITIES_FACTOR = "TSFL Commodities"
VALUE_FACTOR = "TSFL Value"
GROWTH_FACTOR = "TSFL Growth"
MOMENTUM_FACTOR = "TSFL Momentum"
SIZE_FACTOR = "TSFL Size"


# ---------------------------------------------------------------------------
# Yahoo data fetch: monthly returns panel
# ---------------------------------------------------------------------------

def fetch_returns_from_yahoo(
    tickers: Sequence[str],
    start: date | None = None,
    end: date | None = None,
) -> pd.DataFrame:
    """
    Fetch *daily* returns directly from Yahoo, using the tickers as given.

    `tickers` are Yahoo symbols (same as TICKER_NAME_MAP keys).
    We:
      - download daily adjusted prices
      - extract Adj Close (or Close)
      - compute daily simple returns
    """
    start_str = start.isoformat() if isinstance(start, date) else None
    end_str = end.isoformat() if isinstance(end, date) else None

    logger.info(
        "Downloading DAILY data from Yahoo for %d symbol(s) between %s and %s",
        len(tickers),
        start_str,
        end_str,
    )

    # One multi-ticker download; columns are a MultiIndex: (field, ticker)
    data = yf.download(
        tickers=" ".join(tickers),
        start=start_str,
        end=end_str,
        interval="1d",       # <<< DAILY
        auto_adjust=True,
        progress=False,
        # no group_by="ticker" so level 0 = field, level 1 = ticker
    )

    if data is None:
        raise RuntimeError("Yahoo download returned None. Check your internet connection or ticker symbols.")
    
    if data.empty:
        raise RuntimeError("Yahoo download returned an empty DataFrame.")

    # ------------------------------------------------------------------
    # Extract Adj Close (preferred) or Close, robust to MultiIndex/flat
    # ------------------------------------------------------------------
    if isinstance(data.columns, pd.MultiIndex):
        lvl0 = data.columns.get_level_values(0)

        if "Adj Close" in lvl0:
            adj_close = data["Adj Close"].copy()
        elif "Close" in lvl0:
            adj_close = data["Close"].copy()
        else:
            raise RuntimeError(
                f"No 'Adj Close' or 'Close' found in Yahoo columns: {sorted(set(lvl0))}"
            )
    else:
        # Single ticker case → flat columns
        cols = data.columns
        if "Adj Close" in cols:
            adj_close = data[["Adj Close"]].copy()
        elif "Close" in cols:
            adj_close = data[["Close"]].copy()
        else:
            raise RuntimeError(
                f"No 'Adj Close' or 'Close' found in Yahoo columns: {list(cols)}"
            )
        adj_close.columns = list(tickers)

    # ------------------------------------------------------------------
    # Daily simple returns (one row per trading day)
    # ------------------------------------------------------------------
    # Avoid the deprecation warning for default fill_method
    rets = adj_close.pct_change(fill_method=None).iloc[1:]
    
    # Ensure rets is always a DataFrame
    if isinstance(rets, pd.Series):
        rets = rets.to_frame()
    
    # Drop tickers that are completely NaN
    rets = rets.dropna(how="all", axis=1)

    rets.index = pd.to_datetime(rets.index)
    logger.info("Raw Yahoo DAILY returns panel shape: %s", rets.shape)
    return rets

# ---------------------------------------------------------------------------
# Cleaning: rename to TSFL names and resample/dropna
# ---------------------------------------------------------------------------

def clean_latest_2s_factors_price_data(
    df: pd.DataFrame,
    freq: str | None = None,   # kept only for compatibility; we don't use it
) -> pd.DataFrame:
    """
    Minimal cleaning for 2S construction on daily returns.

    We only:
      1. Rename tickers to canonical TSFL names.
      2. Coerce to numeric.
      3. Drop rows that are completely NaN (keep partial rows and zeros).

    No resampling. No aggressive NaN dropping.
    """

    if df.empty:
        logger.warning("clean_latest_2s_factors_price_data: received empty DataFrame.")
        return df

    # 1. Rename Yahoo tickers -> TSFL names
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [TICKER_NAME_MAP.get(col[0], col[0]) for col in df.columns]
    else:
        df.columns = [TICKER_NAME_MAP.get(col, col) for col in df.columns]

    # 2. Coerce to numeric
    df = df.apply(pd.to_numeric, errors="coerce")

    # optional: keep the index nicely sorted
    df = df.sort_index()

    # 3. Drop rows that are entirely NaN (but keep partial rows and zeros)
    df = df.dropna(axis=0, how="any")

    logger.info("Cleaned (very minimal) panel shape: {}", df.shape)
    return df

# ---------------------------------------------------------------------------
# Factor construction: 2-Stage orthogonalisation
# ---------------------------------------------------------------------------

def _ols_residual(y: pd.Series, X: pd.DataFrame | pd.Series) -> pd.Series:
    """
    Return OLS residuals of y ~ X (no intercept). Gracefully handles NaNs.

    If insufficient observations (< 12) after dropping NaNs, returns (y - y.mean()).
    """
    if isinstance(X, pd.Series):
        X = X.to_frame()
    df = pd.concat([y, X], axis=1).dropna()
    if df.shape[0] < 12:
        return y - y.mean()

    y_c = df.iloc[:, 0]
    X_c = df.iloc[:, 1:]
    model = sm.OLS(y_c, X_c).fit()
    resid = model.resid.reindex(y.index)
    return resid


def calculate_2s_factor_regression_returns(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return wide DataFrame of orthogonalised TSFL factor returns.

    Pipeline:
    1. Core anchors: Equity US, Rates.
    2. Credit = avg residual of IG/HY (US/EU) vs anchors.
    3. Commodities residual vs anchors.
    4. EM composite from residual (EM Equity, EM Credit).
    5. Generic secondary residualisation vs full core panel.
    6. Size (vs core + Small Cap residual).
    7. Momentum (vs core).
    8. Value minus Growth (both residualised vs core).
    9. Quality (orthogonal to core + Value).
    10. Crowded = Long + Short residuals vs (anchors + Momentum).
    """
    # 1. Core anchors
    core = raw_df[ANCHOR_FACTORS].copy()

    # 2. Credit residual average
    credit_resid = []
    for cf in CREDIT_FACTORS:
        if cf in raw_df.columns:
            credit_resid.append(_ols_residual(raw_df[cf], core))
    if credit_resid:
        core["TSFL Credit"] = pd.concat(credit_resid, axis=1).mean(axis=1)

    # 3. Commodities residual vs anchors
    if COMMODITIES_FACTOR in raw_df.columns:
        core[COMMODITIES_FACTOR] = _ols_residual(
            raw_df[COMMODITIES_FACTOR],
            core[ANCHOR_FACTORS],
        )

    # 4. EM composite
    secondary = pd.DataFrame(index=raw_df.index)

    if EM_FACTOR_EQUITY in raw_df.columns and "TSFL Equity US" in core.columns:
        em_equity_resid = _ols_residual(raw_df[EM_FACTOR_EQUITY], core["TSFL Equity US"])
    else:
        em_equity_resid = pd.Series(index=raw_df.index, dtype=float)

    if (
        EM_FACTOR_CREDIT in raw_df.columns
        and "TSFL Credit" in core.columns
        and "TSFL Equity US" in core.columns
    ):
        em_credit_resid = _ols_residual(
            raw_df[EM_FACTOR_CREDIT],
            core[["TSFL Credit", "TSFL Equity US"]],
        )
    else:
        em_credit_resid = pd.Series(index=raw_df.index, dtype=float)

    if not em_equity_resid.empty and not em_credit_resid.empty:
        secondary["TSFL Emerging Markets"] = 0.5 * (em_equity_resid + em_credit_resid)

    # 5. Generic secondary residualisation vs *entire* core panel
    for fac in SECONDARY_FACTOR_ORDER:
        if fac in raw_df.columns:
            secondary[fac] = _ols_residual(raw_df[fac], core)

    # 6. Size (needs small cap residual present)
    if SIZE_FACTOR in raw_df.columns and "TSFL Small Cap" in secondary.columns:
        design = pd.concat([core, secondary["TSFL Small Cap"]], axis=1)
        secondary[SIZE_FACTOR] = _ols_residual(raw_df[SIZE_FACTOR], design)

    # 7. Momentum
    if MOMENTUM_FACTOR in raw_df.columns:
        secondary[MOMENTUM_FACTOR] = _ols_residual(raw_df[MOMENTUM_FACTOR], core)

    # 8. Value minus Growth transformation
    if VALUE_FACTOR in raw_df.columns and GROWTH_FACTOR in raw_df.columns:
        value_resid = _ols_residual(raw_df[VALUE_FACTOR], core)
        growth_resid = _ols_residual(raw_df[GROWTH_FACTOR], core)
        secondary[VALUE_FACTOR] = value_resid - growth_resid

    # 9. Quality (orthogonal to core + Value)
    if "TSFL Quality" in raw_df.columns and VALUE_FACTOR in secondary.columns:
        design = pd.concat([core, secondary[VALUE_FACTOR]], axis=1)
        secondary["TSFL Quality"] = _ols_residual(raw_df["TSFL Quality"], design)

    # 10. Crowded (Long + Short residuals average)
    crowded_inputs = [
        c for c in ["TSFL Most Crowded Long", "TSFL Most Crowded Short"]
        if c in raw_df.columns
    ]
    if len(crowded_inputs) == 2 and MOMENTUM_FACTOR in secondary.columns:
        design = pd.concat([core[ANCHOR_FACTORS], secondary[MOMENTUM_FACTOR]], axis=1)
        long_resid = _ols_residual(raw_df["TSFL Most Crowded Long"], design)
        short_resid = _ols_residual(raw_df["TSFL Most Crowded Short"], design)
        secondary["TSFL Crowded"] = long_resid + short_resid

    tsfl = pd.concat([core, secondary], axis=1)
    tsfl = tsfl.loc[:, ~tsfl.columns.duplicated()]
    return tsfl


# ---------------------------------------------------------------------------
# Output helper
# ---------------------------------------------------------------------------

def export_tsfl_to_excel(tsfl: pd.DataFrame, path: str) -> None:
    """Write wide TSFL factor matrix to a single Excel sheet."""
    tsfl.to_excel(path, sheet_name="TSFL_2S_Factors")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    return date.fromisoformat(s)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TSFL 2-Stage factor construction using Yahoo Finance data.",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD). If omitted, Yahoo default is used.",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD). If omitted, Yahoo default is used.",
    )
    parser.add_argument(
        "--freq",
        type=str,
        default=None,
        help="Optional resample frequency (e.g. 'ME' for month-end). "
            "Default: None = keep daily data.",
    )

    parser.add_argument(
        "--excel-out",
        type=str,
        default=None,
        help="Optional path to write factors to Excel (.xlsx).",
    )
    return parser


def plot_corr_heatmap(corr: pd.DataFrame, title: str, filename: str | None = None) -> None:
    """Plot a correlation matrix as a colored heatmap with overlaid numbers."""
    fig, ax = plt.subplots(figsize=(12, 10))

    # Heatmap
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Correlation", rotation=270, labelpad=15)

    # Axis ticks
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index, fontsize=8)

    # Add correlation numbers on each cell
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            value = float(corr.values[i, j])
            text_color = "white" if abs(value) > 0.5 else "black"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=7)

    ax.set_title(title, fontsize=14)
    fig.tight_layout()

    if filename:
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def main(argv: list[str] | None = None) -> None:
    parser = get_parser()
    args = parser.parse_args()

    # If user didn’t specify a start, default to 10 years ago
    if args.start is None:
        default_start = date.today() - timedelta(days=365 * 10)
        start = default_start
    else:
        start = _parse_date(args.start)
    end = _parse_date(args.end) if args.end is not None else None
    tickers_to_use = list(TICKER_NAME_MAP.keys())

    # 1. Fetch Yahoo returns panel 
    raw_df = fetch_returns_from_yahoo(tickers_to_use, start=start, end=end)

    # 2. Clean + rename to TSFL names
    cleaned_df = clean_latest_2s_factors_price_data(raw_df, freq=args.freq)

    # 3. Build TSFL 2S factors
    tsfl = calculate_2s_factor_regression_returns(cleaned_df)

    logger.info("✅ TSFL 2S regression factors calculated successfully.")
    logger.info(
        "Factors computed ({}):\n - {}",
        len(tsfl.columns),
        "\n - ".join(tsfl.columns),
    )
    logger.debug("Data preview:\n{}", tsfl.tail())

    # 4. Optional Excel output
    if args.excel_out:
        export_tsfl_to_excel(tsfl, args.excel_out)
        logger.info("📄 Excel written to: {}", args.excel_out)
    raw_df.to_excel("raw_weekly_aligned.xlsx")
    tsfl.to_excel("tsfl_2s_factors.xlsx")
    corr_tsfl = tsfl.corr()
    corr_tsfl.to_excel("tsfl_2s_factors_correlation.xlsx")
    plot_corr_heatmap(
        corr_tsfl,
        title="TSFL 2-Step Daily Factor Correlations",
        filename="corr_tsfl_2s_factors.png",
)

    


if __name__ == "__main__":
    main()
