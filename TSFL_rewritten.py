"""
TSFL 2-Stage (2S) Factor Construction – Yahoo Finance version (DAILY).

We keep the "earlier" behavior:
- Master calendar = SPY trading days (so we do NOT introduce US holidays).
- Per-ticker download (avoids multi-ticker calendar bugs).
- Reindex to master calendar, NO filling.
- Drop ONLY rows where ALL tickers are NaN (so holidays disappear).

Only changes requested:
- Momentum and Quality are built in a controlled order to reduce correlation.

  1) Momentum := resid(raw Momentum ~ core anchors)
  2) Value := resid(raw Value ~ core) - resid(raw Growth ~ core)
  3) Quality := resid(raw Quality ~ core + Momentum + Value)

Size:
- Keep TSFL Small Cap in the sheet (you’ll handle PCA in another file).
"""

from __future__ import annotations

from datetime import date
from typing import Sequence, Final, Optional

import argparse
import numpy as np
import pandas as pd
import statsmodels.api as sm
from loguru import logger
import yfinance as yf
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Tickers -> TSFL factor names (with meaning/proxy notes)
# ---------------------------------------------------------------------------

TICKER_NAME_MAP: Final[dict[str, str]] = {
    # Equity / broad markets
    "SPY": "TSFL Equity US",                      # S&P 500 ETF (broad US equity; core equity anchor)

    # Rates / bonds
    "IEF": "TSFL Interest Rates",                 # 7–10Y US Treasuries (proxy for global gov 7–10Y)
    "LQD": "TSFL US IG",                          # US investment grade credit (LUACTRUU proxy)
    "HYG": "TSFL US HY",                          # US high yield credit (LF98TRUU proxy)
    "IEAC.L": "TSFL EU IG",                       # EUR IG corporates (Euro IG proxy)
    "IHYG.L": "TSFL EU HY",                       # EUR HY corporates (Euro HY proxy)
    "EMB": "TSFL EM Credit",                      # EM USD sovereign/corp (EMUSTRUU proxy)

    # Commodities
    "DBC": "TSFL Commodities",                    # Broad commodities basket proxy

    # Emerging markets equity
    "EEM": "TSFL EM Equity",                      # MSCI Emerging Markets equity proxy

    # FX
    "CEW": "TSFL Currency FX",                    # EM currency basket proxy
    "DX-Y.NYB": "TSFL USD FX",                    # US Dollar Index (DXY proxy)  (may fail sometimes)
    "^DXY": "TSFL USD FX",                        # fallback for DXY if needed

    # Inflation
    "TIP": "TSFL US Inflation",                   # US TIPS proxy

    # Style factors
    "^105868-USD-STRD": "TSFL Value",             # MSCI World Value index proxy
    "^105867-USD-STRD": "TSFL Growth",            # MSCI World Growth index proxy
    "IWMO.L": "TSFL Momentum",                    # MSCI World Momentum ETF proxy
    "IWQU.L": "TSFL Quality",                     # MSCI World Quality ETF proxy

    # Small caps (kept as raw factor series)
    "IWM": "TSFL Small Cap",                      # Russell 2000 ETF proxy

    # Alternative premia / hedge-fund style
    "ACWV": "TSFL Low Risk",                      # Global min-vol equity proxy
    "UUP": "TSFL Foreign Exchange Carry",         # USD bullish ETF proxy
    "BKLN": "TSFL Fixed Income Carry",            # Senior loan / carry-like proxy
    "GURU": "TSFL Most Crowded Long",             # Crowded long proxy
    "HDG": "TSFL Most Crowded Short",             # Crowded short proxy

    # Real estate
    "REET": "TSFL Real Estate",                   # Global REIT ETF proxy
}

# Core anchors
ANCHOR_FACTORS: Final[list[str]] = ["TSFL Equity US", "TSFL Interest Rates"]

# Credit inputs
CREDIT_FACTORS: Final[list[str]] = ["TSFL US IG", "TSFL US HY", "TSFL EU IG", "TSFL EU HY"]

# Canonical factor names
EM_FACTOR_EQUITY = "TSFL EM Equity"
EM_FACTOR_CREDIT = "TSFL EM Credit"
COMMODITIES_FACTOR = "TSFL Commodities"
VALUE_FACTOR = "TSFL Value"
GROWTH_FACTOR = "TSFL Growth"
MOMENTUM_FACTOR = "TSFL Momentum"
QUALITY_FACTOR = "TSFL Quality"
SMALLCAP_FACTOR = "TSFL Small Cap"

# Secondary residualisation order — IMPORTANT:
# remove Momentum and Quality from here (they will be built later in controlled form)
SECONDARY_FACTOR_ORDER: Final[list[str]] = [
    "TSFL USD FX",
    "TSFL US Inflation",
    "TSFL Small Cap",
    "TSFL Low Risk",
    "TSFL Foreign Exchange Carry",
    "TSFL Fixed Income Carry",
    "TSFL Real Estate",
    "TSFL Currency FX",
]


# ---------------------------------------------------------------------------
# Helper: download one ticker
# ---------------------------------------------------------------------------

def _download_one_ticker_prices(
    ticker: str,
    start: Optional[str],
    end: Optional[str],
) -> pd.DataFrame:
    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    if df is None or df.empty:
        return pd.DataFrame()
    df.index = pd.to_datetime(df.index).normalize()
    return df.sort_index()


# ---------------------------------------------------------------------------
# Fetch DAILY returns aligned to a master calendar (SPY trading days)
# ---------------------------------------------------------------------------

def fetch_daily_returns_master_calendar(
    tickers: Sequence[str],
    start: Optional[date] = None,
    end: Optional[date] = None,
    *,
    master_ticker: str = "SPY",
    drop_all_nan_rows: bool = True,
    treat_stale_as_nan: bool = True,
) -> pd.DataFrame:
    """
    Earlier behavior:
    - master calendar from SPY
    - per-ticker download
    - returns aligned to master calendar
    - NO forward/back fill
    - drop rows where ALL tickers are NaN (=> US holidays vanish)
    """
    start_str = start.isoformat() if start else None
    end_str = end.isoformat() if end else None

    # Master calendar
    logger.info("Building master calendar from {}", master_ticker)
    m = _download_one_ticker_prices(master_ticker, start_str, end_str)
    if m.empty:
        raise RuntimeError(f"Could not download master calendar ticker {master_ticker}.")

    master_index = pd.to_datetime(m.index).normalize().sort_values()

    out: dict[str, pd.Series] = {}
    failed: list[str] = []

    for t in tickers:
        logger.info("Downloading {}", t)
        df = _download_one_ticker_prices(t, start_str, end_str)

        if df.empty:
            failed.append(t)
            continue

        # Price series
        if "Adj Close" in df.columns:
            px = df["Adj Close"]
        elif "Close" in df.columns:
            px = df["Close"]
        else:
            failed.append(t)
            continue

        # Force 1D Series
        if isinstance(px, pd.DataFrame):
            px = px.squeeze()
        if not isinstance(px, pd.Series):
            px = pd.Series(px, index=df.index)

        px = px.astype(float)

        # Returns (no fill)
        r = px.pct_change(fill_method=None)

        # Optional stale filter (helps some bond/loan ETFs)
        if treat_stale_as_nan and "Volume" in df.columns:
            vol = df["Volume"]
            if isinstance(vol, pd.DataFrame):
                vol = vol.squeeze()
            if not isinstance(vol, pd.Series):
                vol = pd.Series(vol, index=df.index)
            vol = pd.to_numeric(vol, errors="coerce").fillna(0.0)

            unchanged = px.diff().fillna(np.nan).eq(0.0)
            stale = unchanged & vol.eq(0.0)
            r = r.mask(stale)

        r = r.iloc[1:]                  # drop first NaN
        r = r.reindex(master_index)     # align to master calendar
        r.name = t
        out[t] = r

    if not out:
        raise RuntimeError("All tickers failed or produced empty returns.")

    rets = pd.concat(out.values(), axis=1).sort_index()

    # Drop tickers with no data
    rets = rets.dropna(how="all", axis=1)

    # CRITICAL: remove master-calendar rows where everyone is missing (holidays)
    if drop_all_nan_rows:
        rets = rets.dropna(how="all", axis=0)

    if failed:
        logger.warning("Failed tickers: {}", failed)

    logger.info("Final DAILY returns panel shape: {}", rets.shape)
    return rets


# ---------------------------------------------------------------------------
# Cleaning / renaming
# ---------------------------------------------------------------------------

def clean_latest_2s_factors_price_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep partial rows; drop only all-NaN rows.
    Rename tickers -> TSFL names.
    Combine duplicates (e.g. DXY sources) by first non-null per day.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    out.index = pd.to_datetime(out.index).normalize()
    out = out.sort_index()

    out.columns = pd.Index([str(c).strip() for c in out.columns])
    out = out.rename(columns=TICKER_NAME_MAP)
    out = out.apply(pd.to_numeric, errors="coerce")

    if out.columns.duplicated().any():
        combined = {}
        for col in pd.unique(out.columns):
            cols = [c for c in out.columns if c == col]
            if len(cols) == 1:
                combined[col] = out[col]
            else:
                combined[col] = out[cols].bfill(axis=1).iloc[:, 0]
        out = pd.DataFrame(combined, index=out.index).sort_index()

    out = out.dropna(how="all", axis=0)
    logger.info("Cleaned DAILY panel shape: {}", out.shape)
    return out


# ---------------------------------------------------------------------------
# 2-stage construction (OLS residualisation)
# ---------------------------------------------------------------------------

def _ols_residual(y: pd.Series, X: pd.DataFrame | pd.Series) -> pd.Series:
    if isinstance(X, pd.Series):
        X = X.to_frame()

    df = pd.concat([y, X], axis=1).dropna()
    if df.shape[0] < 12:
        return y - y.mean()

    y_c = df.iloc[:, 0]
    X_c = df.iloc[:, 1:]
    model = sm.OLS(y_c, X_c).fit()
    return model.resid.reindex(y.index)


def calculate_2s_factor_regression_returns(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Same as earlier construction, but Momentum + Quality are built in controlled order.
    """
    if raw_df is None or raw_df.empty:
        raise ValueError("raw_df is empty.")

    missing = [c for c in ANCHOR_FACTORS if c not in raw_df.columns]
    if missing:
        raise KeyError(f"Missing anchor columns: {missing}. Available: {list(raw_df.columns)}")

    core = raw_df[ANCHOR_FACTORS].copy()

    # 1) Credit = avg residual vs anchors
    credit_resid = []
    for cf in CREDIT_FACTORS:
        if cf in raw_df.columns:
            credit_resid.append(_ols_residual(raw_df[cf], core))
    if credit_resid:
        core["TSFL Credit"] = pd.concat(credit_resid, axis=1).mean(axis=1)

    # 2) Commodities residual vs anchors
    if COMMODITIES_FACTOR in raw_df.columns:
        core[COMMODITIES_FACTOR] = _ols_residual(raw_df[COMMODITIES_FACTOR], core[ANCHOR_FACTORS])

    secondary = pd.DataFrame(index=raw_df.index)

    # 3) EM composite
    em_equity_resid = pd.Series(index=raw_df.index, dtype=float)
    em_credit_resid = pd.Series(index=raw_df.index, dtype=float)

    if EM_FACTOR_EQUITY in raw_df.columns:
        em_equity_resid = _ols_residual(raw_df[EM_FACTOR_EQUITY], core["TSFL Equity US"])

    if EM_FACTOR_CREDIT in raw_df.columns and "TSFL Credit" in core.columns:
        em_credit_resid = _ols_residual(raw_df[EM_FACTOR_CREDIT], core[["TSFL Credit", "TSFL Equity US"]])

    if (not em_equity_resid.empty) and (not em_credit_resid.empty):
        secondary["TSFL Emerging Markets"] = 0.5 * (em_equity_resid + em_credit_resid)

    # 4) Generic secondaries (EXCLUDING Momentum & Quality)
    for fac in SECONDARY_FACTOR_ORDER:
        if fac in raw_df.columns:
            secondary[fac] = _ols_residual(raw_df[fac], core)

    # ----------------------------
    # ✅ Controlled Momentum
    # ----------------------------
    if MOMENTUM_FACTOR in raw_df.columns:
        secondary[MOMENTUM_FACTOR] = _ols_residual(raw_df[MOMENTUM_FACTOR], core)

    # ----------------------------
    # ✅ Controlled Value (Value - Growth)
    # ----------------------------
    if VALUE_FACTOR in raw_df.columns and GROWTH_FACTOR in raw_df.columns:
        value_resid = _ols_residual(raw_df[VALUE_FACTOR], core)
        growth_resid = _ols_residual(raw_df[GROWTH_FACTOR], core)
        secondary[VALUE_FACTOR] = value_resid - growth_resid

    # ----------------------------
    # ✅ Controlled Quality
    # Quality := resid(raw Quality ~ core + Momentum + Value)
    # ----------------------------
    if QUALITY_FACTOR in raw_df.columns:
        design_parts = [core]
        if MOMENTUM_FACTOR in secondary.columns:
            design_parts.append(secondary[[MOMENTUM_FACTOR]])
        if VALUE_FACTOR in secondary.columns:
            design_parts.append(secondary[[VALUE_FACTOR]])

        design = pd.concat(design_parts, axis=1)
        secondary[QUALITY_FACTOR] = _ols_residual(raw_df[QUALITY_FACTOR], design)

    # 5) Crowded: residualise Long/Short vs (anchors + Momentum) then sum
    if (
        ("TSFL Most Crowded Long" in raw_df.columns)
        and ("TSFL Most Crowded Short" in raw_df.columns)
        and (MOMENTUM_FACTOR in secondary.columns)
    ):
        design = pd.concat([core[ANCHOR_FACTORS], secondary[[MOMENTUM_FACTOR]]], axis=1)
        long_resid = _ols_residual(raw_df["TSFL Most Crowded Long"], design)
        short_resid = _ols_residual(raw_df["TSFL Most Crowded Short"], design)
        secondary["TSFL Crowded"] = long_resid + short_resid

    tsfl = pd.concat([core, secondary], axis=1)
    tsfl = tsfl.loc[:, ~tsfl.columns.duplicated()]
    tsfl = tsfl.dropna(how="all").sort_index()
    return tsfl


# ---------------------------------------------------------------------------
# Plot helper
# ---------------------------------------------------------------------------

def plot_corr_heatmap(corr: pd.DataFrame, title: str, filename: str | None = None) -> None:
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Correlation", rotation=270, labelpad=15)

    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index, fontsize=8)

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


# ---------------------------------------------------------------------------
# CLI / Main
# ---------------------------------------------------------------------------

def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    return date.fromisoformat(s)


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="TSFL 2-Stage factor construction (DAILY) using Yahoo Finance.")
    p.add_argument("--start", type=str, default="2014-10-10", help="Start date YYYY-MM-DD (default: 2014-10-10)")
    p.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD")
    p.add_argument("--excel-out", type=str, default="tsfl_2s_factors_daily.xlsx", help="Output Excel path")
    return p


def main() -> None:
    args = get_parser().parse_args()

    start = _parse_date(args.start)
    end = _parse_date(args.end)

    tickers_to_use = list(TICKER_NAME_MAP.keys())

    # 1) Fetch DAILY Yahoo returns per ticker aligned to SPY calendar (earlier behavior)
    raw_df = fetch_daily_returns_master_calendar(
        tickers_to_use,
        start=start,
        end=end,
        master_ticker="SPY",
        drop_all_nan_rows=True,     # ensures US holidays do not show up
        treat_stale_as_nan=True,
    )

    # 2) Rename + minimal cleaning
    cleaned_df = clean_latest_2s_factors_price_data(raw_df)

    # 3) Build TSFL factors (with Momentum/Quality fix)
    tsfl = calculate_2s_factor_regression_returns(cleaned_df)

    # 4) Save outputs (same output files as earlier)
    raw_df.to_excel("raw_daily_factor_returns.xlsx")
    cleaned_df.to_excel("raw_daily_factor_returns_renamed.xlsx")
    tsfl.to_excel(args.excel_out, sheet_name="TSFL_2S_Factors")

    corr = tsfl.corr()
    corr.to_excel("tsfl_2s_factors_daily_correlation.xlsx")

    plot_corr_heatmap(
        corr,
        title="TSFL 2-Stage DAILY Factor Correlations (Momentum/Quality adjusted)",
        filename="corr_tsfl_2s_factors_daily.png",
    )

    logger.success("Saved DAILY TSFL factors to {}", args.excel_out)


if __name__ == "__main__":
    main()
