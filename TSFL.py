"""
TSFL 2-Stage (2S) Factor Construction – Yahoo Finance version (robust).

Notes / design choices
- Daily adjusted prices are downloaded from Yahoo (auto_adjust=True).
- Returns are computed as simple % returns with NO implicit filling.
- Weekly returns are compounded from daily returns (W-FRI), if requested.
- We keep partial rows (do NOT drop rows with any NaN) because different exchanges have different holidays.
- Better safeguards:
  * handles Yahoo MultiIndex output robustly
  * detects missing tickers and retries individually (optional)
  * avoids MultiIndex dtype errors
  * reports short histories and excessive exact-zero return shares (diagnostic only)
  * fixes duplicate ticker issue: IWM is only used once
- Size proxy fix:
  * IWM is kept as "TSFL Small Cap"
  * IJR is used as "TSFL Size" (S&P SmallCap 600 ETF) as a second size-related proxy, got to check this again, I liked a proposition of S&P - small
"""

from __future__ import annotations

from dataclasses import dataclass
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
# Configuration / Static Mappings
# ---------------------------------------------------------------------------

PROJECT_START: Final[pd.Timestamp] = pd.Timestamp("2014-10-10")

TICKER_NAME_MAP: Final[dict[str, str]] = {
    # Equity / broad markets
    "SPY": "TSFL Equity US",                       # US equity market proxy (SPY)

    # Rates / bonds
    "IEF": "TSFL Interest Rates",                  # 7–10Y US Treasuries
    "LQD": "TSFL US IG",                           # US investment grade credit
    "HYG": "TSFL US HY",                           # US high yield credit
    "IEAC.L": "TSFL EU IG",                        # EUR IG corporates (London)
    "IHYG.L": "TSFL EU HY",                        # EUR HY corporates (London)
    "EMB": "TSFL EM Credit",                       # EM USD sovereign/corp debt

    # Commodities
    "DBC": "TSFL Commodities",                     # broad commodities proxy

    # Emerging markets equity
    "EEM": "TSFL EM Equity",                       # EM equity proxy

    # FX
    "CEW": "TSFL Currency FX",                     # EM currency basket proxy
    "DX-Y.NYB": "TSFL USD FX",                     # US Dollar Index (often works; see safeguards)

    # Inflation
    "TIP": "TSFL US Inflation",                    # US TIPS proxy

    # Style factors – MSCI World (Yahoo indices)
    "^105868-USD-STRD": "TSFL Value",              # MSCI WORLD VALUE index
    "^105867-USD-STRD": "TSFL Growth",             # MSCI WORLD GROWTH index

    # Style ETFs (London)
    "IWMO.L": "TSFL Momentum",                     # MSCI World Momentum (London)
    "IWQU.L": "TSFL Quality",                      # MSCI World Quality (London)

    # Size proxies (FIX: no duplicate IWM)
    "IWM": "TSFL Small Cap",                       # Russell 2000 ETF (small caps)
    "IJR": "TSFL Size",                            # S&P SmallCap 600 ETF (alternative size proxy)

    # Alternative premia / hedge-fund style
    "ACWV": "TSFL Low Risk",                       # global min-vol proxy
    "UUP": "TSFL Foreign Exchange Carry",          # (keep as you had; interpretation to double-check)
    "BKLN": "TSFL Fixed Income Carry",             # senior loan / carry proxy
    "GURU": "TSFL Most Crowded Long",              # crowded long proxy
    "HDG": "TSFL Most Crowded Short",              # crowded short proxy

    # Real estate
    "REET": "TSFL Real Estate",                    # global REIT proxy
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
    "TSFL Size",
]

ANCHOR_FACTORS: Final[list[str]] = ["TSFL Equity US", "TSFL Interest Rates"]
CREDIT_FACTORS: Final[list[str]] = ["TSFL US IG", "TSFL US HY", "TSFL EU IG", "TSFL EU HY"]

EM_FACTOR_EQUITY: Final[str] = "TSFL EM Equity"
EM_FACTOR_CREDIT: Final[str] = "TSFL EM Credit"
COMMODITIES_FACTOR: Final[str] = "TSFL Commodities"
VALUE_FACTOR: Final[str] = "TSFL Value"
GROWTH_FACTOR: Final[str] = "TSFL Growth"
MOMENTUM_FACTOR: Final[str] = "TSFL Momentum"
SIZE_FACTOR: Final[str] = "TSFL Size"


# ---------------------------------------------------------------------------
# Helpers: robust price download (multi-ticker) + retry missing individually
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DownloadReport:
    missing: list[str]
    extra: list[str]
    dropped_all_nan: list[str]
    short_history: dict[str, int]


def _extract_price_field(data: pd.DataFrame, tickers: Sequence[str]) -> pd.DataFrame:
    """
    Extract adjusted close series in a robust way.
    - If auto_adjust=True in yf.download, then 'Close' is already adjusted.
    - Yahoo often returns MultiIndex columns: (Field, Ticker).
    """
    if data is None or data.empty:
        raise RuntimeError("Yahoo download returned None/empty.")

    if isinstance(data.columns, pd.MultiIndex):
        lvl0 = data.columns.get_level_values(0)
        if "Close" in set(lvl0):
            prices = data["Close"].copy()
        elif "Adj Close" in set(lvl0):
            prices = data["Adj Close"].copy()
        else:
            raise RuntimeError(f"No Close/Adj Close field found. Fields seen: {sorted(set(lvl0))}")
    else:
        # Single ticker case
        if "Close" in data.columns:
            prices = data[["Close"]].copy()
        elif "Adj Close" in data.columns:
            prices = data[["Adj Close"]].copy()
        else:
            raise RuntimeError(f"No Close/Adj Close in columns: {list(data.columns)}")
        # rename that single column to the ticker
        prices.columns = [str(tickers[0])]

    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    prices.index = pd.to_datetime(prices.index).normalize()
    prices = prices.sort_index()
    prices.columns = [str(c).strip() for c in prices.columns]
    return prices


def fetch_prices_from_yahoo(
    tickers: Sequence[str],
    start: pd.Timestamp,
    end: Optional[pd.Timestamp] = None,
    *,
    auto_adjust: bool = True,
    retry_missing_individually: bool = True,
    min_obs_warn: int = 500,
) -> tuple[pd.DataFrame, DownloadReport]:
    """
    Download DAILY prices for the given tickers (single Yahoo call), with safeguards.
    Optionally retries missing tickers individually and stitches them into the panel.
    """
    tickers = [str(t).strip() for t in tickers if str(t).strip()]
    tickers = list(dict.fromkeys(tickers))  # remove duplicates, keep order
    if not tickers:
        raise ValueError("No tickers provided.")

    start = pd.to_datetime(start).normalize()
    end = pd.to_datetime(end).normalize() if end is not None else None

    start_str = start.date().isoformat()
    end_str = end.date().isoformat() if end is not None else None

    logger.info(
        "Downloading DAILY prices from Yahoo: {} tickers, start={}, end={}, auto_adjust={}",
        len(tickers), start_str, end_str, auto_adjust
    )

    data = yf.download(
        tickers=" ".join(tickers),
        start=start_str,
        end=end_str,
        interval="1d",
        auto_adjust=auto_adjust,
        progress=False,
        group_by="column",
        threads=True,
    )

    if data is None or data.empty:
        raise RuntimeError("Yahoo download returned None/empty for batch request.")

    prices = _extract_price_field(data, tickers=tickers)

    requested = set(tickers)
    got = set(prices.columns)

    missing = [t for t in tickers if t not in got]
    extra = sorted(list(got - requested))

    # Drop completely empty columns
    dropped_all_nan = prices.columns[prices.isna().all(axis=0)].tolist()
    if dropped_all_nan:
        prices = prices.drop(columns=dropped_all_nan)

    # Retry missing tickers one-by-one (often helps with patchy/failed symbols)
    if retry_missing_individually and missing:
        logger.warning("Missing tickers in batch download (will retry individually): {}", missing)
        for t in missing:
            try:
                d = yf.download(
                    tickers=t,
                    start=start_str,
                    end=end_str,
                    interval="1d",
                    auto_adjust=auto_adjust,
                    progress=False,
                    group_by="column",
                    threads=False,
                )
                if d is not None and not d.empty:
                    p = _extract_price_field(d, tickers=[t])
                    if t in p.columns and not p[t].isna().all():
                        prices = prices.join(p[[t]], how="outer")
                        logger.info("Recovered ticker via individual retry: {}", t)
            except Exception as e:
                logger.warning("Retry failed for {}: {}", t, e)

        prices = prices.sort_index()
        got = set(prices.columns)
        missing = [t for t in tickers if t not in got]

    # Short history report
    obs = prices.notna().sum().to_dict()
    short_history = {t: int(obs.get(t, 0)) for t in prices.columns if int(obs.get(t, 0)) < min_obs_warn}

    if missing:
        logger.warning("Still missing after retry: {}", missing)
    if extra:
        logger.warning("Extra columns returned (not requested): {}", extra)
    if short_history:
        logger.warning("Tickers with short history (<{} obs): {}", min_obs_warn, short_history)

    rep = DownloadReport(
        missing=missing,
        extra=extra,
        dropped_all_nan=dropped_all_nan,
        short_history=short_history,
    )
    return prices, rep


# ---------------------------------------------------------------------------
# Returns construction (daily + optional weekly compounded)
# ---------------------------------------------------------------------------

def prices_to_daily_returns(close: pd.DataFrame) -> pd.DataFrame:
    """
    Daily simple returns, no implicit filling.
    Keeps NaNs where price is missing.
    """
    close = close.sort_index()
    daily = close.pct_change(fill_method=None).iloc[1:]
    daily = daily.dropna(how="all", axis=0)  # drop days where all tickers missing
    daily = daily.dropna(how="all", axis=1)  # drop tickers with no returns at all
    return daily


def daily_to_weekly_compounded(daily_rets: pd.DataFrame) -> pd.DataFrame:
    """
    Weekly returns = compounded daily returns within each Fri-anchored week.
    Does NOT fill missing daily returns; if a ticker is missing within a week,
    the weekly value for that ticker may become NaN (which is fine; keep partial rows).
    """
    weekly = (1.0 + daily_rets).resample("W-FRI").prod(min_count=1) - 1.0
    weekly = weekly.dropna(how="all", axis=0)
    weekly = weekly.dropna(how="all", axis=1)
    return weekly


# ---------------------------------------------------------------------------
# Cleaning / renaming
# ---------------------------------------------------------------------------

def rename_to_tsfl_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Ensure plain string columns (no MultiIndex here, but keep safe)
    df.columns = [str(c).strip() for c in df.columns]
    # Rename only known tickers
    rename_map = {k: v for k, v in TICKER_NAME_MAP.items() if k in df.columns}
    df = df.rename(columns=rename_map)
    return df


def minimal_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.sort_index()
    # Keep partial rows; only drop rows where everything is NaN
    df = df.dropna(axis=0, how="all")
    # Drop columns that are all NaN
    df = df.dropna(axis=1, how="all")
    return df


def diagnose_exact_zeros(rets: pd.DataFrame, *, threshold: float = 0.01) -> None:
    """
    Diagnostic only: report tickers with unusually large share of exact zeros.
    """
    denom = rets.notna().sum().replace(0, np.nan)
    zero_share = ((rets == 0).sum() / denom).dropna()
    suspects = zero_share[zero_share > threshold].sort_values(ascending=False)
    if len(suspects) > 0:
        logger.warning("Tickers with >{:.2f}% exact-zero returns:", 100 * threshold)
        for t, zs in suspects.items():
            logger.warning("  {} : {:.2f}%", t, 100 * float(zs))


# ---------------------------------------------------------------------------
# TSFL 2S construction
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
    # Hard check: anchors must exist
    missing_anchors = [c for c in ANCHOR_FACTORS if c not in raw_df.columns]
    if missing_anchors:
        raise KeyError(
            f"Missing anchor columns: {missing_anchors}. "
            f"Available columns: {list(raw_df.columns)}"
        )

    core = raw_df[ANCHOR_FACTORS].copy()

    # 2) Credit residual average vs anchors
    credit_resid = []
    for cf in CREDIT_FACTORS:
        if cf in raw_df.columns:
            credit_resid.append(_ols_residual(raw_df[cf], core))
    if credit_resid:
        core["TSFL Credit"] = pd.concat(credit_resid, axis=1).mean(axis=1)

    # 3) Commodities residual vs anchors
    if COMMODITIES_FACTOR in raw_df.columns:
        core[COMMODITIES_FACTOR] = _ols_residual(raw_df[COMMODITIES_FACTOR], core[ANCHOR_FACTORS])

    secondary = pd.DataFrame(index=raw_df.index)

    # 4) EM composite
    em_equity_resid = pd.Series(index=raw_df.index, dtype=float)
    em_credit_resid = pd.Series(index=raw_df.index, dtype=float)

    if EM_FACTOR_EQUITY in raw_df.columns:
        em_equity_resid = _ols_residual(raw_df[EM_FACTOR_EQUITY], core["TSFL Equity US"])
    if EM_FACTOR_CREDIT in raw_df.columns and "TSFL Credit" in core.columns:
        em_credit_resid = _ols_residual(raw_df[EM_FACTOR_CREDIT], core[["TSFL Credit", "TSFL Equity US"]])

    if em_equity_resid.notna().any() and em_credit_resid.notna().any():
        secondary["TSFL Emerging Markets"] = 0.5 * (em_equity_resid + em_credit_resid)

    # 5) Generic secondary residualisation vs full core
    for fac in SECONDARY_FACTOR_ORDER:
        if fac in raw_df.columns:
            secondary[fac] = _ols_residual(raw_df[fac], core)

    # 6) Momentum orthogonal to core
    if MOMENTUM_FACTOR in raw_df.columns:
        secondary[MOMENTUM_FACTOR] = _ols_residual(raw_df[MOMENTUM_FACTOR], core)

    # 7) Value minus Growth (both residualised vs core)
    if VALUE_FACTOR in raw_df.columns and GROWTH_FACTOR in raw_df.columns:
        value_resid = _ols_residual(raw_df[VALUE_FACTOR], core)
        growth_resid = _ols_residual(raw_df[GROWTH_FACTOR], core)
        secondary[VALUE_FACTOR] = value_resid - growth_resid

    # 8) Quality orthogonal to core + Value
    if "TSFL Quality" in raw_df.columns and VALUE_FACTOR in secondary.columns:
        design = pd.concat([core, secondary[VALUE_FACTOR]], axis=1)
        secondary["TSFL Quality"] = _ols_residual(raw_df["TSFL Quality"], design)

    # 9) Optional: Size (orthogonal to core + Small Cap if present)
    if "TSFL Size" in raw_df.columns and "TSFL Small Cap" in secondary.columns:
        design = pd.concat([core, secondary["TSFL Small Cap"]], axis=1)
        secondary["TSFL Size"] = _ols_residual(raw_df["TSFL Size"], design)

    # 10) Crowded (Long + Short residuals average vs anchors + Momentum)
    crowded_inputs = [c for c in ["TSFL Most Crowded Long", "TSFL Most Crowded Short"] if c in raw_df.columns]
    if len(crowded_inputs) == 2 and MOMENTUM_FACTOR in secondary.columns:
        design = pd.concat([core[ANCHOR_FACTORS], secondary[MOMENTUM_FACTOR]], axis=1)
        long_resid = _ols_residual(raw_df["TSFL Most Crowded Long"], design)
        short_resid = _ols_residual(raw_df["TSFL Most Crowded Short"], design)
        secondary["TSFL Crowded"] = long_resid + short_resid

    tsfl = pd.concat([core, secondary], axis=1)
    tsfl = tsfl.loc[:, ~tsfl.columns.duplicated()]
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
# CLI
# ---------------------------------------------------------------------------

def _parse_date(s: str | None) -> pd.Timestamp | None:
    if not s:
        return None
    return pd.Timestamp(s).normalize()


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="TSFL 2-Stage factor construction (Yahoo Finance, robust).")
    p.add_argument("--start", type=str, default=str(PROJECT_START.date()), help="Start date YYYY-MM-DD")
    p.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD")
    p.add_argument("--freq", type=str, default="W", choices=["D", "W"], help="Return frequency: D=daily, W=weekly compounded (W-FRI)")
    p.add_argument("--retry-missing", action="store_true", help="Retry missing tickers individually (recommended)")
    p.add_argument("--zero-threshold", type=float, default=0.01, help="Report tickers with >threshold exact-zero returns (diagnostic)")
    return p


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    args = get_parser().parse_args()

    start = _parse_date(args.start) or PROJECT_START
    end = _parse_date(args.end) if args.end else None

    tickers_to_use = list(TICKER_NAME_MAP.keys())

    # 1) Download prices (robust)
    close, rep = fetch_prices_from_yahoo(
        tickers_to_use,
        start=start,
        end=end,
        retry_missing_individually=args.retry_missing,
    )

    # 2) Daily returns
    daily = prices_to_daily_returns(close)
    diagnose_exact_zeros(daily, threshold=float(args.zero_threshold))

    # 3) Choose frequency for factor construction
    if args.freq.upper() == "W":
        rets = daily_to_weekly_compounded(daily)
        rets = rets.loc[rets.index >= PROJECT_START]
        raw_out = "raw_returns_weekly.xlsx"
    else:
        rets = daily.loc[daily.index >= PROJECT_START]
        raw_out = "raw_returns_daily.xlsx"

    # 4) Save raw returns for debugging
    rets.to_excel(raw_out)

    # 5) Rename to TSFL names + minimal clean
    rets_tsfl = minimal_clean(rename_to_tsfl_names(rets))

    # 6) Build 2S TSFL factors
    tsfl = calculate_2s_factor_regression_returns(rets_tsfl)

    # 7) Save outputs
    out_name = "tsfl_2s_factors_weekly.xlsx" if args.freq.upper() == "W" else "tsfl_2s_factors_daily.xlsx"
    tsfl.to_excel(out_name)

    corr = tsfl.corr()
    corr_name = "tsfl_2s_factors_weekly_correlation.xlsx" if args.freq.upper() == "W" else "tsfl_2s_factors_daily_correlation.xlsx"
    corr.to_excel(corr_name)

    plot_corr_heatmap(
        corr,
        title=f"TSFL 2-Stage {'Weekly' if args.freq.upper()=='W' else 'Daily'} Factor Correlations",
        filename=f"corr_tsfl_2s_factors_{'weekly' if args.freq.upper()=='W' else 'daily'}.png",
    )

    logger.success("Saved raw returns to {}", raw_out)
    logger.success("Saved TSFL factors to {}", out_name)
    logger.success("Missing tickers after safeguards: {}", rep.missing)


if __name__ == "__main__":
    main()
