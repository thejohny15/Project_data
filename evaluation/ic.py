# evaluation/ic.py

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr


def compute_daily_ic(
    predictions: pd.DataFrame,
    targets: pd.DataFrame,
    method: str = "spearman"
) -> pd.DataFrame:

    preds = predictions.copy()
    targs = targets.copy()

    preds["date"] = pd.to_datetime(preds["date"])
    targs["date"] = pd.to_datetime(targs["date"])

    df = preds.merge(
        targs[["date", "stock", "y"]],
        on=["date", "stock"],
        how="inner"
    )

    if df.empty:
        raise ValueError("Merged prediction/target DataFrame is empty.")

    ic_records = []

    for date, g in df.groupby("date"):

        g = g[["score", "y"]].dropna()

        if len(g) < 5 or g["score"].nunique() < 3:
            continue

        scores = g["score"].to_numpy()
        rets   = g["y"].to_numpy()

        try:
            if method == "spearman":
                ic, _ = spearmanr(scores, rets)
            elif method == "pearson":
                ic, _ = pearsonr(scores, rets)
            else:
                raise ValueError("method must be 'spearman' or 'pearson'")
        except Exception:
            continue

        if isinstance(ic, (int, float, np.number)) and np.isfinite(ic):
            ic_records.append({"date": date, "ic": ic})

    ic_df = pd.DataFrame(ic_records)

    if ic_df.empty:
        raise ValueError("IC DataFrame is empty — no valid daily ICs computed.")

    return ic_df


def summarize_ic(ic_df: pd.DataFrame) -> dict:

    ic = ic_df["ic"].dropna().to_numpy()
    T = len(ic)

    if T == 0:
        raise ValueError("No IC observations to summarize.")

    mean_ic = np.mean(ic)
    std_ic  = np.std(ic, ddof=1)
    t_stat  = mean_ic / (std_ic / np.sqrt(T)) if std_ic > 0 else np.nan

    return {
        "mean_ic": float(mean_ic),
        "std_ic": float(std_ic),
        "t_stat": float(t_stat),
        "n_obs": int(T)
    }
