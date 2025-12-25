# targets/build_target.py

import pandas as pd
import numpy as np
from config import HORIZON_DAYS

def build_weekly_target(returns, sector_map):
    """
    returns: DataFrame with columns [date, stock, return]
    sector_map: DataFrame with columns [stock, sector]
    """

    # merge sector info
    df = returns.merge(sector_map, on="stock", how="left")

    # compute forward 5-day return
    df["R_fwd"] = (
        df.groupby("stock")["return"]
        .rolling(HORIZON_DAYS)
        .sum()
        .shift(-HORIZON_DAYS)
        .reset_index(level=0, drop=True)
    )

    # sector demeaning
    df["R_sector_mean"] = (
        df.groupby(["date", "sector"])["R_fwd"]
        .transform("mean")
    )

    df["R_rel"] = df["R_fwd"] - df["R_sector_mean"]

    # cross-sectional standardization
    df["y"] = (
        df.groupby("date")["R_rel"]
        .transform(lambda x: (x - x.mean()) / x.std())
    )

    return df[["date", "stock", "y"]]
