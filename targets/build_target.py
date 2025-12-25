# targets/build_target.py

import pandas as pd
from config import HORIZON_DAYS

def build_weekly_target(returns_long, sector_map):
    """
    Builds HORIZON_DAYS forward return target.
    """

    df = returns_long.copy()

    # Merge sector information
    df = df.merge(sector_map, on="stock", how="left")

    # Sort properly
    df = df.sort_values(["stock", "date"])

    # Forward HORIZON_DAYS return
    df["y"] = (
        df.groupby("stock")["return"]
          .rolling(HORIZON_DAYS)
          .sum()
          .shift(-HORIZON_DAYS)
          .reset_index(level=0, drop=True)
    )

    # Drop rows where target is undefined (last HORIZON_DAYS per stock)
    df = df.dropna(subset=["y"])

    return df[["date", "stock", "y", "sector"]]
