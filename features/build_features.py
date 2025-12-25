# features/build_features.py

import pandas as pd

def build_features(betas):
    """
    betas: DataFrame with columns [date, stock, beta_1, ..., beta_11]
    """

    X = betas.copy()

    # example: beta changes
    beta_cols = [c for c in X.columns if c.startswith("beta_")]
    for c in beta_cols:
        X[f"d_{c}"] = X.groupby("stock")[c].diff()

    return X
