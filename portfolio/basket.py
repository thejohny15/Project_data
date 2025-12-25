# portfolio/basket.py

import pandas as pd
from config import LONG_Q, SHORT_Q

def build_basket(scores):
    """
    scores: DataFrame [date, stock, score]
    """

    def _one_day(df):
        n = len(df)
        df = df.sort_values("score", ascending=False)
        df["weight"] = 0.0

        df.iloc[:int(LONG_Q*n), df.columns.get_loc("weight")] = 1.0
        df.iloc[-int(SHORT_Q*n):, df.columns.get_loc("weight")] = -1.0

        return df

    return scores.groupby("date").apply(_one_day).reset_index(drop=True)
