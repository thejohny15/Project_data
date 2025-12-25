# evaluation/walk_forward.py

import numpy as np

def walk_forward_splits(
    dates,
    min_train_size=500,
    step=1
):
    """
    Generator yielding walk-forward train/test indices.

    dates : array-like, sorted unique dates
    min_train_size : minimum number of dates for first training window
    step : step size between folds
    """

    dates = np.array(dates)

    for i in range(min_train_size, len(dates) - 1, step):
        train_dates = dates[:i]
        test_date = dates[i]

        yield train_dates, test_date
