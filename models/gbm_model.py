# models/gbm_model.py

import lightgbm as lgb

def fit_gbm(X_train, y_train, X_val, y_val):
    params = {
        "objective": "regression",
        "learning_rate": 0.03,
        "max_depth": 3,
        "num_leaves": 8,
        "verbosity": -1,
        "seed": 42
    }

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval   = lgb.Dataset(X_val, label=y_val)

    model = lgb.train(
        params,
        dtrain,
        valid_sets=[dval],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=0)  # disables verbose output
        ]
    )

    return model
