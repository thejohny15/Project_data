# models/linear_model.py

import numpy as np
from sklearn.linear_model import Ridge

def fit_ridge(X, y, alpha=1.0):
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(X, y)
    return model

def predict(model, X):
    return model.predict(X)
