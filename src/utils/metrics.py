import numpy as np
from scipy.stats import ks_2samp, linregress


def _to_1d(arr):
    """ """
    return np.asarray(arr, dtype=float).ravel()


def rmse(y, y_hat):
    """Root Mean Squared Error"""
    y = _to_1d(y)
    y_hat = _to_1d(y_hat)
    mask = ~np.isnan(y) & ~np.isnan(y_hat)
    y = y[mask]
    y_hat = y_hat[mask]
    return float(np.sqrt(np.mean((y - y_hat) ** 2)))

def bias(y, y_hat):
    """Mean Bias = mean(pred - true)"""
    y = _to_1d(y)
    y_hat = _to_1d(y_hat)
    mask = ~np.isnan(y) & ~np.isnan(y_hat)
    y = y[mask]
    y_hat = y_hat[mask]
    return float(np.mean(y_hat - y))

def mape(y, y_hat):
    """Mean Absolute Percentage Error"""
    y = _to_1d(y)
    y_hat = _to_1d(y_hat)
    mask = (y != 0) & ~np.isnan(y) & ~np.isnan(y_hat)
    y = y[mask]
    y_hat = y_hat[mask]
    return float(100 * np.mean(np.abs((y - y_hat) / y)))

def r_squared(y, y_hat):
    """ Logarithmic R^2 """
    y = _to_1d(y)
    y_hat = _to_1d(y_hat)
    slope, intercept, r_value, p_value, std_err = linregress(y, y_hat)
    return float(r_value ** 2 * 100)

def ks_test(y, y_hat):
    """ """
    y = _to_1d(y)
    y_hat = _to_1d(y_hat)
    stat, p_value = ks_2samp(y, y_hat)
    return {
        "ks_stat": float(stat),
        "ks_p_value": float(p_value),
    }


METRICS_REGISTRY = {
    "MAPE": {
        "fn": mape,
        "greater_is_better": False,
        "label": "MAPE (%)",
        "fmt": "{:.4f}",
    },
    "R2": {
        "fn": r_squared,
        "greater_is_better": True,
        "label": "R² (%)",
        "fmt": "{:.4f}",
    },
    "RMSE": {
        "fn": rmse,
        "greater_is_better": False,
        "label": "RMSE",
        "fmt": "{:.4f}",
    },
    "Bias": {
        "fn": bias,
        "greater_is_better": False,
        "label": "Bias",
        "fmt": "{:.4f}",
    },
}

DEFAULT_METRICS = ["MAPE", "R2", "RMSE", "Bias"]

def compute_metrics(y_true, y_pred, metric_names=None):
    """ """
    if metric_names is None:
        metric_names = DEFAULT_METRICS

    results = {}

    # Scaler Metric
    for name in metric_names:
        cfg = METRICS_REGISTRY[name]
        results[name] = cfg["fn"](y_true, y_pred)

    ks_results = ks_test(y_true, y_pred)
    results.update(ks_results)  # ks_stat, ks_p_value

    return results
