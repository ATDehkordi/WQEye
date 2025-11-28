# src/utils/evaluation.py
import numpy as np
from src.utils.scaling_utils import inverse_transform_y
from src.utils.metrics import compute_metrics

def evaluate_on_test_set(
    model,
    X_test,
    y_test,
    scalers,
    metric_names=None,
):
    """
    """
    y_pred_scaled = model.predict(X_test)

    y_pred_original = np.squeeze(inverse_transform_y(y_pred_scaled, scalers))
    y_test_original = np.squeeze(inverse_transform_y(y_test, scalers))

    metrics = compute_metrics(y_test_original, y_pred_original, metric_names)

    return y_test_original, y_pred_original, metrics
