# src/utils/shap_utils.py
from typing import Tuple
import numpy as np
import pandas as pd
import shap

from sklearn.ensemble import RandomForestRegressor

from src.models.random_forest import RandomForestModel
from src.models.ann import ANNModel
from src.models.kan import KANModel

def _is_tree_model(model_wrapper) -> bool:
    raw = getattr(model_wrapper, "model", None)
    return isinstance(model_wrapper, RandomForestModel) or isinstance(raw, RandomForestRegressor)

def build_explainer(model_wrapper, X_background: pd.DataFrame):
    raw_model = getattr(model_wrapper, "model", None)

    if _is_tree_model(model_wrapper):
        explainer = shap.TreeExplainer(raw_model)
        mode = "tree"
    else:
        feature_names = list(X_background.columns)

        def predict_fn(data_as_ndarray):
            df = pd.DataFrame(data_as_ndarray, columns=feature_names)
            y_pred = model_wrapper.predict(df)
            return np.asarray(y_pred).reshape(-1)

        background = X_background.sample(
            min(50, len(X_background)), random_state=0
        )
        explainer = shap.KernelExplainer(predict_fn, background)
        mode = "kernel"

    return explainer, mode



def compute_shap_values(
    model_wrapper,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    max_background: int = 100,
    max_samples: int = 200,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
    """
   
    """
    X_bg = X_train.sample(
        n=min(max_background, len(X_train)),
        random_state=0,
    )

    X_sample = X_test.sample(
        n=min(max_samples, len(X_test)),
        random_state=0,
    )

    explainer, mode = build_explainer(model_wrapper, X_bg)

    if mode == "tree":
        shap_values = explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
    else:
        shap_values = explainer.shap_values(X_sample.values)

    return X_sample, np.array(shap_values)