# File: src/models/random_forest.py
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, ParameterGrid
import pandas as pd
import numpy as np
import streamlit as st

from src.models.base_model import BaseMLModel
from src.utils.scaling_utils import mape, r_squared, ytest_to_initial_scale


class RandomForestModel(BaseMLModel):
    """Random Forest model implementation."""

    def __init__(self):
        self.model = None
        self.best_params = None

    @property
    def name(self) -> str:
        return "Random Forest"
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> None:
        """Train the Random Forest model with the best parameters."""
        params = kwargs.get("params", self.best_params if self.best_params else {})
        # --- FIX: Explicitly cast parameters to integer to avoid scikit-learn type errors ---
        # This handles cases where parameters are stored as floats (e.g., 100.0) from a DataFrame.
        n_estimators_val = int(params.get('n_estimators', 100))
        min_samples_split_val = int(params.get('min_samples_split', 2))
        min_samples_leaf_val = int(params.get('min_samples_leaf', 1))
        
        self.model = RandomForestRegressor(
            n_estimators=n_estimators_val,
            min_samples_split=min_samples_split_val,
            min_samples_leaf=min_samples_leaf_val,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)


    def predict(self, X: pd.DataFrame, **kwargs) -> pd.Series:
        """Make predictions using the trained model."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        return pd.Series(self.model.predict(X), index=X.index)
    

    def tune(self, X_train: pd.DataFrame, y_train: pd.Series, 
             param_grid: dict, k_folds: int, scalers: dict, **kwargs) -> pd.DataFrame:
        """Tune hyperparameters using grid search and cross-validation."""
        param_names = list(param_grid.keys())
        results_dict = {key: [] for key in param_names}
        results_dict.update({"Val_MAPE": [], "Val_R2": []})

        param_combinations = list(ParameterGrid(param_grid))
        total_combinations = len(param_combinations)

        st.write(f"Total parameter combinations: **{total_combinations}**")
        status = st.status("Starting tuning process...", expanded=False)
        progress_bar = st.progress(0)

        # for params in param_combinations:
        for i, params in enumerate(param_combinations):
            status.update(label=f"Testing Combination {i+1}/{total_combinations}")
            kf = KFold(n_splits=k_folds)
            val_mape, val_r2 = [], []

            for train_idx, val_idx in kf.split(X_train):
               # --- FIX: Explicitly cast parameters to integer inside the tuning loop as well ---
                n_estimators_val = int(params.get('n_estimators', 100))
                min_samples_split_val = int(params.get('min_samples_split', 2))
                min_samples_leaf_val = int(params.get('min_samples_leaf', 1))

                model = RandomForestRegressor(
                    n_estimators=n_estimators_val,
                    min_samples_split=min_samples_split_val,
                    min_samples_leaf=min_samples_leaf_val,
                    random_state=42,
                    n_jobs=-1
                )
                model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])

                # Rescale predictions
                y_val_pred_rescaled = model.predict(X_train.iloc[val_idx])
                y_val_pred_original = np.squeeze(ytest_to_initial_scale(
                    y_val_pred_rescaled, scalers['min_max_scalerY'], 
                    scalers['transformerY'], scalers['shift_value_Y']
                ))
                y_val_true_original = np.squeeze(ytest_to_initial_scale(
                    y_train.iloc[val_idx], scalers['min_max_scalerY'], 
                    scalers['transformerY'], scalers['shift_value_Y']
                ))

                # Compute metrics
                val_mape.append(mape(y_val_true_original, y_val_pred_original))
                val_r2.append(r_squared(y_val_true_original, y_val_pred_original))

            for name in param_names:
                results_dict[name].append(params[name])
            results_dict["Val_MAPE"].append(np.mean(val_mape))
            results_dict["Val_R2"].append(np.mean(val_r2))
            progress_bar.progress((i + 1) / total_combinations)

        # Create and rank results
        df = pd.DataFrame(results_dict)
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["Val_MAPE", "Val_R2"])
        df["MAPE_rank"] = df["Val_MAPE"].rank(method="min", ascending=True)
        df["R2_rank"] = df["Val_R2"].rank(method="min", ascending=False)
        df["combined_rank"] = (df["MAPE_rank"] + df["R2_rank"]) / 2.0
        df_sorted = df.sort_values(by="combined_rank", ascending=True)
        self.best_params = df_sorted.iloc[0][param_names].to_dict()

        status.update(label="Tuning Complete! Ranking results...", state="complete")
        return df_sorted, param_names
    


    def get_default_param_grid(self) -> dict:
        """Return the default hyperparameter grid."""
        return {
            'n_estimators': [50, 100, 200],
            # 'max_depth': [10, 20, None],
            'min_samples_split': [2,3,4,5],
            'min_samples_leaf': [1,2,3,4,5],
            # 'max_features': ['sqrt'],
            # 'bootstrap': [True]
        }
    
    def get_param_definitions(self) -> dict:
        """Return the parameter definitions for UI configuration."""
        return {
            'n_estimators': {
                'label': 'Number of Trees',
                'ui_widget': 'text_list',
                'type': int,
                'help': "Enter integer values, separated by commas (e.g., 50, 100, 200).",
                'placeholder': 'e.g., 50, 100, 200'
            },
            # 'max_depth': {
            #     'label': 'Max Depth of Tree',
            #     'ui_widget': 'text_list',
            #     'type': int,
            #     'help': "Enter integer values (e.g., 5, 10, 20). Use 'None' for no limit.",
            #     'placeholder': 'e.g., 5, 10, 20'

            # },
            'min_samples_split': {
                'label': 'Min Samples Split',
                'ui_widget': 'text_list',
                'type': int,
                'help': "Enter integer values (e.g., 2, 5, 10).",
                'placeholder': 'e.g., 2, 5, 10'
            },
            'min_samples_leaf': {
                'label': 'Min Samples Leaf',
                'ui_widget': 'text_list',
                'type': int,
                'help': "Enter integer values (e.g., 1, 2, 4).",
                'placeholder': 'e.g., 1, 2, 4'

            }
            # 'max_features': {
            #     'label': 'Max Features',
            #     'ui_widget': 'multiselect',
            #     'options': ['sqrt', 'log2', 0.5, 1.0],
            #     'type': str,
            #     'help': "Select options for the number of features to consider at each split."
            # },
            # 'bootstrap': {
            #     'label': 'Bootstrap',
            #     'ui_widget': 'multiselect',
            #     'options': [True, False],
            #     'type': bool,
            #     'help': "Enable or disable bootstrap sampling."
            # }
        }