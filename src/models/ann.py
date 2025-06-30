# File: src/models/ann.py
import pandas as pd
import numpy as np
import streamlit as st

from sklearn.model_selection import KFold, ParameterGrid
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from src.models.base_model import BaseMLModel
from src.utils.scaling_utils import mape, r_squared, ytest_to_initial_scale

class ANNModel(BaseMLModel):
    """Artificial Neural Network model implementation."""

    def __init__(self):
        self.model = None
        self.best_params = None

    @property
    def name(self) -> str:
        return "Artificial Neural Network"

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> None:
        """Train the ANN model with the best parameters."""
        params = kwargs.get("params", self.best_params if self.best_params else {})
        default_layers = [[X_train.shape[1]], [X_train.shape[1], X_train.shape[1]]]

        self.model = self._build_model(
            input_shape=X_train.shape[1],
            num_of_layers_neurons=params.get('num_of_layers_neurons', default_layers[0]),
            act_function_hidden_layers=params.get('act_function_hidden_layers', 'relu'),
            act_function_output_layers=params.get('act_function_output_layers', 'linear'),
            optimizer_name=params.get('optimizer_func', 'adam'),
            learning_rate=params.get('lr', 1e-3),
            loss_func=params.get('loss_func', 'mean_squared_error')
        )
        self.model.fit(
            X_train, y_train,
            epochs=params.get('num_of_epochs', 100),
            batch_size=params.get('batch_size', 32),
            verbose=0
        )

    def predict(self, X: pd.DataFrame, **kwargs) -> pd.Series:
        """Make predictions using the trained model."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        return pd.Series(self.model.predict(X, verbose=0).flatten(), index=X.index)

    def tune(self, X_train: pd.DataFrame, y_train: pd.Series,
             param_grid: dict, k_folds: int, scalers: dict, **kwargs) -> pd.DataFrame:
        """Tune hyperparameters using grid search and cross-validation."""

        # Add num_of_layers_neurons to param_grid
        layers_configs = [[X_train.shape[1]], [X_train.shape[1], X_train.shape[1]]]
        if 'num_of_layers_neurons' not in param_grid:
            param_grid['num_of_layers_neurons'] = layers_configs
        else:
            # Ensure layers_configs are included
            param_grid['num_of_layers_neurons'] = [
                config for config in param_grid['num_of_layers_neurons'] if config in layers_configs
            ] or layers_configs

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
                # Build model
                model = self._build_model(
                    input_shape=X_train.shape[1],
                    num_of_layers_neurons=params.get('num_of_layers_neurons', layers_configs[0]),
                    act_function_hidden_layers=params.get('act_function_hidden_layers', 'relu'),
                    act_function_output_layers=params.get('act_function_output_layers', 'linear'),
                    optimizer_name=params.get('optimizer_func', 'adam'),
                    learning_rate=params.get('lr', 1e-3),
                    loss_func=params.get('loss_func', 'mean_squared_error')
                )

                # Train model
                model.fit(
                    X_train.iloc[train_idx], y_train.iloc[train_idx],
                    epochs=params.get('num_of_epochs', 100),
                    batch_size=params.get('batch_size', 32),
                    verbose=0
                )

                # Rescale predictions
                y_val_pred_rescaled = model.predict(X_train.iloc[val_idx], verbose=0)
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
            'act_function_hidden_layers': ['relu'],
            'optimizer_func': ['adam'],
            'num_of_epochs': [100],
            'act_function_output_layers': ['linear'],
            'loss_func': ['mean_squared_error'],
            'batch_size': [32],
            'lr': [1e-3]
        }

    def get_param_definitions(self) -> dict:
        """Return the parameter definitions for UI configuration."""
        return {
            'act_function_hidden_layers': {
                'label': 'Hidden Layers Activation',
                'ui_widget': 'multiselect',
                'options': ['relu', 'tanh', 'logistic'],
                'type': str,
                'help': "Select one or more activation functions to test."
            },
            'optimizer_func': {
                'label': 'Optimizer',
                'ui_widget': 'multiselect',
                'options': ['adam', 'sgd', 'rmsprop'],
                'type': str,
                'help': "Select one or more optimizers."
            },
            'num_of_epochs': {
                'label': 'Number of Epochs',
                'ui_widget': 'text_list',
                'type': int,
                'placeholder': 'e.g., 100, 200, 500',
                'help': "Enter integer values, separated by commas (e.g., 100, 200)."
            },
            'act_function_output_layers': {
                'label': 'Output Layer Activation',
                'ui_widget': 'multiselect',
                'options': ['linear', 'sigmoid'],
                'type': str,
                'help': "Select activation for the final output layer."
            },
            'loss_func': {
                'label': 'Loss Function',
                'ui_widget': 'multiselect',
                'options': ['mean_squared_error', 'mean_absolute_error'],
                'type': str,
                'help': "Select the loss function to minimize."
            },
            'batch_size': {
                'label': 'Batch Size',
                'ui_widget': 'text_list',
                'type': int,
                'placeholder': 'e.g., 32, 64',
                'help': "Enter integer values, separated by commas (e.g., 32, 64)."
            },
            'lr': {
                'label': 'Learning Rate',
                'ui_widget': 'text_list',
                'type': float,
                'placeholder': 'e.g., 0.001, 0.0001',
                'help': "Enter float values, separated by commas (e.g., 0.001, 0.0001)."
            }
            
        }

    def _build_model(self, input_shape: int, num_of_layers_neurons: list, 
                     act_function_hidden_layers: str, act_function_output_layers: str, 
                     optimizer_name: str, learning_rate: float, loss_func: str) -> keras.Model:
        """Build and compile the ANN model."""
        inputs = keras.Input(shape=(input_shape,))
        x = inputs

        for units in num_of_layers_neurons:
            x = layers.Dense(units, activation=act_function_hidden_layers)(x)

        outputs = layers.Dense(1, activation=act_function_output_layers)(x)
        model = keras.Model(inputs=inputs, outputs=outputs)

        optimizer_name = optimizer_name.lower()
        if optimizer_name == 'adam':
            optimizer = Adam(learning_rate=learning_rate)
        elif optimizer_name == 'sgd':
            optimizer = SGD(learning_rate=learning_rate)
        elif optimizer_name == 'rmsprop':
            optimizer = RMSprop(learning_rate=learning_rate)
        else:
            optimizer = Adam(learning_rate=learning_rate)

        model.compile(optimizer=optimizer, loss=loss_func, metrics=['mean_absolute_error'])
        return model