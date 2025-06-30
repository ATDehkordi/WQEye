# src/components/ml_hyperparam_ann_component.py

import streamlit as st
import numpy as np
import pandas as pd
import time
from datetime import datetime
from sklearn.model_selection import KFold, ParameterGrid
from src.config.state_manager import StateManager
from src.utils.scaling_utils import mape, r_squared, ytest_to_initial_scale

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import RMSprop

def ann_tuning_ui(X_train, y_train, X_test, k_folds, scalers, page_name: str = "machine_learning"):
    start_time = time.time()

    param_grid= StateManager.get_page_state(page_name, 'param_grid')
    param_grid['num_of_layers_neurons'] = [[X_train.shape[1]], [X_train.shape[1], X_train.shape[1]]]

    param_names = list(param_grid.keys())
    results_dict = {key: [] for key in param_names}
    results_dict.update({"Val_MAPE": [], "Val_R2": []})

    param_combinations = list(ParameterGrid(param_grid))
    total_combinations = len(param_combinations)
    
    st.write(f"Total parameter combinations: **{total_combinations}**")
    status = st.status("Starting tuning process...", expanded=False)
    progress_bar = st.progress(0)
    results_placeholder = st.empty()

    for i, params in enumerate(param_combinations):
        status.update(label=f"Testing Combination {i+1}/{total_combinations}")


        kf = KFold(n_splits=k_folds)
        # Initialize lists for fold metrics
        val_mape, val_r2 = [], []
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):

            # Build model
            inputs = keras.Input(shape=(X_train.shape[1],))
            x = inputs

            for units in params['num_of_layers_neurons']:
                x = layers.Dense(units, activation=params['act_function_hidden_layers'])(x)

            outputs = layers.Dense(1, activation=params['act_function_output_layers'])(x)
            model = keras.Model(inputs=inputs, outputs=outputs)

            optimizer_name = params['optimizer_func'].lower()
            learning_rate = params['lr']
            
            if optimizer_name == 'adam':
                optimizer = Adam(learning_rate=learning_rate)
            elif optimizer_name == 'sgd':
                optimizer = SGD(learning_rate=learning_rate)
            elif optimizer_name == 'rmsprop':
                optimizer = RMSprop(learning_rate=learning_rate)
            else:
                # Default to Adam if optimizer is not recognized
                st.warning(f"Optimizer '{optimizer_name}' not recognized. Defaulting to Adam.")
                optimizer = Adam(learning_rate=learning_rate)

            model.compile(optimizer=optimizer, loss=params['loss_func'], metrics=['mean_absolute_error'])

            # Train model
            history = model.fit(
                X_train.iloc[train_idx], y_train.iloc[train_idx],
                epochs=params['num_of_epochs'],
                batch_size=params['batch_size'],
                verbose=0
            )

            min_max_scalerY = scalers['min_max_scalerY']
            transformerY = scalers['transformerY']
            shift_value_Y = scalers['shift_value_Y']

            y_val_pred_rescaled = model.predict(X_train.iloc[val_idx], verbose=0)
            y_pred_rescaled = model.predict(X_test, verbose=0)

            y_val_pred_original = np.squeeze(ytest_to_initial_scale(
                y_val_pred_rescaled, min_max_scalerY, transformerY, shift_value_Y
            ))
            y_val_true_original = np.squeeze(ytest_to_initial_scale(
                y_train[val_idx], min_max_scalerY, transformerY, shift_value_Y
            ))

            mape_val = mape(y_val_true_original,y_val_pred_original)
            r2_val = r_squared(y_val_true_original, y_val_pred_original)

            val_mape.append(mape_val)
            val_r2.append(r2_val)

        for name in param_names:
            results_dict[name].append(params[name])
        results_dict["Val_MAPE"].append(np.mean(val_mape))
        results_dict["Val_R2"].append(np.mean(val_r2))

    df = pd.DataFrame(results_dict)

    # 2) Remove rows with infinities or NaNs (if necessary)
    df = df.replace( [np.inf, -np.inf], np.nan).dropna(subset=["Val_MAPE", "Val_R2"])

    # 1) Rank MAPE from best (1) to worst (larger number)
    df["MAPE_rank"] = df["Val_MAPE"].rank(method="min", ascending=True)

    # 2) Rank R2 from best (1) to worst
    df["R2_rank"] = df["Val_R2"].rank(method="min", ascending=False)

    # 3) Combine the ranks by some weighting scheme
    # For simplicity, let's do equal weighting (mean of the two ranks)
    df["combined_rank"] = (df["MAPE_rank"] + df["R2_rank"]) / 2.0

    # 4) Sort by the combined rank
    df_sorted = df.sort_values(by="combined_rank", ascending=True)

    end_time = time.time()
    tuning_duration = end_time - start_time
    
    best_params = df_sorted.iloc[0].to_dict()
    best_params_metadata = {
        "model_name": "Artificial Neural Network",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "val_mape": best_params["Val_MAPE"],
        "val_r2": best_params["Val_R2"],
        "parameters": {k: v for k, v in best_params.items() if k in param_names},
        "tuning_duration_seconds": tuning_duration,
        "total_combinations": total_combinations,
        "k_folds": k_folds
    }

    # Store best parameters in StateManager
    StateManager.set_page_state(page_name, 'best_params', best_params_metadata)

    # Display tuning summary
    st.markdown("### Tuning Summary")
    st.write(f"- **Model**: Artificial Neural Network")
    st.write(f"- **Total Combinations Tested**: {total_combinations}")
    st.write(f"- **K-Folds**: {k_folds}")
    st.write(f"- **Tuning Duration**: {tuning_duration:.2f} seconds")

    # Display best parameters in a cleaner format
    st.markdown("##### Best Parameters Found")
    best_params_df = pd.DataFrame({
        "Parameter": [k for k in best_params.keys() if k in param_names + ["Val_MAPE", "Val_R2"]],
        "Value": [best_params[k] for k in best_params.keys() if k in param_names + ["Val_MAPE", "Val_R2"]]
    })
    st.table(best_params_df)


    progress_bar.progress((i + 1) / total_combinations)
    status.update(label="Tuning Complete! Ranking results...", state="complete")

    return df_sorted
