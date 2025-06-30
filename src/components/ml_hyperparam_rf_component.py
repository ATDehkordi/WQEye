# src/components/ml_hyperparam_rf_component.py


from sklearn.ensemble import RandomForestRegressor
import streamlit as st
import numpy as np
import pandas as pd

import time
from datetime import datetime


from src.config.state_manager import StateManager
from sklearn.model_selection import KFold, ParameterGrid
from src.config.state_manager import StateManager
from src.utils.scaling_utils import mape, r_squared, ytest_to_initial_scale

def rfr_tuning_ui(X_train, y_train, X_test, k_folds, scalers, page_name: str = "machine_learning"):
    start_time = time.time()
    param_grid = StateManager.get_page_state(page_name, 'param_grid')
    
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
        val_mape, val_r2 = [], []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            # Initialize Random Forest model with current parameters
            model = RandomForestRegressor(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', None),
                min_samples_split=params.get('min_samples_split', 2),
                min_samples_leaf=params.get('min_samples_leaf', 1),
                max_features=params.get('max_features', 'sqrt'),
                random_state=42,
                n_jobs=-1
            )

            # Train model
            model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])

            # Get scalers for rescaling
            min_max_scalerY = scalers['min_max_scalerY']
            transformerY = scalers['transformerY']
            shift_value_Y = scalers['shift_value_Y']

            # Predict on validation set
            y_val_pred_rescaled = model.predict(X_train.iloc[val_idx])
            y_val_pred_original = np.squeeze(ytest_to_initial_scale(
                y_val_pred_rescaled, min_max_scalerY, transformerY, shift_value_Y
            ))
            y_val_true_original = np.squeeze(ytest_to_initial_scale(
                y_train.iloc[val_idx], min_max_scalerY, transformerY, shift_value_Y
            ))

            # Compute metrics
            mape_val = mape(y_val_true_original, y_val_pred_original)
            r2_val = r_squared(y_val_true_original, y_val_pred_original)

            val_mape.append(mape_val)
            val_r2.append(r2_val)

        # Store results
        for name in param_names:
            results_dict[name].append(params[name])
        results_dict["Val_MAPE"].append(np.mean(val_mape))
        results_dict["Val_R2"].append(np.mean(val_r2))

        progress_bar.progress((i + 1) / total_combinations)

    # Create results dataframe
    df = pd.DataFrame(results_dict)

    # Remove rows with infinities or NaNs
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["Val_MAPE", "Val_R2"])

    # Rank MAPE (lower is better)
    df["MAPE_rank"] = df["Val_MAPE"].rank(method="min", ascending=True)

    # Rank R2 (higher is better)
    df["R2_rank"] = df["Val_R2"].rank(method="min", ascending=False)

    # Combine ranks with equal weighting
    df["combined_rank"] = (df["MAPE_rank"] + df["R2_rank"]) / 2.0

    # Sort by combined rank
    df_sorted = df.sort_values(by="combined_rank", ascending=True)

    end_time = time.time()
    tuning_duration = end_time - start_time
    
    best_params = df_sorted.iloc[0].to_dict()
    best_params_metadata = {
        "model_name": "Random Forest Regressor",
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
    st.write(f"- **Model**: Random Forest Regressor")
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

   
   
    status.update(label="Tuning Complete! Ranking results...", state="complete")

    return df_sorted