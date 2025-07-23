# src/pages/machine_learning.py
import base64
import io
import json
import pickle

import joblib
import streamlit as st
import time
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.services.zip_download_helper import create_zip_buffer, get_download_link
from src.components.ml_hyperparam_rf_component import rfr_tuning_ui
from src.components.ml_hyperparam_form_component import hyperparameter_form_ui
from src.config.state_manager import StateManager
from src.components.ml_hyperparam_ann_component import ann_tuning_ui
from src.components.ml_data_loader_component import load_ml_data
from src.components.ml_model_selector_component import model_selection_ui
from src.models.registry import ModelRegistry
from src.utils.log_scale_transform import ytest_to_initial_scale
from src.utils.scaling_utils import mape, r_squared


PAGE_NAME = "machine_learning"

def show():
    st.title("Machine learning")
    # Load Data
    train_df, test_df, scalers = load_ml_data(page_name=PAGE_NAME)

    # Stop execution if data is not loaded
    if train_df is None or test_df is None:
        st.stop()

    try:
        features = [col for col in train_df.columns if col != 'target']
        X_train = train_df[features]
        y_train = train_df['target']
        X_test = test_df[features]
        y_test = test_df['target']
    except KeyError:
        st.error("Error: A 'target' column was not found in one of your dataframes. Please ensure both files have this column.")
        st.stop()

    tab1, tab2, tab3, tab4 = st.tabs([
        "**① Model Selection**", "**② Hyperparameter Tuning**", "**③ Train & Evaluate**", "**④ Save Model**"
    ])

    # == TAB 1: Model Selection =======================================================
    with tab1:
        model_selection_ui(X_train, X_test, features, train_df, page_name= PAGE_NAME)

    with tab2:
        # Get the selected model name and its configuration
        model_name = StateManager.get_page_state(PAGE_NAME, 'selected_model')
        # model_config = StateManager.get_model_config(model_name)

        # Initialize registry and get model
        registry = ModelRegistry()
        model = registry.get_model(model_name)
        if not model:
            st.error(f"Model {model_name} not found.")
            st.stop()

        default_tuning = st.checkbox(
            "Use Default Tuning Grid", 
            value= True,
            # key="use_defaults_chkbx",
            help="If checked, a pre-defined grid will be used. If unchecked, the form below will allow you to define your own grid."
        )
        # Persist the choice immediately so the UI stays consistent
        StateManager.set_page_state(PAGE_NAME, 'default_tuning', default_tuning)
        
        
        param_grid, k_folds, submitted = hyperparameter_form_ui(model_name=model_name,
                                                                page_name=PAGE_NAME,
                                                                default_tuning=default_tuning,
                                                                X_train=X_train)
        
        if submitted:

            # Save the choices made inside the form
            StateManager.set_page_state(PAGE_NAME, 'param_grid', param_grid)

            if not param_grid:
                st.error("Parameter grid is empty. Please define a grid or use the default.")
                st.stop()

            
            st.info("The tuning process has started. This may take some time.")
            start_time = time.time()
            # Tuning the model
            combinations_result, param_names = model.tune(X_train, y_train, param_grid, k_folds, scalers)

            end_time = time.time()
            tuning_duration = end_time - start_time


            best_params = combinations_result.iloc[0].to_dict()
            best_params_metadata = {
                "model_name": model.name,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "val_mape": best_params["Val_MAPE"],
                "val_r2": best_params["Val_R2"],
                "parameters": {k: v for k, v in best_params.items() if k in param_names},
                "tuning_duration_seconds": tuning_duration,
                "total_combinations": len(combinations_result),
                "k_folds": k_folds
            }

            # Store best parameters in StateManager
            StateManager.set_page_state(PAGE_NAME, 'best_params', best_params_metadata)

            # Display tuning summary
            st.markdown("### Tuning Summary")
            st.write(f"- **Model**:{model.name}")
            st.write(f"- **Total Combinations Tested**: {len(combinations_result)}")
            st.write(f"- **K-Folds**: {k_folds}")
            st.write(f"- **Tuning Duration**: {tuning_duration:.2f} seconds")
            # st.json(best_params_metadata['parameters'])
            st.metric("Validation MAPE", f"{best_params_metadata['val_mape']:.4f}")
            st.metric("Validation R2", f"{best_params_metadata['val_r2']:.4f}")

    with tab3:
        # Check if tuning has been completed
        best_params_metadata = StateManager.get_page_state(PAGE_NAME, 'best_params')
        if not best_params_metadata:
            st.warning("Please complete hyperparameter tuning in the Hyperparameter Tuning tab first.")
            st.stop()

        # Display best parameters
        st.markdown("### Best Hyperparameters")
        st.markdown("The following parameters were selected based on the tuning results:")
        st.json(best_params_metadata['parameters'])
        # Train model with best parameters
        if st.button("Train Model with Best Parameters"):
            with st.spinner("Training model with best parameters..."):
                start_time = time.time()
                model.fit(X_train, y_train, params=best_params_metadata['parameters'])
                end_time = time.time()
                training_duration = end_time - start_time

                # Evaluate on test set
                y_pred = model.predict(X_test)
                y_pred_original = np.squeeze(ytest_to_initial_scale(
                    y_pred, scalers['min_max_scalerY'], scalers['transformerY'], scalers['shift_value_Y']
                ))
                y_test_original = np.squeeze(ytest_to_initial_scale(
                    y_test, scalers['min_max_scalerY'], scalers['transformerY'], scalers['shift_value_Y']
                ))

                test_mape = mape(y_test_original, y_pred_original)
                test_r2 = r_squared(y_test_original, y_pred_original)

                # Store evaluation results in StateManager
                evaluation_results = {
                    "model_name": model_name,
                    "test_mape": test_mape,
                    "test_r2": test_r2,
                    "training_duration_seconds": training_duration,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }

                # --- CHANGE: Save the trained model object to the session state ---
                StateManager.set_page_state(PAGE_NAME, 'trained_model_object', model)
                StateManager.set_page_state(PAGE_NAME, 'evaluation_results', evaluation_results)

                st.success(f"Model trained successfully in {training_duration:.2f} seconds!")

                # Display evaluation results
                # st.markdown("### Test Set Performance")
                left,right = st.columns([1,1])

                with left:
                    st.metric("Test MAPE", f"{test_mape:.4f}")
                with right:
                    st.metric("Test R2", f"{test_r2:.4f}")

                # Display predictions vs actuals

                results_df = pd.DataFrame({
                    "Actual": y_test_original,
                    "Predicted": y_pred_original
                })

                # Create scatter plot with Plotly
                fig = px.scatter(
                    results_df,
                    x="Actual",
                    y="Predicted",
                    title="Actual vs Predicted Values",
                    labels={"Actual": "Actual Values", "Predicted": "Predicted Values"},
                    width=800,
                    height=600
                )

                # Add reference line (y=x)
                min_val = min(results_df["Actual"].min(), results_df["Predicted"].min())
                max_val = max(results_df["Actual"].max(), results_df["Predicted"].max())
                fig.add_trace(
                    go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode="lines",
                        name="Reference Line (y=x)",
                        line=dict(color="red", dash="dash")
                    )
                )

                # Update layout
                fig.update_layout(
                    showlegend=True,
                    xaxis_title="Actual Values",
                    yaxis_title="Predicted Values",
                    title_x=0.5
                )

                st.plotly_chart(fig, use_container_width=True)
    # == tab 4: save model =========================================================
    with tab4:

        # --- get all necessary artifacts from the state manager ---
        trained_model = StateManager.get_page_state(PAGE_NAME, 'trained_model_object')
        eval_results = StateManager.get_page_state(PAGE_NAME, 'evaluation_results', 'None')
        best_params=StateManager.get_page_state(PAGE_NAME, 'best_params','None')

        if not trained_model:
            st.warning(
                "no trained model found. please train a model in the previous tab first."
            )
        else:
            full_metadata = {
                "model_name": best_params.get("model_name", "unknown"),
                "features_used": features,
                "best_hyperparameters": best_params.get("parameters", {}),
                "tuning_summary": {
                    "validation_mape": best_params.get("val_mape"),
                    "validation_r2": best_params.get("val_r2"),
                    "tuning_duration_seconds": best_params.get("tuning_duration_seconds")
                },
                "final_evaluation_on_test_set": {
                    "test_mape": eval_results.get("test_mape"),
                    "test_r2": eval_results.get("test_r2"),
                    "training_duration_seconds": eval_results.get("training_duration_seconds")
                }
            }

            metadata_bytes = json.dumps(full_metadata, indent=4).encode('utf-8')

            files_for_zip = {
                "model.pkl": pickle.dumps(trained_model),
                "metadata.json": metadata_bytes,
                "scalers.pkl": pickle.dumps(scalers)
            }
            # zip_buffer  = get_zip_file(train_df, test_df, _scalers_and_transformers=scalers)
            # download_link = get_download_link(zip_buffer.getvalue(), "modeling_data.zip")
            
            file_name = f"{best_params.get("model_name", "unknown").replace(' ', '_').lower()}_package.zip"

            zip_buffer = create_zip_buffer(files_for_zip)
            download_link = get_download_link(
                zip_buffer.getvalue(), 
                file_name, 
                "download modeling data"
            )
            st.markdown(download_link, unsafe_allow_html=True)



        #     def download_model(model):
        #         output_model = pickle.dumps(model)
        #         b64 = base64.b64encode(output_model).decode()
        #         href = f'<a href="data:file/output_model;base64,{b64}">Download Trained Model .pkl File</a> '
        #         st.markdown(href, unsafe_allow_html=True)
            
        #     download_model(trained_model)
            # st.download_button(
            #     label="📥 Download Trained Model (.pkl)",
            #     data=pickle.dumps(trained_model),
            #     file_name="trained_model.pkl",
            #     mime="application/octet-stream",
            #     use_container_width=True
            # )
    # # # == TAB 4: Save Model =========================================================
    # with tab4:
    #     st.subheader("Download Your Trained Model")
        
    #     # --- NEW, ROBUST DOWNLOAD LOGIC ---
    #     trained_model = StateManager.get_page_state(PAGE_NAME, 'trained_model_object')
    #     if not trained_model:
    #         st.warning(
    #             "No trained model found. Please go to the **'Train & Evaluate'** tab and click "
    #             "'Train Model with Best Parameters' first."
    #         )
    #         st.stop()

        # # A container to manage the download UI state 
        # download_container = st.container(border=True)
        
        # # Use a state variable to track if the download file is prepared
        # is_ready = StateManager.get_page_state(PAGE_NAME, 'download_ready', False)

        # def set_download_ready():
        #     StateManager.set_page_state(PAGE_NAME, 'download_ready', True)

        # def reset_download_ready():
        #     StateManager.set_page_state(PAGE_NAME, 'download_ready', False)

        # # If the file is not yet prepared, show the "Prepare" button
        # if not is_ready:
        #     download_container.info("The trained model is available. Click the button below to prepare the file for download.")
        #     download_container.button(
        #         "Prepare Model for Download",
        #         on_click=set_download_ready,
        #         use_container_width=True,
        #         type="primary"
        #     )
        # # If the file is prepared, show the "Download" button
        # else:
        #     with st.spinner("Serializing model..."):
        #         model_bytes = pickle.dumps(trained_model)
        #         trained_model_name = trained_model.name
            
        #     download_container.success("Your model file is ready!")
        #     download_container.download_button(
        #         label="📥 Download Model (.pkl)",
        #         data=model_bytes,
        #         file_name=f"{trained_model_name.replace(' ', '_').lower()}_model.pkl",
        #         mime="application/octet-stream",
        #         use_container_width=True,
        #         on_click=reset_download_ready # Reset the state after download to allow another go
        #     )
        #     download_container.info("Clicking the download button will reset this section. You can prepare the file again if needed.")
 