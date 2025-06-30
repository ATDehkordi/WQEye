import streamlit as st
from src.models.registry import ModelRegistry
from src.config.models_config import MODELS_CONFIG
from src.config.state_manager import StateManager

def model_selection_ui(X_train, X_test, features, train_df, page_name:str='machine_learning'):
    registry = ModelRegistry()

    col1, col2 = st.columns([1, 1.3])
    with col1:
        model_names = registry.get_model_names()
        if not model_names:
            st.error("No models found in registry.")
            return
       
        selected_model = st.radio(
            "Choose a model to train:",
            options=model_names,
            key="selected_model",
            index=0            
        )
        # Save selected model code via StateManager
        StateManager.set_page_state(page_name, 'selected_model', selected_model)
    with col2:
        with st.container(border=True):
            st.metric("Training Samples", f"{len(X_train):,}")
            st.metric("Test Samples", f"{len(X_test):,}")
            st.metric("Number of Features", f"{len(features)}")
            with st.expander("Show a preview of the training data"):
                st.dataframe(train_df.head())