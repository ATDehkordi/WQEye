# src/components/ml_hyperparam_form_component.py

import streamlit as st
import pandas as pd
from src.models.registry import ModelRegistry
from src.config.models_config import parse_string_to_list
from src.config.state_manager import StateManager



def hyperparameter_form_ui(model_name: str, page_name: str, default_tuning: bool = True, X_train: pd.DataFrame = None) -> tuple[dict, int, bool]:
    """Display UI for hyperparameter tuning."""

    param_grid = {}

    registry = ModelRegistry()
    model = registry.get_model(model_name)
    if not model:
        st.error(f"Model {model_name} not found.")
        st.stop()


    with st.form(key=f"{page_name}_tuning_form"):

        # if default_tuning:
        #     with st.expander("View Default Parameter Grid"):
        #         default_grid = model_config.get("default_tuning_grid", {})
        #         st.json(default_grid)
        #     param_grid = default_grid

        # else:
        param_definitions = model.get_param_definitions()
        default_grid = model.get_default_param_grid(X_train)

        custom_grid = {}
        col1, col2 = st.columns(2)

        for i, (param_key, config) in enumerate(param_definitions.items()):
            target_col = col1 if i % 2 == 0 else col2
            default_value = default_grid.get(param_key, [])
            default_str = ', '.join(map(str, default_value)) if isinstance(default_value, list) else str(default_value)

            with target_col:
                widget_type = config.get('ui_widget', 'text_list')

                if widget_type == 'multiselect':
                    selected_options = st.multiselect(
                        label=config['label'],
                        options=config['options'],
                        default=default_value if default_tuning else [],
                        help=config['help'],
                        key=f"{model_name}_{param_key}"
                    )
                    if selected_options:
                        custom_grid[param_key] = selected_options

                elif widget_type == 'text_list':
                    input_str = st.text_input(
                        label=config['label'],
                        value=default_str if default_tuning else '',
                        help=config['help'],
                        placeholder=config.get('placeholder', ''),
                        key=f"{model_name}_{param_key}"
                    )
                    if input_str:
                        parsed = parse_string_to_list(input_str, config.get('type'))
                        if parsed:
                            custom_grid[param_key] = parsed

        param_grid = custom_grid
        left,right = st.columns(2)

        with left:
            k_folds = st.number_input(
                "Number of Folds for Cross-Validation",
                min_value=2,
                max_value=20,  
                value=2,
                step=1,
                key="k_folds",
                help="Enter the number of folds for cross-validation (must be at least 2)."
            )            

        
        submitted = st.form_submit_button("Tune")
        return param_grid, k_folds, submitted