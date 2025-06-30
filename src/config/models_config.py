# File Location: src/config/models_config.py

import streamlit as st

MODELS_CONFIG = {
    "Random Forest (RF)": {
        "param_definitions": {
            'n_estimators': {
                'label': 'Number of Trees',
                'ui_widget': 'text_list',
                'type': int,
                'help': "Enter integer values, separated by commas (e.g., 50, 100, 200)."
            },
            'max_depth': {
                'label': 'Max Depth of Tree',
                'ui_widget': 'text_list',
                'type': int,
                'help': "Enter integer values (e.g., 5, 10, 20). Use 'None' for no limit."
            },
            'criterion': {
                'label': 'Criterion',
                'ui_widget': 'multiselect',
                'options': ['squared_error', 'absolute_error', 'friedman_mse'],
                'type': str,
                'help': "Select one or more functions to measure the quality of a split."
            },
            'min_samples_split': {
                'label': 'Min Samples Split',
                'ui_widget': 'text_list',
                'type': int,
                'help': "Enter integer values (e.g., 2, 5, 10)."
            },
            'min_samples_leaf': {
                'label': 'Min Samples Leaf',
                'ui_widget': 'text_list',
                'type': int,
                'help': "Enter integer values (e.g., 1, 2, 4)."
            },
            'max_features': {
                'label': 'Max Features',
                'ui_widget': 'multiselect',
                'options': ['auto', 'sqrt', 'log2', 0.5, 1.0],
                'type': str,
                'help': "Select options for the number of features to consider at each split."
            },
            'bootstrap': {
                'label': 'Bootstrap',
                'ui_widget': 'multiselect',
                'options': [True, False],
                'type': bool,
                'help': "Enable or disable bootstrap sampling."
            }
        },
        "default_tuning_grid": {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'criterion': ['squared_error'],
            'min_samples_split': [10],
            'min_samples_leaf': [4],
            'max_features': ['sqrt'],
            'bootstrap': [True]
        }
    },
    "Artificial Neural Network (ANN)": {
        "param_definitions": {

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
        },
        "default_tuning_grid": {
            'act_function_hidden_layers': ['relu'],
            'optimizer_func': ['adam'],
            'num_of_epochs': [100],
            'act_function_output_layers': ['linear'],
            'loss_func': ['mean_squared_error'],
            'batch_size': [32],
            'lr': [1e-3]
        }
    }
}


def parse_string_to_list(input_str: str, target_type, input_shape=None):
    """
    Helper function to parse a comma-separated or semicolon-separated string into a list of a specific type.
    """
    if not input_str:
        return []

    # Handle dynamic input shape for ANN
    if 'INPUT_SHAPE' in input_str:
        if input_shape is None:
            st.error(
                "Input shape is required for dynamic layer parsing but was not provided.")
            return []
        input_str = input_str.replace('INPUT_SHAPE', str(input_shape))

    # Special handling for 'None' string
    if 'none' in input_str.lower():
        # This handles cases like "10, 20, None"
        def convert(s):
            s = s.strip().lower()
            return None if s == 'none' else target_type(s)
        return [convert(s) for s in input_str.split(',')]

    # Special handling for tuples like (50,50); (100,) used for ANN layers
    if target_type == 'dynamic_tuple':
        try:
            # Safely evaluate tuples from a string
            return [eval(s.strip()) for s in input_str.split(';')]
        except Exception as e:
            st.error(f"Error parsing tuple string: {e}")
            return []

    # Standard comma-separated parsing for numbers and strings
    try:
        return [target_type(s.strip()) for s in input_str.split(',')]
    except (ValueError, TypeError):
        return []
