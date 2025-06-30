# src/components/parameter_selector_component.py
import streamlit as st
from datetime import date
from settings import PARAM_OPTIONS
from src.config.state_manager import StateManager

def parameter_selector():
    """Display and manage Water Quality Parameter and Date Range selection."""

    col1, col2 = st.columns([1, 1])

    with col1:
        # Step 1: Get the currently stored parameter value from the state manager.
        # If nothing is stored, default to the first option in the list.
        default_value = PARAM_OPTIONS[0]
        stored_param = StateManager.get_page_state('init_data', 'water_quality_param', default_value)

        # Step 2: Find the index of the stored parameter.
        # This is crucial because the 'index' parameter of selectbox needs an integer.
        try:
            default_index = PARAM_OPTIONS.index(stored_param)
        except ValueError:
            # If the stored value is somehow invalid (e.g., from an older version),
            # safely default to the first option.
            default_index = 0

        # Step 3: Create the selectbox using the calculated index.
        # We remove the 'key' parameter to avoid conflicts with our manual state management.
        water_quality_param = st.selectbox(
            "Water Quality Parameter:",
            options=PARAM_OPTIONS,
            index=default_index
        )

        # Step 4: Save the user's current selection back to the state manager.
        # This ensures that if the user changes the value, it gets saved for the next run.
        StateManager.set_page_state('init_data', 'water_quality_param', water_quality_param)

    # with col2:
    #     date_range = st.date_input(
    #         "Date Range:",
    #         min_value=date(2000, 1, 1),
    #         max_value=date.today(),
    #         key="date_range"
    #     )
        # StateManager.set_page_state("parameter_selector", "date_range", date_range)

    with col2:
        # Add a "Reset to Default" button
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)

        if st.button("Reset", key="reset_parameters"):
            StateManager.reset()  # Reset all session state variables to their initialized values
            # st.write("initialized" not in st.session_state)

            # st.success("All parameters and states reset to default values!")

    return water_quality_param