# src/config/state_manager.py
import streamlit as st
import pandas as pd
from typing import Optional, List, Tuple
from datetime import date
from src.config.models_config import MODELS_CONFIG


class StateManager:
    """Centralized state management for the WQEye application."""

    @staticmethod
    def initialize():
        """Initialize default session state variables."""
        # st.session_state.setdefault("water_quality_param", "turbidity")
        st.session_state.setdefault("insitu_data", None)
        st.session_state.setdefault("region", None)
        st.session_state.setdefault("station_list", None)
        st.session_state.setdefault("selected_stations_detail", None)
        st.session_state.setdefault("date_range", (date(2013, 1, 1), date.today()))
        st.session_state.setdefault("page_states", {})
        st.session_state.setdefault("usgs_download_data", None)
        st.session_state.setdefault("usgs_zip_buffer", b"")
        st.session_state.setdefault("usgs_download_ready", False)
        st.session_state.setdefault("insitu_data_source", None)
        st.session_state.setdefault("sensor", None)

    @staticmethod
    def get_model_config(model_name: str) -> Optional[dict]:
        """
        Retrieves the configuration for a specific model from the central config file.
        """
        return MODELS_CONFIG.get(model_name)
    
    @staticmethod
    def set_insitu_data(df: Optional[pd.DataFrame], water_quality_param: str):
        """Set insitu_data with validation based on water_quality_param."""
        if df is not None:
            # Define required columns
            required_columns = ["site_no", "datetime_utc", water_quality_param]
            if not all(col in df.columns for col in required_columns):
                st.error(f"Invalid data format. Required columns: site_no, datetime_utc, {water_quality_param}")
                return False
            # Validate datetime_utc column
            try:
                df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])
            except Exception as e:
                st.error(f"Invalid format for datetime_utc column: {str(e)}")
                return False
            # Remove duplicates
            df = df.drop_duplicates()
            st.session_state["insitu_data"] = df
            st.success("In-situ data saved successfully!")
            return True
        return False

    @staticmethod
    def get_insitu_data() -> Optional[pd.DataFrame]:
        """Get insitu_data from session state."""
        return st.session_state.get("insitu_data")

    @staticmethod
    def set_page_state(page: str, key: str, value):
        """Store page-specific state."""
        if page not in st.session_state["page_states"]:
            st.session_state["page_states"][page] = {}
        st.session_state["page_states"][page][key] = value

    @staticmethod
    def get_page_state(page: str, key: str, default=None):
        """Retrieve page-specific state."""
        return st.session_state["page_states"].get(page, {}).get(key, default)

    @staticmethod
    def clear_page_state(page: str):
        """Clear state for a specific page."""
        if page in st.session_state["page_states"]:
            del st.session_state["page_states"][page]
    
    @staticmethod
    def reset():
        """Reset all session state variables to their initial state."""
        # Preserve Streamlit internal keys (if any)
        protected_keys = [key for key in st.session_state.keys() if key.startswith("_")]

        # Clear all keys except protected ones
        for key in list(st.session_state.keys()):
            if key not in protected_keys:
                del st.session_state[key]
        
        # Reinitialize session state
        StateManager.initialize()