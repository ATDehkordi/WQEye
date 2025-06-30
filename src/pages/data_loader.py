# src/pages/data_loader.py
import streamlit as st
from datetime import date
from settings import PARAM_OPTIONS
from src.config.state_manager import StateManager

# from src.components.usgs_data_fetcher_component import usgs_data_fetcher
from src.components.upload_and_merge_component import upload_and_merge_insitudata
from src.components.parameter_selector_component import parameter_selector


def show():
    """Display the Data Loader page."""
    st.subheader("Data Loader")

    # Get parameters from selector component
    water_quality_param = parameter_selector()

    # Create tabs for data input methods
    # tabs = st.tabs(["Manual Data Import", "USGS Data Fetcher"])
    tabs = st.tabs(["Manual Data Import"])

    with tabs[0]:
        upload_and_merge_insitudata(water_quality_param)
    # with tabs[1]:
    #     usgs_data_fetcher(water_quality_param, date_range)