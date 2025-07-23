# src/pages/data_loader.py
import streamlit as st
from datetime import date
from settings import PARAM_OPTIONS,ENABLE_USGS_FETCHER
from src.config.state_manager import StateManager

# from src.components.usgs_data_fetcher_component import usgs_data_fetcher
from src.components.insitu_data_uploader_component import render_insitu_uploader
from src.components.parameter_selector_component import parameter_selector


def show():
    """Display the Data Loader page."""
    st.subheader("Data Loader")
    page_name='data_loader'

    # Get parameters from selector component
    water_quality_param = parameter_selector(page_name)

    # Create tabs for data input methods
    tab_titles = ["Manual Data Import"]
    if ENABLE_USGS_FETCHER:
        tab_titles.append("USGS Data Fetcher")

    tabs = st.tabs(tab_titles)

    with tabs[0]:
        render_insitu_uploader(water_quality_param, page_name)
    
    # --- conditionally render usgs tab ---
    if ENABLE_USGS_FETCHER:
        with tabs[1]:
            # your logic for the usgs fetcher can live here safely
            st.info("USGS data fetcher is currently under development.")
            # usgs_data_fetcher(...)