# src/router.py
import streamlit as st
from streamlit_option_menu import option_menu

from src.pages import data_loader, rs_sampling, matching, preprocessing, machine_learning, evaluation
from src.config.state_manager import StateManager

def run():
    """Run the application router."""
    StateManager.initialize()

    with st.sidebar:
        selected = option_menu(
            menu_title="WQEye",
            options=["Data Loader", 'RS Sampling', "Matching", "Preprocessing", "Machine Learning","Export"],
            icons=[
                "file-earmark-arrow-up",
                "binoculars",
                "link-45deg",
                "gear-wide-connected",
                "cpu-fill",
                "clipboard-check"
            ],
            menu_icon="cast",
            default_index=0,
            styles={
                "icon": {"color": "#4a4a4a", "font-size": "18px"},
                "nav-link": {
                    "font-size": "16px",
                    "text-align": "left",
                    "margin": "5px",
                    "--hover-color": "#2b313e",
                },
                "nav-link-selected": {"background-color": "#a9a9a9", "color": "white"},
            },
        )
        # Save current page
        StateManager.set_page_state("router", "current_page", selected)


    if selected == "Data Loader":
        data_loader.show()
    elif selected == 'RS Sampling':
        rs_sampling.show()
    elif selected == "Matching":
        matching.show()
    elif selected == "Preprocessing":
        preprocessing.show()
    elif selected == "Machine Learning":
        machine_learning.show()
    elif selected == "Export":
        evaluation.show()
