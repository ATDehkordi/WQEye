# src/router.py
import streamlit as st
from streamlit_option_menu import option_menu

from src.pages import data_loader, rs_sampling, matching, preprocessing, machine_learning, export
from src.config.state_manager import StateManager


PAGE_REGISTRY = {
    "Data Loader": {
        "module": data_loader,
        "icon": "file-earmark-arrow-up",
    },
    "RS Sampling": {
        "module": rs_sampling,
        "icon": "binoculars",
    },
    "Matching": {
        "module": matching,
        "icon": "link-45deg",
    },
    "Preprocessing": {
        "module": preprocessing,
        "icon": "gear-wide-connected",
    },
    "Machine Learning": {
        "module": machine_learning,
        "icon": "cpu-fill",
    },
    "Export": {
        "module": export,
        "icon": "clipboard-check",
    }
}


def run():
    """run the application router."""
    StateManager.initialize()

    with st.sidebar:
        # 2. build menu options and icons dynamically from the registry
        page_titles = list(PAGE_REGISTRY.keys())
        page_icons = [details["icon"] for details in PAGE_REGISTRY.values()]

        selected = option_menu(
            menu_title="WQEye",
            options=page_titles,
            icons=page_icons,
            menu_icon="cast",
            default_index=0,
            # styles can be kept as they are
        )
    
    StateManager.set_page_state("router", "current_page", selected)
    # st.json(st.session_state)
    # 3. get the selected module from the registry and run it
    page_to_show = PAGE_REGISTRY[selected]["module"]
    page_to_show.show()


# def run():
#     """Run the application router."""
#     StateManager.initialize()

#     with st.sidebar:
#         selected = option_menu(
#             menu_title="WQEye",
#             options=["Data Loader", 'RS Sampling', "Matching", "Preprocessing", "Machine Learning","Export"],
#             icons=[
#                 "file-earmark-arrow-up",
#                 "binoculars",
#                 "link-45deg",
#                 "gear-wide-connected",
#                 "cpu-fill",
#                 "clipboard-check"
#             ],
#             menu_icon="cast",
#             default_index=0,
#             styles={
#                 "icon": {"color": "#4a4a4a", "font-size": "18px"},
#                 "nav-link": {
#                     "font-size": "16px",
#                     "text-align": "left",
#                     "margin": "5px",
#                     "--hover-color": "#2b313e",
#                 },
#                 "nav-link-selected": {"background-color": "#a9a9a9", "color": "white"},
#             },
#         )
#         # Save current page
#         StateManager.set_page_state("router", "current_page", selected)


#     if selected == "Data Loader":
#         data_loader.show()
#     elif selected == 'RS Sampling':
#         rs_sampling.show()
#     elif selected == "Matching":
#         matching.show()
#     elif selected == "Preprocessing":
#         preprocessing.show()
#     elif selected == "Machine Learning":
#         machine_learning.show()
#     elif selected == "Export":
#         export.show()
