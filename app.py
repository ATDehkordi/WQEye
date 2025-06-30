# app.py
import streamlit as st
from settings import PARAM_OPTIONS, initialize_gee
from src.config.state_manager import StateManager

# Initialize page configuration
st.set_page_config(page_title="WQEye", layout="wide")

# Initialize GEE and session state
initialize_gee()
# StateManager.initialize()

# Load CSS globally
for css_file in ["assets/css/layout.css", "assets/css/messages.css"]:
    with open(css_file, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

from src import router
# Run router
router.run()