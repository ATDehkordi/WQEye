import pandas as pd
import streamlit as st
import os
import ee

@st.cache_data
def read_file_from_path(file_path: str):
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".csv":
        return pd.read_csv(file_path)
    elif ext in [".xls", ".xlsx"]:
        return pd.read_excel(file_path)
    else:
        return None


def read_file(file_input):
    try:
        if isinstance(file_input, str):
            return read_file_from_path(file_input)

        elif hasattr(file_input, 'name'):
            ext = os.path.splitext(file_input.name)[-1].lower()
            if ext == ".csv":
                return pd.read_csv(file_input)
            elif ext in [".xls", ".xlsx"]:
                return pd.read_excel(file_input)
            else:
                st.error("Unsupported file type.")
                return None
        else:
            return None

    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None


def drawn_to_ee_polygon(drawn_coordinates):
    coords = drawn_coordinates["geometry"]["coordinates"][0]
    return ee.Geometry.Polygon(coords)
