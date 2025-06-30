# zip_download_helper.py
import io
import pickle
import zipfile
import streamlit as st
import base64
import pandas as pd

@st.cache_resource(show_spinner=False)
def get_zip_file(train_df: pd.DataFrame, test_df: pd.DataFrame, _scalers_and_transformers: dict) -> io.BytesIO:
    """
    creates a ZIP file in memory containing:
    - train_data.csv
    - test_data.csv
    - scalers.pkl (pickled Python objects)

    Returns:
        BytesIO: in-memory ZIP file buffer
    """
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("train_data.csv", train_df.to_csv(index=False).encode("utf-8"))
        zf.writestr("test_data.csv", test_df.to_csv(index=False).encode("utf-8"))
        zf.writestr("scalers.pkl", pickle.dumps(_scalers_and_transformers))
    zip_buffer.seek(0)
    return zip_buffer


def get_download_link(zip_bytes: bytes, filename: str = "modeling_data.zip") -> str:
    """
    Creates a base64-encoded HTML anchor tag for downloading a ZIP file.

    Args:
        zip_bytes (bytes): Content of the ZIP file
        filename (str): Name of the downloadable file

    Returns:
        str: HTML <a> tag as download link
    """
    b64 = base64.b64encode(zip_bytes).decode()
    href = f'<a href="data:application/zip;base64,{b64}" download="{filename}">Download File</a>'
    return href