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

@st.cache_resource(show_spinner=False)
def create_zip_buffer(files_to_zip: dict[str, bytes]) -> io.BytesIO:
    """
    creates a generic zip file in memory from a dictionary of files.
    
    args:
        files_to_zip (dict): a dictionary where keys are the desired
                             filenames (str) and values are the file
                             content (bytes).
    returns:
        io.bytesio: in-memory zip file buffer.
    """
    zip_buffer = io.BytesIO() 
    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        # iterate over the dictionary and write each file
        for filename, content_bytes in files_to_zip.items():
            zf.writestr(filename, content_bytes)
            
    zip_buffer.seek(0)
    return zip_buffer

# def get_download_link(zip_bytes: bytes, filename: str = "modeling_data.zip") -> str:
def get_download_link(zip_bytes: bytes, filename: str, link_text: str) -> str:

    """
    Creates a base64-encoded HTML anchor tag for downloading a ZIP file.

    Args:
        zip_bytes (bytes): Content of the ZIP file
        filename (str): Name of the downloadable file

    Returns:
        str: HTML <a> tag as download link
    """
    b64 = base64.b64encode(zip_bytes).decode()
    # href = f'<a href="data:application/zip;base64,{b64}" download="{filename}">Download File</a>'
    href = f'<a href="data:application/zip;base64,{b64}" download="{filename}">{link_text}</a>'

    return href