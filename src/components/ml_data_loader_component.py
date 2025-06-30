# src/components/ml_data_loader_component.py

import streamlit as st
from src.utils.zip_checker import check_zip_contents
from src.config.state_manager import StateManager

def load_ml_data(page_name: str):

    ml_toggle = st.toggle("Upload ML Data", key='ml_toggle')

    if ml_toggle:
        zip_file = st.file_uploader(
            "Upload a ZIP file",
            type=['zip'],
            key="zip_upload",
            help="Please upload a ZIP file that contains the following files:\n"
                "- `train_data.csv`: Training dataset\n"
                "- `test_data.csv`: Testing dataset\n"
                "- `scalers.pkl`: Pickled scalers and transformers used during preprocessing\n\n"
                "Make sure the filenames match exactly. The app will check their presence automatically."
        )
        if zip_file:
            is_valid, contents, error = check_zip_contents(zip_file)
            if is_valid:
                train_df = contents["train_df"]
                test_df = contents["test_df"]
                scalers = contents["scalers"]
                st.success("ZIP file loaded successfully!")
                return train_df, test_df, scalers
            else:
                st.error(error)
                return None, None, None
        else:
            return None, None, None

    else:

        if StateManager.get_page_state('preprocessing', 'train_df', None) is None:
            st.warning("No data found. Please upload data.")
            return None, None, None

        else:
            st.success("Using data from Main Session")
            train_df = StateManager.get_page_state('preprocessing', 'train_df', None)
            test_df = StateManager.get_page_state('preprocessing', 'test_df', None)
            scalers = StateManager.get_page_state('preprocessing', 'scalers', None)
            return train_df, test_df, scalers
