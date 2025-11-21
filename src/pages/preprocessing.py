from datetime import date
import json
import pickle
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split


from src.services.zip_download_helper import create_zip_buffer, get_download_link, get_zip_file
from src.utils.log_scale_transform import log_scale_transform
from src.config.state_manager import StateManager
from settings import PARAM_OPTIONS, SENSORS_CONFIG
from src.utils.scaling_utils import apply_scaling


def show():
    st.title("Preprocessing & Exploration")
    PAGE_NAME = 'preprocessing'
    # --- Create Tabs for Each Step of the Workflow ---
    tab1, tab2, tab3 = st.tabs([
        "**① Load & Inspect Data**",
        "**② Configure Preprocessing**",
        "**③ Run & Download**"
    ])
    # == TAB 1: LOAD & INSPECT DATA =========================================
    with tab1:
        # st.header("Select Your Dataset")
        
        with st.container(border=False):
            upload_own_data = st.toggle(
                "Upload a new matched file",
                help="If ON, upload a new CSV. If OFF, use 'matched_df' from the main session."
            )
            matched_df = None
            if upload_own_data:
                uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'], label_visibility="visible")

                if uploaded_file:
                    matched_df = pd.read_csv(uploaded_file)
                    StateManager.set_page_state(PAGE_NAME, 'is_custom_upload', True)
                    StateManager.set_page_state(PAGE_NAME, 'matched_df', matched_df)
                    StateManager.set_page_state(PAGE_NAME, 'data_loaded', True)

                else:
                    StateManager.set_page_state(PAGE_NAME, 'is_custom_upload', False)
                    StateManager.set_page_state(PAGE_NAME, 'data_loaded', False)
                    StateManager.set_page_state(PAGE_NAME, 'matched_df', None)
            else:
                matched_df= StateManager.get_page_state('matching', 'matched_df')
                # Get the matched_df from the global state
                if matched_df is not None:
                    # matched_df = st.session_state.get('matched_df')
                    StateManager.set_page_state(PAGE_NAME, 'is_custom_upload', False) # False because read from main session
                    StateManager.set_page_state(PAGE_NAME, 'data_loaded', True)
                    StateManager.set_page_state(PAGE_NAME, 'matched_df', matched_df)
                    st.success("✅ Using `matched_df` from the main session.")
                else:
                    st.warning("⚠️ `matched_df` not found in session. Please generate it in the previous step or upload a file.", icon="⚠️")
                    StateManager.set_page_state(PAGE_NAME, 'data_loaded', False)
                    StateManager.set_page_state(PAGE_NAME, 'matched_df', None)

           
        # Once data is loaded and in page state, show the overview
        if StateManager.get_page_state(PAGE_NAME, 'data_loaded', False):
            df_display = StateManager.get_page_state(PAGE_NAME, 'matched_df')
            st.header("Data Overview")
            st.dataframe(df_display.head(10))
            
            exp_col1, exp_col2 = st.columns(2)
            with exp_col1:
                with st.expander("Summary Statistics"):
                    st.write(df_display.describe())
            with exp_col2:
                with st.expander("Column Info (Data Types)"):
                    st.dataframe(df_display.dtypes.astype(str).reset_index().rename(columns={'index': 'Column', 0: 'Dtype'}))

    # == TAB 2: CONFIGURE PREPROCESSING =====================================
    with tab2:
        if not StateManager.get_page_state(PAGE_NAME, 'data_loaded', False):
            st.info("⬅️ Please load a dataset in the first tab to begin configuration.")
            st.stop()

        matched_df = StateManager.get_page_state(PAGE_NAME, 'matched_df')

        col_config1, col_config2 = st.columns(2)

        with col_config1:
            with st.container(border=False):


                is_custom_upload = StateManager.get_page_state(PAGE_NAME, 'is_custom_upload', False)

                if is_custom_upload:

                    target_column = st.selectbox(
                        "Select Target Parameter", 
                        options=PARAM_OPTIONS,
                        index=0,
                        help="Select the water quality parameter you want to model."
                    )

                    if target_column not in matched_df:
                        st.error(
                            f"Error: Column '{target_column}' not found in the dataset for the selected parameter. "
                            f"Please check your data or choose a different parameter."
                        )
                        st.stop()

                    available_sensors = list(SENSORS_CONFIG.keys())
                    selected_sensor = st.selectbox(
                        "Select the Sensor",
                        options=available_sensors,
                        index=0
                    )
                    StateManager.set_page_state(PAGE_NAME, 'sensor', selected_sensor)
                    StateManager.set_page_state(PAGE_NAME, 'target_column', target_column)
                else:
                    # sensor_from_session = st.session_state.get("sensor", "S2")
                    target_column = StateManager.get_page_state('data_loader', 'water_quality_param')
                    sensor_from_session = StateManager.get_page_state('rs_sampling', 'sensor')
                    
                    StateManager.set_page_state(PAGE_NAME, 'sensor', sensor_from_session)
                    StateManager.set_page_state(PAGE_NAME, 'target_column', target_column)
                    # wq_from_session = st.session_state.get("water_quality_param", "turbidity")
                    # wq_from_session = StateManager.get_page_state('data_loader', 'water_quality_param')

                    # st.write(target_column)
                    # StateManager.set_page_state(PAGE_NAME, 'internal_target_col', wq_from_session)


                sensor = StateManager.get_page_state(PAGE_NAME, 'sensor')
                band_columns = SENSORS_CONFIG.get(sensor, {}).get("bands", [])
                selected_features = st.multiselect(
                            f"Select Spectral Bands (Features for {SENSORS_CONFIG[sensor]['name']})", 
                            options=band_columns,
                            default= band_columns[:3]
                        )
                StateManager.set_page_state(PAGE_NAME, 'selected_features', selected_features)
        # st.write(matched_df)
        with col_config2:
            with st.container(border=False):
                options = ["LogScale", "MinMaxScaler"]

                scaler_name = st.selectbox(
                    "Select Scaling Method",
                    options=options,
                    index=options.index(StateManager.get_page_state(PAGE_NAME, 'scaler_name', 'LogScale'))
                )
                StateManager.set_page_state(PAGE_NAME, 'scaler_name', scaler_name)
                
                test_size = st.slider(
                    "Test Set Size", 0.1, 0.9, step=0.1,
                    value=0.3
                )
                StateManager.set_page_state(PAGE_NAME, 'test_size', test_size)
                st.info(f"Train Size: **{(1 - test_size) * 100:.0f}%** | Test Size: **{test_size * 100:.0f}%**")


        with st.container(border=False):
            target_column = StateManager.get_page_state(PAGE_NAME, 'target_column')

            # if is_custom_upload:
            #     target_col = internal_target_col
            # else:
            #     target_col = wq_from_session
            
            # # st.write(matched_df)

            st.subheader(f" Distribution of Target: {target_column}")

            fig = px.histogram(matched_df, x=target_column, nbins=50, color_discrete_sequence=["#009688"])
            st.plotly_chart(fig, use_container_width=True)


    # == TAB 3: RUN & DOWNLOAD ===============================================
    with tab3:
        if not StateManager.get_page_state(PAGE_NAME, 'data_loaded', False):            
            st.info("⬅️ Please load data and configure your settings in the previous tabs first.")
            st.stop()
        
        st.header("Apply Preprocessing and Get Your Datasets")
        
        # Retrieve all settings from page state for review
        # target_col = StateManager.get_page_state(PAGE_NAME, 'internal_target_col')
        target_column = StateManager.get_page_state(PAGE_NAME, 'target_column')
        selected_features = StateManager.get_page_state(PAGE_NAME, 'selected_features')
        scaler_name = StateManager.get_page_state(PAGE_NAME, 'scaler_name')
        test_size = StateManager.get_page_state(PAGE_NAME, 'test_size')
        filtered_df = StateManager.get_page_state(PAGE_NAME, 'filtered_df', matched_df)

        if filtered_df.empty and not selected_features:
            st.error(
                "**Action Required in Tab ②:**"
                "\n- 1. **No Data Remaining:** Your filtering resulted in an empty dataset. Please adjust the filter range."
                "\n- 2. **No Features Selected:** You have not selected any spectral bands. Please choose your features.",
                icon="🚨"
            )
            st.stop()

        elif filtered_df.empty:
                st.warning(
                    "**No Data Remaining:** Your filtering in **Tab ②** resulted in an empty dataset. "
                    "Please adjust the filter range.",
                     
                )
                st.stop()
        elif not selected_features:
            st.warning(
                "**No Features Selected:** You have not selected any spectral bands as features in **Tab ②**. "
                "Please select at least one feature to proceed.",
              
            )
            st.stop()
        
        with st.expander("**Review Your Configuration**", expanded=True):
            st.write(f"- **Target Variable:** `{target_column}`")
            st.write(f"- **Feature Bands ({len(selected_features)}):** `{', '.join(selected_features)}`")
            st.write(f"- **Scaling Method:** `{scaler_name}`")
            st.write(f"- **Test/Train Split:** `{test_size*100:.0f}% / {(1-test_size)*100:.0f}%`")
            st.write(f"- **Data Points After Filtering:** `{len(filtered_df)}`")

        if st.button("**Generate Processed Datasets**", use_container_width=True, type="primary"):
            # Get metadata
            site_number = filtered_df['site_no'].values if 'site_no' in filtered_df.columns else np.zeros(
                len(filtered_df))
            dates = filtered_df['ImageAquisition_time'].values if 'ImageAquisition_time' in filtered_df.columns else np.zeros(
                len(filtered_df))
            satellite = filtered_df['spacecraft_name'].values if 'spacecraft_name' in filtered_df.columns else np.zeros(
                len(filtered_df))


            x = filtered_df[selected_features].values
            y = filtered_df[target_column].values
            
            x_rescaled, y_rescaled, scalers, extras = apply_scaling(x, y, scaler_name, site_number, dates, satellite)
                
            
            indices = np.arange(len(x_rescaled))
            X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
                x_rescaled,
                y_rescaled,
                indices,
                test_size=test_size,
                random_state=42
            )

            
            train_df = pd.DataFrame(X_train, columns=selected_features)
            train_df["target"] = y_train
            
            test_df = pd.DataFrame(X_test, columns=selected_features)
            test_df["target"] = y_test
            

            # Save final dataframes to page state
            StateManager.set_page_state(PAGE_NAME, 'train_df', train_df)
            StateManager.set_page_state(PAGE_NAME, 'test_df', test_df)
            StateManager.set_page_state(PAGE_NAME, 'train_idx', train_idx)
            StateManager.set_page_state(PAGE_NAME, 'test_idx', test_idx)
            StateManager.set_page_state(PAGE_NAME, 'scalers', scalers)

            sensor_name = StateManager.get_page_state(PAGE_NAME, 'sensor', 'sensor')
            # selected_param_display = StateManager.get_page_state(PAGE_NAME,'selected_param_display')
            test_size_str = f"{int(test_size * 100)}testsize"
            today_str = date.today().strftime('%Y%m%d')

            # example: 250705_s2_turbidity_30testsize_modeling_data.zip
            param_safe = str(target_column).replace(" ", "_").lower()
            filename = f"{today_str}_{sensor_name}_{param_safe}_{test_size_str}_modeling_data.zip"
            # filename = f"{today_str}_{sensor_name}_{target_column.replace(" ", "_")}_{test_size_str}_modeling_data.zip"

            metadata = {
                "sensor":sensor_name,
                "target_variable": target_column,
                "selected_features": selected_features,
                "scaling_method": scaler_name,
                "test_train_split": f"{test_size*100:.0f}% test / {(1-test_size)*100:.0f}% train",
                "data_points_after_filtering": len(filtered_df)
            }
            # convert dictionary to a formatted json string, then to bytes
            metadata_bytes = json.dumps(metadata, indent=4).encode("utf-8")

            files_for_zip = {
                "train_data.csv": train_df.to_csv(index=False).encode("utf-8"),
                "test_data.csv": test_df.to_csv(index=False).encode("utf-8"),
                "scalers.pkl": pickle.dumps(scalers),
                "metadata.json": metadata_bytes  # add the new metadata file
            }
            # zip_buffer  = get_zip_file(train_df, test_df, _scalers_and_transformers=scalers)
            # download_link = get_download_link(zip_buffer.getvalue(), "modeling_data.zip")

            zip_buffer = create_zip_buffer(files_for_zip)
            download_link = get_download_link(
                zip_buffer.getvalue(), 
                filename, 
                "download modeling data"
            )
            st.markdown(download_link, unsafe_allow_html=True)
