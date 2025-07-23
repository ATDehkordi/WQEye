#  src/pages/matching.py
from io import StringIO
import pandas as pd
import streamlit as st

from src.config.state_manager import StateManager
from settings import PARAM_CODE_MAP
from src.services.usgs_cache_services import cache_matchup_usgs_and_rs_data
from src.apps.RemoteSensingDataPreparation import RemoteSensingDataPreparation
from src.utils.matchup_func import match_up_usgs_and_rs_data

def show():
    st.title("Matching")
    page_name='matching'
    rs_collection = StateManager.get_page_state('rs_sampling', 'rs_collections_by_site')

    # if 'rs_collections_by_site' not in st.session_state:
    if not rs_collection:
        st.warning(
            "⚠️ rs_collections_by_site data not found.\n\n"
            "To use this page, please first use rs_sampling  "
        )
        return


    threshold = st.number_input(
        "Enter the matching threshold (in seconds)", 
        min_value=0, 
        max_value=86400,  # 1 day
        value=StateManager.get_page_state(page_name, 'matching_threshold',3600),       # default = 20 minutes
        step=60,
    )
    StateManager.set_page_state(page_name, 'matching_threshold', threshold)

    col1, col2 = st.columns([1, 1])

    with col1:
        run_clicked = st.button("Run Matching")

    if run_clicked:
        session_dict = dict(st.session_state)
        parameter_code = PARAM_CODE_MAP[StateManager.get_page_state('data_loader', 'water_quality_param')]

        df_all = []

        progress_bar = st.progress(0)
        status_text = st.empty()
        site_keys = list(rs_collection.keys())
        total_stations = len(site_keys)

        for i, site_no in enumerate(site_keys):
            usgs_data = st.session_state['insitu_data']
            usgs_data_site = usgs_data[usgs_data['site_no'] == site_no]

            rs_data = rs_collection[site_no]['sampled_values_at_station'].getInfo()['features']

            df = cache_matchup_usgs_and_rs_data(
                _rs_data=rs_data,
                usgs_data_site=usgs_data_site,
                parameter_code=parameter_code,
                threshold=threshold
            )

            df_all.append(df)

            progress = int((i + 1) / total_stations * 100)
            progress_bar.progress(progress)
            status_text.text(f"Processing station {i + 1}/{total_stations} (site_no: {site_no})")

        if df_all:
            final_df = pd.concat(df_all, ignore_index=True)
            # st.session_state["matched_df"] = final_df
            StateManager.set_page_state(page_name, 'matched_df', final_df)
            st.success("✅ Matching completed for all sites.")
        else:
            st.warning("⚠️ No data was matched.")
            # st.session_state["matched_df"] = None
            StateManager.set_page_state(page_name, 'matched_df', None)

    matched_df = StateManager.get_page_state(page_name, "matched_df")
    matched_df_ready = matched_df is not None and not matched_df.empty

    # matched_df_ready = (
    #     "matched_df" in st.session_state 
    #     and st.session_state["matched_df"] is not None 
    #     and not st.session_state["matched_df"].empty
    # )
    filename = 'init_df.csv'
    if matched_df_ready:

        sensor_name = StateManager.get_page_state('rs_sampling', 'sensor', 'unknown_sensor')
        wq_param = StateManager.get_page_state('data_loader', 'water_quality_param', 'unknown_param')
        threshold = StateManager.get_page_state(page_name, 'matching_threshold', 'na')

        # example: matched_data_landsat-8_turbidity_3600s.csv
        filename = f"matched_data_{sensor_name}_{wq_param}_{threshold}s.csv"


        csv_buffer = StringIO()
        # st.session_state["matched_df"].to_csv(csv_buffer, index=False)
        matched_df.to_csv(csv_buffer, index=False)

        csv_data = csv_buffer.getvalue()
    else:
        csv_data = ""

    with col2:
        st.download_button(
            label="📥 Download Matched Data",
            data=csv_data,
            file_name=filename,
            mime="text/csv",
            disabled=not matched_df_ready,
        )
