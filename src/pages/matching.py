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

    if 'rs_collections_by_site' not in st.session_state:
        st.warning(
            "⚠️ rs_collections_by_site data not found.\n\n"
            "To use this page, please first use rs_collections_by_site "
        )
        return


    threshold = st.number_input(
        "Enter the matching threshold (in seconds)", 
        min_value=0, 
        max_value=86400,  # 1 day
        value=3600,       # default = 20 minutes
        step=60,
        key='matching_threshold'
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        run_clicked = st.button("Run Matching")

    if run_clicked:
        session_dict = dict(st.session_state)
        parameter_code = PARAM_CODE_MAP[StateManager.get_page_state('init_data', 'water_quality_param')]

        df_all = []

        progress_bar = st.progress(0)
        status_text = st.empty()
        site_keys = list(session_dict['rs_collections_by_site'].keys())
        total_stations = len(site_keys)

        for i, site_no in enumerate(site_keys):
            usgs_data = st.session_state['insitu_data']
            usgs_data_site = usgs_data[usgs_data['site_no'] == site_no]

            rs_data = session_dict['rs_collections_by_site'][site_no]['sampled_values_at_station'].getInfo()['features']

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
            st.session_state["matched_df"] = final_df
            st.success("✅ Matching completed for all sites.")
        else:
            st.warning("⚠️ No data was matched.")
            st.session_state["matched_df"] = None


    matched_df_ready = (
        "matched_df" in st.session_state 
        and st.session_state["matched_df"] is not None 
        and not st.session_state["matched_df"].empty
    )

    if matched_df_ready:
        csv_buffer = StringIO()
        st.session_state["matched_df"].to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
    else:
        csv_data = ""

    with col2:
        st.download_button(
            label="📥 Download Matched Data",
            data=csv_data,
            file_name="matched_data.csv",
            mime="text/csv",
            disabled=not matched_df_ready,
        )
