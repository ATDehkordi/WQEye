# src/pages/rs_sampling.py
import streamlit as st
import pandas as pd

from settings import SENSORS_CONFIG
from src.components.station_editor_component import station_editor
from src.services.usgs_cache_services import cache_rs_dataprepration
from src.config.state_manager import StateManager


def show():
    PAGE_NAME = 'rs_sampling'

    # Set the page title with custom styling
    st.markdown("<h1 style='text-align: center; color: #4B5EAA;'>Remote Sensing Data Overview</h1>", unsafe_allow_html=True)
    # st.session_state['sensor'] = None if 'sensor' not in st.session_state else st.session_state['sensor']

    # Check if in-situ data exists
    insitu_data = StateManager.get_insitu_data()
    if insitu_data is None or insitu_data.empty:
        st.warning(
            "⚠️ In-situ data not found.\n\n"
            "To use this page, please upload your in-situ data manually "
            "or download using the USGS data fetcher in the Data Loader page."
        )
        return

    # Retrieve the in-situ data from session state
    # temp_df = st.session_state['insitu_data']
    # --- Display Data Summary Expander ---
    temp_df = insitu_data.copy()
    temp_df['datetime_utc'] = pd.to_datetime(temp_df['datetime_utc'], errors='coerce')
    min_date = temp_df['datetime_utc'].min().strftime('%Y-%m-%d')
    max_date = temp_df['datetime_utc'].max().strftime('%Y-%m-%d')
    StateManager.set_page_state(PAGE_NAME, "date_range", (min_date, max_date))
    
    unique_site_nos = temp_df['site_no'].nunique()

    # Expandable summary section with styled content
    with st.expander("Summary of Data", expanded=False):
        st.markdown("<h3 style='color: #2E4057;'>Data Overview</h3>", unsafe_allow_html=True)
        st.write(f"First in-situ measurement: **{min_date}**")
        st.write(f"Last in-situ measurement: **{max_date}**")
        st.write(f"Number of unique sites: {unique_site_nos}")
    
    st.divider()

    # --- User Controls Section ---
    col1, col2, col3 = st.columns([2.5, 1.5, 1.5])

    with col1:
         # --- REPLACEMENT FOR st.pills ---
        sensor_options = list(SENSORS_CONFIG.keys())
        
        # Step 1: Get the previously selected sensor from state, default to the first option
        stored_sensor = StateManager.get_page_state(PAGE_NAME, "sensor", sensor_options[0])
        
        # Step 2: Find the index of the stored sensor for the radio button
        try:
            default_index = sensor_options.index(stored_sensor)
        except ValueError:
            default_index = 0 # Fallback to first item if not found

        # Step 3: Use st.radio which allows setting a default index
        selected_sensor = st.radio(
            label="Select Sensor",
            options=sensor_options,
            format_func=lambda key: f"{SENSORS_CONFIG[key]['icon']} {SENSORS_CONFIG[key]['name']}",
            index=default_index,
            horizontal=True, # This makes it look like pills
        )
        
        # Step 4: Save the current selection back to the state
        StateManager.set_page_state(PAGE_NAME, "sensor", selected_sensor)

    with col2:
        # Global cloud coverage percentage input with styled slider
        cloud_coverage = st.slider(
            label="Global Cloud Coverage (%)",
            min_value=0,
            max_value=100,
            value=20,
            step=1,
            format="%d%%",
            help="Specify the maximum allowed cloud coverage percentage for the satellite images."
        )

    with col3:
        # Buffer distance input with styled number input
        buffer_distance = st.number_input(
            label="Buffer Distance (m)",
            min_value=0,
            max_value=10000,
            value=250,
            step=50,
            help="Specify the buffer distance in meters around the point of interest."
        )
    # Store control values in session state
    StateManager.set_page_state(PAGE_NAME, "cloud_coverage", cloud_coverage)
    StateManager.set_page_state(PAGE_NAME, "buffer_distance", buffer_distance)

    # --- CRITICAL FIX: Check if a sensor is selected BEFORE proceeding ---
    selected_sensor = StateManager.get_page_state(PAGE_NAME, "sensor")
    if not selected_sensor:
        st.info("Please select a sensor to start data preparation.")
        return  # Stop execution until a sensor is chosen

    # Display station editor if selected_stations_detail exists
    selected_stations = st.session_state.get("selected_stations_detail")
    if selected_stations is not None and not selected_stations.empty:
        station_editor_result = station_editor()
    else:
        st.info("No station metadata found.")
        return

    # # Store in session_state
    # st.session_state['cloud_coverage'] = cloud_coverage
    # st.session_state['buffer_distance'] = buffer_distance

    if station_editor_result is not None:
        final_station_list = station_editor_result
        insitu_data = st.session_state['insitu_data']
        collection_name = StateManager.get_page_state(PAGE_NAME, "sensor")
        date_range = StateManager.get_page_state(PAGE_NAME, "date_range")
        buffer_distance = StateManager.get_page_state(PAGE_NAME, "buffer_distance")
        parameter_code = None


        coll_dict = {}
        total_stations = len(final_station_list)

        status = st.status("Starting download remote sensing data...", expanded=False)
        progress_bar = st.progress(0)

        for i, (_, row) in enumerate(final_station_list.iterrows()):
            status.update(label=f"station {i+1}/{total_stations}")

            station_point = (row['lon'], row['lat'])
            site_no = int(row["USGS station code"])
            usgs_data = insitu_data[insitu_data.site_no == site_no]

            coll = cache_rs_dataprepration(
                station_point, date_range, collection_name,
                buffer_distance, parameter_code, site_no
            )
            coll_dict[site_no] = coll

            progress = int((i + 1) / total_stations * 100)
            progress = min(100, max(0, progress))

            progress_bar.progress(progress)

        status.update(label="All stations processed successfully", state="complete")

        st.session_state['rs_collections_by_site'] = coll_dict

    