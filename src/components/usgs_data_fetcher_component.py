# src/components/usgs_data_fetcher_component.py
import streamlit as st
import pandas as pd
import io
import zipfile
from settings import PARAM_CODE_MAP
from src.components.map_component import render_map
from src.config.paths import DEFAULT_STATION_PATH
from src.services.usgs_cache_services import cache_fetch_station_data, cache_download_usgs_data
from src.config.state_manager import StateManager
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

def usgs_data_fetcher(water_quality_param: str, date_range: tuple):
    """Fetch USGS data based on user inputs."""
    # st.markdown("### 🌐 USGS Data Fetcher")
    StateManager.initialize()

    with st.form(key="usgs_form", border=False):
        submitted = st.form_submit_button("Show Stations")

        if submitted:
            region = StateManager.get_page_state("usgs_data_fetcher", "region")
            if region:
                parameter_code = PARAM_CODE_MAP[water_quality_param]
                station_list = cache_fetch_station_data(region, parameter_code)
                StateManager.set_page_state("usgs_data_fetcher", "station_list", station_list)
                st.success("Stations loaded successfully!")
            else:
                st.warning("Please draw a valid region on the map.")

    # Render map
    map_data = render_map(csv_path=DEFAULT_STATION_PATH)
    if map_data.get("last_active_drawing"):
        StateManager.set_page_state("usgs_data_fetcher", "region", map_data["last_active_drawing"])
        st.success("Region drawn successfully!")

    # Display station list
    station_list = StateManager.get_page_state("usgs_data_fetcher", "station_list")
    if station_list is not None:
        st.markdown("#### 📍 USGS Stations")
        df = station_list[[
            'site_no', 'dec_lat_va', 'dec_long_va', 'data_availability', 'last_capture_time'
        ]].rename(columns={
            'data_availability': 'Station Status',
            'last_capture_time': 'Last Measurement',
            'dec_lat_va': 'Latitude',
            'dec_long_va': 'Longitude',
            'site_no': 'USGS Station Code'
        })

        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_selection('multiple', use_checkbox=True)
        gridOptions = gb.build()

        grid_response = AgGrid(
            df,
            gridOptions=gridOptions,
            height=200,
            update_mode=GridUpdateMode.SELECTION_CHANGED,
            theme='light',
            fit_columns_on_grid_load=True
        )

        selected_rows = grid_response['selected_rows']
        # Check if selected_rows is a DataFrame and not empty
        has_selection = isinstance(selected_rows, pd.DataFrame) and not selected_rows.empty



        # Create two columns for the buttons to place them side by side
        col1, col2 = st.columns(2)

        # Prepare Data button in the first column
        with col1:
            # Prepare Data button: Enabled only if there are selected stations
            if st.button("Prepare Data", key="download_usgs_data", disabled=not has_selection):
                # StateManager.set_page_state("usgs_data_fetcher", "selected_stations_detail", selected_rows)
                st.session_state['selected_stations_detail'] = selected_rows
                selected_site_nos = selected_rows["USGS Station Code"].tolist()
                with st.spinner("Preparing USGS data..."):
                    all_data = []
                    zip_buffer = io.BytesIO()
                    start_date = date_range[0].strftime('%Y-%m-%d')
                    end_date = date_range[1].strftime('%Y-%m-%d')
                    parameter_code = PARAM_CODE_MAP[water_quality_param]

                    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                        for site_no in selected_site_nos:
                            try:
                                data = cache_download_usgs_data(
                                    site_no=site_no,
                                    date_range=(start_date, end_date),
                                    parameter_code=parameter_code
                                )
                                if data is not None and not data.empty:
                                    all_data.append(data)
                                    file_name = f"{water_quality_param}_{site_no}.csv"
                                    csv_data = data.to_csv(index=False).encode('utf-8')
                                    zip_file.writestr(file_name, csv_data)
                                else:
                                    st.warning(f"No data available for {site_no}")
                            except Exception as e:
                                st.warning(f"Failed to download data for {site_no}: {str(e)}")

                    if all_data:
                        combined_data = pd.concat(all_data, ignore_index=True)
                        if StateManager.set_insitu_data(combined_data, water_quality_param):
                            st.session_state["insitu_data_source"] = "usgs"
                            st.session_state["usgs_download_data"] = combined_data
                            st.session_state["usgs_zip_buffer"] = zip_buffer.getvalue()
                            st.session_state["usgs_download_ready"] = True
                            st.success("Data downloaded and ready for export!")
                    else:
                        st.info("No data available for download.")
                        st.session_state["usgs_download_ready"] = False
        # Download Data button in the second column
        with col2:
            zip_data = st.session_state["usgs_zip_buffer"]  # No need for default, as it's guaranteed to be bytes
            download_ready = st.session_state.get("usgs_download_ready", False)

            st.download_button(
                label="Download Data",
                data=zip_data,
                file_name=f"{water_quality_param}_stations_data.zip" if download_ready else "no_data.zip",
                mime="application/zip",
                key="download_zipped_data",
                disabled=not download_ready
            )
            
            # # Download Data button: Always visible, enabled only if download_ready is True
            # zip_data = st.session_state.get("usgs_zip_buffer", None)
            # download_ready = st.session_state.get("usgs_download_ready", False)

            # st.download_button(
            #     label="Download Data",
            #     data=zip_data if download_ready else b"",
            #     file_name=f"{water_quality_param}_stations_data.zip" if download_ready else "no_data.zip",
            #     mime="application/zip",
            #     key="download_zipped_data",
            #     disabled=not download_ready
            # )

# # src/components/usgs_data_fetcher_component.py
# import streamlit as st
# from datetime import date
# import pandas as pd
# import io
# import zipfile
# from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode


# from settings import PARAM_CODE_MAP, PARAM_OPTIONS
# from src.components.map_component import render_map
# from src.config.paths import DEFAULT_STATION_PATH
# from src.services.usgs_cache_services import cache_fetch_station_data, cache_download_usgs_data
# from src.config.state_manager import StateManager

# def usgs_data_fetcher():

#     StateManager.initialize()


#     with st.form(key="usgs_form", border=False):
#         col1, col2, col3 = st.columns([2, 2, 1])

#         with col1:
#             wq_params = st.selectbox(
#                 "WQPs:",
#                 options=PARAM_OPTIONS,
#                 index=PARAM_OPTIONS.index(StateManager.get_page_state("usgs_data_fetcher", "water_quality_param", PARAM_OPTIONS[0])),
#                 key="water_quality_param"
#             )

#         with col2:
#             date_range = st.date_input(
#                 "Date Range:",
#                 value=(date(2013, 1, 1), date.today()),
#                 min_value=date(2000, 1, 1),
#                 max_value=date.today(),
#                 key="date_range"
#             )

#         with col3:
#             st.markdown(
#                 '<style>button[title="Show Stations"] { vertical-align: middle;}</style>', unsafe_allow_html=True)
#             submitted = st.form_submit_button("Show Stations")

#         if submitted:
#             region_drawn = "region" in st.session_state and st.session_state["region"] is not None
#             is_ready = region_drawn and wq_params and date_range

#             if is_ready:
#                 st.session_state["station_list"] = None
#                 region = st.session_state.get("region")
#                 parameter_code = PARAM_CODE_MAP[st.session_state.get("water_quality_param")]

#                 # Usgs = UsgsStationProcessor(
#                 #     region=region, parameter_code=PARAM_CODE_MAP[st.session_state.get("water_quality_param")])
#                 # station_list = Usgs.fetch_and_check_station_availability()

#                 station_list = cache_fetch_station_data(region, parameter_code)
#                 st.session_state["station_list"] = station_list
#             else:
#                 st.warning(
#                     "Please provide a valid region, parameter, and date range.")
#     with st.container():
#         map_data = render_map(csv_path=DEFAULT_STATION_PATH)
#         if map_data.get("last_active_drawing"):
#             # st.session_state["region"] = drawn_to_ee_polygon(map_data["last_active_drawing"])
#             st.session_state["region"] = map_data["last_active_drawing"]

#             st.success("Region drawn successfully!")

#     with st.container():
#         if "station_list" in st.session_state:
#             st.subheader("USGS Stations")
#             st.markdown('<div class="dataframe-section">',
#                         unsafe_allow_html=True)

#             df = st.session_state["station_list"][[
#                 'site_no', 'dec_lat_va',
#                 'dec_long_va', 'data_availability', 'last_capture_time'
#             ]].rename(columns={'data_availability': 'station status',
#                                'last_capture_time': 'last measurement',
#                                'dec_lat_va': 'lat',
#                                'dec_long_va': 'lon',
#                                'site_no': 'USGS station code'
#                                })

#             gb = GridOptionsBuilder.from_dataframe(df)
#             gb.configure_selection('multiple', use_checkbox=True)
#             gridOptions = gb.build()

#             grid_response = AgGrid(
#                 df,
#                 gridOptions=gridOptions,
#                 height=200,
#                 update_mode=GridUpdateMode.SELECTION_CHANGED,
#                 allow_unsafe_jscode=True,
#                 theme='light',
#                 fit_columns_on_grid_load=True
#             )

#             selected_rows = grid_response['selected_rows']

#             if isinstance(selected_rows, pd.DataFrame) and not selected_rows.empty:

#                 st.session_state["selected_stations_detail"] = pd.DataFrame(selected_rows)
                
#                 selected_site_nos = selected_rows["USGS station code"].tolist()

#                 start_date = st.session_state["date_range"][0].strftime('%Y-%m-%d')
#                 end_date = st.session_state["date_range"][1].strftime('%Y-%m-%d')
#                 parameter_code = PARAM_CODE_MAP[st.session_state.get("water_quality_param")]
#                 param_name = st.session_state.get("water_quality_param")


#                 all_data = []
#                 zip_buffer = io.BytesIO()

#                 if st.button("Prepare Data", key="download_zipped_data_button"):
#                     with st.spinner("Downloading Data..."):

#                         all_data = []
#                         zip_buffer = io.BytesIO()

#                         with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
#                             for site_no in selected_site_nos:
#                                 try:
#                                     data = cache_download_usgs_data(site_no= site_no, date_range=(start_date, end_date), parameter_code=parameter_code)
#                                     # data = usgs.download_usgs_data(
#                                     #     site_no=site_no,
#                                     #     date_range=(start_date, end_date),
#                                     #     parameter_code=parameter_code
#                                     # )
#                                     if data is not None and not data.empty:
#                                         data["site_no"] = site_no
#                                         all_data.append(data)

#                                         file_name = f"{param_name}_{site_no}.csv"
#                                         csv_data = data.to_csv(index=False).encode('utf-8')
#                                         zip_file.writestr(file_name, csv_data)
#                                     else:
#                                         st.warning(f"No data available for {site_no}")

#                                 except Exception as e:
#                                     st.warning(f"Failed to download data for {site_no}: {e}")

#                         if all_data:
#                             st.session_state["insitu_data"] = pd.concat(all_data, ignore_index=True)

#                             zip_buffer.seek(0)
#                             st.download_button(
#                                 label="Download Zipped Data",
#                                 data=zip_buffer,
#                                 file_name=f"{param_name}_stations_data.zip",
#                                 mime="application/zip",
#                                 key="download_zipped_data"
#                             )
#                         else:
#                             st.info("No data downloaded.")

#             st.markdown('</div>', unsafe_allow_html=True)

#     for css_file in ["assets/css/layout.css", "assets/css/messages.css"]:
#         with open(css_file, encoding="utf-8") as f:
#             st.markdown(f"<style>{f.read()}</style>",
#                         unsafe_allow_html=True)
