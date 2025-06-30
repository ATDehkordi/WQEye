# src/components/upload_and_merge_component.py
import streamlit as st
import pandas as pd
from src.utils.functions import read_file
from src.config.state_manager import StateManager

def upload_and_merge_insitudata(water_quality_param: str):
    """Upload and merge in-situ data files with validation."""
    # Initialize state
    # StateManager.initialize()

    # File uploader section
    uploaded_files = st.file_uploader(
        f"Upload your in-situ data with columns: site_no, datetime_utc, latitude, longitude, {water_quality_param}",
        type=["csv", "xls", "xlsx"],
        accept_multiple_files=True,
        help=f"Upload files with consistent columns including site_no, datetime_utc, latitude, longitude, and {water_quality_param}."
    )

    # Retrieve previous data if exists
    saved_combined_df = StateManager.get_page_state("upload_and_merge", "combined_df", default=None)
    saved_skipped_files = StateManager.get_page_state("upload_and_merge", "skipped_files", default=[])

    combined_df = None
    skipped_files = []

    # If user uploaded new files
    if uploaded_files:
        df_list = []
        base_columns = None

        for file in uploaded_files:
            try:
                temp_df = read_file(file)
                if temp_df is None:
                    raise ValueError("Returned DataFrame is None")

                if base_columns is None:
                    base_columns = list(temp_df.columns)
                    df_list.append(temp_df)
                elif list(temp_df.columns) == base_columns:
                    df_list.append(temp_df)
                else:
                    skipped_files.append(file.name)

            except Exception as e:
                skipped_files.append(file.name)
                st.warning(f"⚠️ Failed to read {file.name}: {str(e)}")

        if df_list:
            combined_df = pd.concat(df_list, ignore_index=True)
        else:
            combined_df = None

        # Save processed data in StateManager
        StateManager.set_page_state("upload_and_merge", "combined_df", combined_df)
        StateManager.set_page_state("upload_and_merge", "skipped_files", skipped_files)

    # If no new upload, try to load saved data
    else:
        combined_df = saved_combined_df
        skipped_files = saved_skipped_files

    # Show preview and confirmation section
    if combined_df is not None:
        data_preview_section(combined_df)

        if st.button("Confirm and Save Data", key="confirm_insitu_data"):
            if StateManager.set_insitu_data(combined_df, water_quality_param):
                st.session_state["insitu_data_source"] = "manual"
                st.session_state["selected_stations_detail"] = build_station_list(combined_df)
                st.success("✅ Data saved successfully!")

    # Display any skipped files
    if skipped_files:
        st.warning("⚠️ Some files were skipped due to column mismatch or errors:")
        for name in skipped_files:
            st.markdown(f"- {name}")

def data_preview_section(df: pd.DataFrame):
    """Display a preview of the data."""
    if st.toggle("Show data preview", value=False, key="data_preview_toggle"):
        st.markdown("#### 📊 Data Preview")
        st.info(f"Dataset Size: {df.shape[0]} rows, {df.shape[1]} columns")
        st.dataframe(df.head(), use_container_width=True)

def build_station_list(df: pd.DataFrame) -> pd.DataFrame:
    """Extract station list from in-situ data."""
    # Normalize column names to lowercase
    df_lower_cols = {col.lower(): col for col in df.columns}

    site_no_col = df_lower_cols.get("site_no")
    lat_col = df_lower_cols.get("latitude")
    lon_col = df_lower_cols.get("longitude")

    site_nos = df[site_no_col].unique()

    # Extract latitude and longitude per station if available
    latitudes = (
        df.groupby(site_no_col)[lat_col].first()
        if lat_col else pd.Series([None]*len(site_nos), index=site_nos)
    )

    longitudes = (
        df.groupby(site_no_col)[lon_col].first()
        if lon_col else pd.Series([None]*len(site_nos), index=site_nos)
    )

    # Create station DataFrame
    station_list = pd.DataFrame({
        "USGS station code": site_nos,
        "Latitude": [latitudes.get(s) for s in site_nos],
        "Longitude": [longitudes.get(s) for s in site_nos],
        "Last Measurement": [None] * len(site_nos),
        "Station Status": [None] * len(site_nos)
    })

    return station_list

        
# # src/components/upload_and_merge_component.py
# import streamlit as st
# import pandas as pd

# from src.utils.functions import read_file
# from src.config.state_manager import StateManager

# # def data_preview_section(df):
# #     show_preview = st.toggle("Show data preview", value=False, key="data_preview_toggle")
# #     if show_preview:
# #         st.subheader("📊 Preview of Data")
# #         st.info(f"Final DataSet Size:  {df.shape}")
# #         st.dataframe(df, use_container_width=False)




# def upload_and_merge_insitudata():

#     uploaded_files = st.file_uploader(
#             "Upload your in-situ data", 
#             type=["csv", "xls", "xlsx"],
#             accept_multiple_files=True,
#             help="Upload files with consistent column structure."
#     )

#     if uploaded_files:
#         df_list = []
#         skipped_files = []
#         base_columns = None

#         for file in uploaded_files:
#             try:
#                 temp_df = read_file(file)
#                 if base_columns is None:
#                     base_columns = list(temp_df.columns)
#                     df_list.append(temp_df)
#                 elif list(temp_df.columns) == base_columns:
#                     df_list.append(temp_df)
#                 else:
#                     skipped_files.append(file.name)
#             except Exception as e:
#                 skipped_files.append(file.name)
#                 st.warning(f"Failed to read {file.name}: {str(e)}")

#         if df_list:
#             combined_df = pd.concat(df_list, ignore_index=True)
#             # Show preview before saving
#             data_preview_section(combined_df)
#             if st.button("Confirm and Save Data", key="confirm_insitu_data"):
#                 StateManager.set_insitu_data(combined_df)
        
#         if skipped_files:
#             st.warning("⚠️ Some files were skipped due to column mismatch or errors:")
#             st.write([f"- {name}" for name in skipped_files])

# def data_preview_section(df: pd.DataFrame):
#     """Display a preview of the data."""
#     show_preview = st.toggle("Show data preview", value=False, key="data_preview_toggle")
#     if show_preview:
#         st.markdown("#### 📊 Data Preview")
#         st.info(f"Dataset Size: {df.shape[0]} rows, {df.shape[1]} columns")
#         st.dataframe(df.head(), use_container_width=True)


#     # df = []
#     # skipped_files = []
#     # base_columns = None

#     # if uploaded_file is not None:
#     #     for file in uploaded_file:
#     #         try:
#     #             temp_df = read_file(file)
#     #             if base_columns is None:
#     #                 base_columns = list(temp_df.columns)
#     #                 df.append(temp_df)
#     #             elif list(temp_df.columns) == base_columns:
#     #                 df.append(temp_df)
#     #             else:
#     #                 skipped_files.append(file.name)
#     #         except Exception as e:
#     #             skipped_files.append(file.name)

#     #     if df:
#     #         combined_df = pd.concat(df, ignore_index=True)
#     #         st.session_state["insitu_data"] = combined_df
#     #         # data_preview_section(combined_df)

#     #     if skipped_files:
#     #         st.warning("⚠️ Some files were skipped due to column mismatch or read errors:")
#     #         for name in skipped_files:
#     #             st.markdown(f"- ❌ `{name}`")