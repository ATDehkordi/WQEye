# src/components/insitu_data_uploader_component.py
import streamlit as st
import pandas as pd
from src.utils.functions import read_file
from src.config.state_manager import StateManager

def _process_uploaded_files(uploaded_files: list, required_columns: set) -> tuple[list, list]:
    """
    processes a list of uploaded files, validates them, and returns valid dataframes and skipped files.
    this function is self-contained and adheres to srp.
    """
    valid_dfs = []
    skipped_files_with_reason = []

    for file in uploaded_files:
        try:
            temp_df = read_file(file)
            if temp_df is None:
                raise ValueError("file could not be read or is empty.")

            # improved validation: check if required columns exist in the dataframe
            if required_columns.issubset(temp_df.columns):
                valid_dfs.append(temp_df)
            else:
                missing_cols = required_columns - set(temp_df.columns)
                skipped_files_with_reason.append(
                    {"name": file.name, "reason": f"missing columns: {', '.join(missing_cols)}"}
                )
        except Exception as e:
            skipped_files_with_reason.append({"name": file.name, "reason": f"read error: {str(e)}"})
    
    return valid_dfs, skipped_files_with_reason



def render_insitu_uploader(water_quality_param: str, page_name:str):
    """Upload and merge in-situ data files with validation."""

    help_text = f"""
        **guide: please upload your `csv` or `xlsx` files.**

        each file must include the following columns:
        - `site_no` (station identifier)
        - `datetime_utc` (time in a standard format)
        - `latitude` (geographical latitude)
        - `longitude` (geographical longitude)
        - `{water_quality_param}` (your selected parameter)
        ---
        *pro-tip: for faster performance, we recommend using the `csv` format.*
    """

    uploaded_files = st.file_uploader(
        label="upload your in-situ data files",
        type=["csv", "xls", "xlsx"],
        accept_multiple_files=True,
        help=help_text
    )



    # If user uploaded new files
    if uploaded_files:
        required_columns = {"site_no", "datetime_utc", "latitude", "longitude", water_quality_param}

        valid_dfs, rejected_files_info = _process_uploaded_files(uploaded_files, required_columns)

        insitu_data_df = pd.concat(valid_dfs, ignore_index=True) if valid_dfs else None
        StateManager.set_page_state(page_name, "insitu_data_df", insitu_data_df)
        StateManager.set_page_state(page_name, "rejected_files_info", rejected_files_info)

    else:
        # load from state if no new files were uploaded
        insitu_data_df = StateManager.get_page_state(page_name, "insitu_data_df")
        rejected_files_info = StateManager.get_page_state(page_name, "rejected_files_info", default=[])

    # --- display results and confirmation ---
    if insitu_data_df is not None:
        data_preview_section(insitu_data_df)

        is_confirmed = StateManager.get_page_state(page_name, 'data_is_confirmed', default=False)

        if st.button("confirm and save data"):
            if StateManager.set_insitu_data(insitu_data_df, water_quality_param):

                StateManager.set_page_state(page_name, 'data_is_confirmed', True)
                is_confirmed = True
                StateManager.set_page_state(page_name, 'insitu_data_source', "manual")
                StateManager.set_page_state(page_name, 'selected_stations_detail', build_station_list(insitu_data_df))

        if is_confirmed:
            st.success("In-situ data saved successfully!")

    if rejected_files_info:
        st.warning("⚠️ some files were skipped:")
        for file_info in rejected_files_info:
            st.markdown(f"- **{file_info['name']}**: {file_info['reason']}")
    #     df_list = []
    #     base_columns = None

    #     for file in uploaded_files:
    #         try:
    #             temp_df = read_file(file)
    #             if temp_df is None:
    #                 raise ValueError("Returned DataFrame is None")

    #             if base_columns is None:
    #                 base_columns = list(temp_df.columns)
    #                 df_list.append(temp_df)
    #             elif list(temp_df.columns) == base_columns:
    #                 df_list.append(temp_df)
    #             else:
    #                 skipped_files.append(file.name)

    #         except Exception as e:
    #             skipped_files.append(file.name)
    #             st.warning(f"⚠️ Failed to read {file.name}: {str(e)}")

    #     if df_list:
    #         combined_df = pd.concat(df_list, ignore_index=True)
    #     else:
    #         combined_df = None

    #     # Save processed data in StateManager
    #     StateManager.set_page_state("upload_and_merge", "combined_df", combined_df)
    #     StateManager.set_page_state("upload_and_merge", "skipped_files", skipped_files)

    # # If no new upload, try to load saved data
    # else:
    #     combined_df = saved_combined_df
    #     skipped_files = saved_skipped_files

    # # Show preview and confirmation section
    # if combined_df is not None:
    #     data_preview_section(combined_df)

    #     if st.button("Confirm and Save Data", key="confirm_insitu_data"):
    #         if StateManager.set_insitu_data(combined_df, water_quality_param):
    #             st.session_state["insitu_data_source"] = "manual"
    #             st.session_state["selected_stations_detail"] = build_station_list(
    #                 combined_df)
    #             st.success("✅ Data saved successfully!")

    # # Display any skipped files
    # if skipped_files:
    #     st.warning("⚠️ Some files were skipped due to column mismatch or errors:")
    #     for name in skipped_files:
    #         st.markdown(f"- {name}")


def data_preview_section(df: pd.DataFrame):
    """Display a preview of the data."""
    if st.toggle("Show data preview", value=False, key="data_preview_toggle"):
        st.markdown("#### Data Preview")
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