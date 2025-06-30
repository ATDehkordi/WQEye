import streamlit as st
import pandas as pd
from src.apps.RemoteSensingDataPreparation import RemoteSensingDataPreparation
from src.apps.UsgsStationProcessor import UsgsStationProcessor
from src.utils.functions import drawn_to_ee_polygon
from src.utils.matchup_func import convert_features_to_dataframe, match_up_usgs_and_rs_data

@st.cache_data(ttl=86400)
def cache_fetch_station_data(region, parameter_code):
    """
    Fetch USGS station data for a given region and parameter code.
    """
    try:
        region =drawn_to_ee_polygon(region)
        usgs = UsgsStationProcessor(region=region, parameter_code=parameter_code)
        return usgs.fetch_and_check_station_availability()
    except Exception as e:
        st.error(f"Error fetching station data: {e}")

@st.cache_data(ttl=86400)
def cache_download_usgs_data(site_no, date_range, parameter_code):
    """
    Download USGS data for a specific station, date range, and parameter code.
    """
    usgs = UsgsStationProcessor(region=None, parameter_code=parameter_code)
    return usgs.download_usgs_data(site_no=site_no, date_range=date_range, parameter_code=parameter_code)


@st.cache_data(ttl=86400)
def cache_rs_dataprepration(station_point, date_range, collection_name, buffer_distance, parameter_code, site_no):
    """
    
    """
    neighborhood_dimension = 1
    rs= RemoteSensingDataPreparation(collection_name=collection_name,
                                station_point=station_point,
                                region = None,
                                date_range=date_range,
                                neighborhood_dimension= neighborhood_dimension,
                                buffer_distance = buffer_distance
                            )
    ## Run the pipeline for the imagecollection
    return rs.run_pipeline()



@st.cache_data(ttl=86400)
def cache_matchup_usgs_and_rs_data(_rs_data, usgs_data_site, parameter_code, threshold):
    feature_collection = match_up_usgs_and_rs_data(
        _rs_data,
        usgs_data_site,
        parameter_code=parameter_code,
        threshold=threshold
    )
    df = convert_features_to_dataframe(feature_collection, parameter_code=parameter_code)
    return df