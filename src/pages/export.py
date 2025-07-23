# src/pages/export.py

import base64
import pickle
import io
import json
import zipfile
from folium.raster_layers import ImageOverlay

from matplotlib import colormaps
import folium
import pandas as pd
import requests
import streamlit as st
import geopandas as gpd
import geemap.foliumap as geemap
import numpy as np
import folium


import ee
from datetime import timedelta, datetime
from streamlit_folium import st_folium, folium_static
from src.components.export_page_component import render_image_selection_tab, render_prediction_tab, render_roi_tab
from settings import SENSORS_CONFIG
from src.config.state_manager import StateManager
from src.utils.functions import uploaded_file_to_gdf
from folium.plugins import Draw, MeasureControl

from src.utils.zip_checker import check_and_extract_zip_contents
from src.utils.log_scale_transform import ytest_to_initial_scale

PAGE_NAME = 'export'
CENTER_START = [39.949610, -75.150282]
ZOOM_START = 6


@st.cache_data(ttl=3600)  # Cache results for 1 hour
def find_available_images(sensor, start_date, end_date, th_cloud, _region, interested_date):
    """
    Performs the GEE search and returns a dictionary of available images.
    Results are cached to prevent re-running the search unnecessarily.
    The _region argument is used by the caching mechanism to detect changes.
    """
    # st.info("Searching for available satellite images...") # Show info message
    interested_date_ee = ee.Date(interested_date.strftime('%Y-%m-%d'))
    roi = geemap.gdf_to_ee(_region, geodesic=False)

    sensor_info = SENSORS_CONFIG[sensor]
    collection = (
        ee.ImageCollection(sensor_info['collection'])
        .filterBounds(roi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt(sensor_info['cloud_property'], th_cloud))
        .map(sensor_info['scale_function'])
    )

    if collection.size().getInfo() == 0:
        return {}, None  # Return an empty dict if no images are found

    properties_to_get = ['system:time_start'] + \
        sensor_info['unique_properties']
    image_info_list = collection.reduceColumns(
        reducer=ee.Reducer.toList().repeat(len(properties_to_get)),
        selectors=properties_to_get
    ).get('list').getInfo()

    image_details_list = [dict(zip(properties_to_get, item_values))
                          for item_values in zip(*image_info_list)]
    interested_timestamp = interested_date_ee.getInfo()['value']

    sorted_image_details = sorted(
        image_details_list,
        key=lambda item: abs(
            int(item['system:time_start']) - interested_timestamp)
    )

    image_options = {}
    for item in sorted_image_details:
        date_str = datetime.fromtimestamp(
            item['system:time_start'] / 1000).strftime('%Y-%m-%d')
        props_str = " | ".join(
            [f"{prop}: {item[prop]}" for prop in sensor_info['unique_properties']])
        display_name = f"{date_str} | {props_str}"
        image_options[display_name] = item

    return image_options, collection



def initialize_map(center, zoom):
    """
    Initializes a folium map with drawing and measurement controls.
    """
    # Ensure scrollWheelZoom is True as per user's original code
    m = folium.Map(location=center, zoom_start=zoom, scrollWheelZoom=True)

    draw = Draw(export=False,
                filename='custom_drawn_polygons.geojson',
                position='topleft',
                draw_options={'polyline': False,
                              'rectangle': True,  # keep rectangle option for flexibility in drawing
                              'polygon': {'showArea': True, 'showLength': False, 'metric': False, 'feet': False},
                              'circle': {'showArea': True, 'showLength': False, 'metric': False, 'feet': False},
                              'circlemarker': False,
                              'marker': False,
                              },
                edit_options={'poly': {'allowIntersection': False}})
    draw.add_to(m)
    MeasureControl(position='bottomleft', primary_length_unit='miles',
                   secondary_length_unit='meters', primary_area_unit='sqmiles', secondary_area_unit=np.nan).add_to(m)
    return m


style_roi = {
    "color": "#ff3939",
    "fillOpacity": 0,
    "weight": 3,
    "opacity": 1,
    "dashArray": "5, 5",
}


def show():
    """
    Displays the streamlit application with tabs for ROI selection, image download, and prediction.
    """

    tab1, tab2, tab3 = st.tabs([
        "**① Region Of Interest**", "**② Download Image**", "**③ Prediction**"
    ])

    with tab1:
        active_roi_gdf= render_roi_tab()
        # store the result in state for other tabs to use
        if active_roi_gdf is not None:
            StateManager.set_page_state(PAGE_NAME, 'region', active_roi_gdf)
        else:
            # ensure region is cleared if no valid file is uploaded
            StateManager.set_page_state(PAGE_NAME, 'region', None)

    with tab2:
        render_image_selection_tab(PAGE_NAME)

    with tab3:
        render_prediction_tab()
