from datetime import datetime
import pandas as pd
import streamlit as st
import os
import ee
import geopandas as gpd
from streamlit.runtime.uploaded_file_manager import UploadedFile

from settings import SENSORS_CONFIG

# @st.cache_data
# def read_file_from_path(file_path: str):
#     ext = os.path.splitext(file_path)[-1].lower()
#     if ext == ".csv":
#         return pd.read_csv(file_path)
#     elif ext in [".xls", ".xlsx"]:
#         return pd.read_excel(file_path)
#     else:
#         return None


# def read_file(file_input):
#     try:
#         if isinstance(file_input, str):
#             return read_file_from_path(file_input)

#         elif hasattr(file_input, 'name'):
#             ext = os.path.splitext(file_input.name)[-1].lower()
#             if ext == ".csv":
#                 return pd.read_csv(file_input)
#             elif ext in [".xls", ".xlsx"]:
#                 return pd.read_excel(file_input)
#             else:
#                 st.error("Unsupported file type.")
#                 return None
#         else:
#             return None

#     except Exception as e:
#         st.error(f"Error reading file: {e}")
#         return None
def read_file(file_input: str | UploadedFile) -> pd.DataFrame | None:
    """
    reads a file from a path or a streamlit uploadedfile object and returns a pandas dataframe.
    handles csv, xls, and xlsx formats.
    
    returns none if the file format is unsupported or an error occurs.
    this function does not produce any ui side-effects (e.g., st.error).
    """
    try:
        # determine file extension, whether from a path (str) or an object with a .name attribute
        if isinstance(file_input, str):
            file_path = file_input
            ext = os.path.splitext(file_path)[-1].lower()
        elif hasattr(file_input, 'name'):
            file_path = file_input
            ext = os.path.splitext(file_input.name)[-1].lower()
        else:
            # unsupported input type
            return None

        # read the file based on its extension
        if ext == ".csv":
            return pd.read_csv(file_path)
        elif ext in [".xls", ".xlsx"]:
            # pandas needs 'openpyxl' for xlsx and 'xlrd' for xls.
            # ensure these are in requirements.txt
            return pd.read_excel(file_path)
        else:
            # unsupported file extension
            return None

    except Exception as e:
        # log the error for debugging purposes if you have a logger
        # print(f"error reading file: {e}") # for local debugging
        return None

def drawn_to_ee_polygon(drawn_coordinates):
    coords = drawn_coordinates["geometry"]["coordinates"][0]
    return ee.Geometry.Polygon(coords)



def applyscale_S2(image):
    return image.multiply(0.0001)


def applyscale_L89(image):
    opticalBands = image.select('SR_B.*').multiply(0.0000275).add(-0.2)
    thermalBands = image.select('ST_B.*').multiply(0.00341802).add(149.0)
    return image.addBands(opticalBands, None, True).addBands(thermalBands, None, True)




def uploaded_file_to_gdf(uploaded):
        """Convert an uploaded GeoJSON (.json/.geojson) into a GeoDataFrame (EPSG:4326).

        Parameters
        ----------
        uploaded : st.runtime.uploaded_file_manager.UploadedFile
            The file selected by the user from the Streamlit file‑uploader.

        Returns
        -------
        geopandas.GeoDataFrame | None
            A GeoDataFrame in WGS84 CRS, or *None* if conversion fails.
        """
        if uploaded is None:
            return None

        import tempfile
        import uuid

        ext = os.path.splitext(uploaded.name)[1].lower()
        if ext not in (".json", ".geojson"):
            return None  # Only JSON / GeoJSON are accepted

        tmp_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}{ext}")
        with open(tmp_path, "wb") as fp:
            fp.write(uploaded.getbuffer())

        try:
            gdf = gpd.read_file(tmp_path)
            return gdf.to_crs("epsg:4326")
        except Exception:
            return None





# --- KEY CHANGE 1: Function to find and cache image search results ---
@st.cache_data(ttl=3600) # Cache results for 1 hour
def find_available_images(sensor, start_date, end_date, th_cloud, _region, interested_date):
    """
    Performs the GEE search and returns a dictionary of available images.
    Results are cached to prevent re-running the search unnecessarily.
    The _region argument is used by the caching mechanism to detect changes.
    """
    # st.info("Searching for available satellite images...") # Show info message
    interested_date_ee= ee.Date(interested_date.strftime('%Y-%m-%d'))
    sensor_info = SENSORS_CONFIG[sensor]
    collection = (
        ee.ImageCollection(sensor_info['collection'])
        .filterBounds(_region)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt(sensor_info['cloud_property'], th_cloud))
        .map(sensor_info['scale_function'])
    )

    if collection.size().getInfo() == 0:
        return {},None # Return an empty dict if no images are found

    properties_to_get = ['system:time_start'] + sensor_info['unique_properties']
    image_info_list = collection.reduceColumns(
        reducer=ee.Reducer.toList().repeat(len(properties_to_get)),
        selectors=properties_to_get
    ).get('list').getInfo()

    image_details_list = [dict(zip(properties_to_get, item_values)) for item_values in zip(*image_info_list)]
    interested_timestamp = interested_date_ee.getInfo()['value']

    sorted_image_details = sorted(
        image_details_list,
        key=lambda item: abs(int(item['system:time_start']) - interested_timestamp)
    )

    image_options = {}
    for item in sorted_image_details:
        date_str = datetime.fromtimestamp(item['system:time_start'] / 1000).strftime('%Y-%m-%d')
        props_str = " | ".join([f"{prop}: {item[prop]}" for prop in sensor_info['unique_properties']])
        display_name = f"{date_str} | {props_str}"
        image_options[display_name] = item

    return image_options, collection
