import ee

import pandas as pd
from io import StringIO
import requests
from datetime import datetime
import re

from settings import STATION_LIST_API_URL, DATA_AVAILABLE_API_URL, TIMEZONE_MAPPING


def get_bbox_from_region_usgs(region) -> str:
    """
        Convert an Earth Engine Geometry (Polygon) to a standard bBox (minLon, minLat, maxLon, maxLat).

        Input:
            region: An Earth Engine Polygon object (ee.Geometry.Polygon)

        Output:
            bbox: A string in the format "minLon,minLat,maxLon,maxLat"
    """
    if isinstance(region, ee.Feature):
        region = region.geometry()

    # Ensure the input is an ee.Geometry
    if not isinstance(region, ee.Geometry):
        raise ValueError(
            f"Invalid input type: Expected ee.Geometry or ee.Feature, but got {type(region)}")

    # Ensure the geometry is a Polygon
    if region.type().getInfo() != 'Polygon':
        raise ValueError(
            f"Invalid Geometry type: Expected 'Polygon', but got '{region.type().getInfo()}'")

    # get the bounds of the polygon
    bounds = region.bounds()
    # get the coordinates of the bounding box
    bbox_coords = bounds.getInfo()['coordinates'][0]
    # extract minLon, minLat, maxLon, maxLat from the bounding box coordinates
    min_lon = round(min(coord[0] for coord in bbox_coords), 6)
    min_lat = round(min(coord[1] for coord in bbox_coords), 6)
    max_lon = round(max(coord[0] for coord in bbox_coords), 6)
    max_lat = round(max(coord[1] for coord in bbox_coords), 6)
    # Calculate the area of the bounding box (in square degrees)
    lon_diff = abs(max_lon - min_lon)
    lat_diff = abs(max_lat - min_lat)
    area = lon_diff * lat_diff
    if area > 25:
        raise ValueError("The bBox dimensions cannot exceed 25 degrees.")

    # create the bBox string
    bbox = f"{min_lon},{min_lat},{max_lon},{max_lat}"
    return bbox


def get_station_list_within_bbox(bbox: str, parameter_code: str = "63680"):
    """
        Get a list of stations within the specified bounding box and parameter code.

    args:
        bbox: str, bounding box to query stations within. '-81.158612,32.069403,-80.989526,32.1517'
        parameter_cd: str, parameter code to retrieve data for (default is 63680).
    Returns: pandas DataFrame containing the list of stations
    """

    query_params = {
        "bBox": bbox,
        "format": "rdb",
        # Instantaneous values (time-series measurements typically recorded by automated equipment at frequent intervals (e.g., hourly)
        "hasDataTypeCd": "iv",
        "parameterCd": str(parameter_code),
    }
    print("********", parameter_code)

    columns = ['site_no', 'station_nm',
               'site_tp_cd', 'dec_lat_va', 'dec_long_va']

    response = requests.get(STATION_LIST_API_URL, params=query_params)

    # Check if the request was successful
    if response.status_code == 200:
        station_data_raw = response.text
        lines = station_data_raw.splitlines()
        data = [line for line in lines if not line.startswith('#')]
        station_list = pd.read_csv(StringIO("\n".join(data)), sep='\t')
        station_list = station_list[columns][1:]

        return station_list

    # Handle 404 error: no stations found
    elif response.status_code == 404:
        raise ValueError(
            "No stations found within the specified bounding box. Possible reasons include "
            "invalid site numbers, parameter unavailability, or no data for the requested area."
        )

    else:
        raise ConnectionError(
            f"Failed to retrieve station list. Status code: {response.status_code}")


def check_last_measurement(acquisition_data):
    """Check the most recent measurement date from the acquired data."""
    # Convert 'datetime' column to datetime format
    acquisition_data['datetime'] = pd.to_datetime(
        acquisition_data['datetime'], format='%Y-%m-%d %H:%M', errors='coerce')
    # Find the most recent timestamp (last measurement)
    last_measurement_time = acquisition_data.loc[acquisition_data['datetime'].idxmax(
    )]['datetime']
    return last_measurement_time


def get_station_data_availability(station_id, parameter_code):
    """Fetch data availability and last measurement time for a given station."""
    # Initialize an empty list to store data availability details
    station_data_availability = []

    # Set up the query parameters for the request
    query_params = {
        "sites": station_id,
        "parameterCd": parameter_code,
        "format": "rdb"
    }

    # Send the request to the API
    response = requests.get(DATA_AVAILABLE_API_URL, params=query_params)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        data_txt = response.text

        # Check if 'datetime' is present in the response
        if "datetime" in data_txt:
            # Process the response data to remove comment lines
            lines = data_txt.splitlines()
            data = [line for line in lines if not line.startswith('#')]
            # Convert the data to a pandas DataFrame
            acquisition_data = pd.read_csv(StringIO("\n".join(data)), sep='\t')
            # Get the last measurement time from the data

            last_measurement_time = check_last_measurement(acquisition_data)

            today = datetime.today().date()

            # Check if the last measurement is today
            if last_measurement_time.date() == today:
                station_data_availability.append({
                    "site_no": station_id,
                    "data_availability": "Measurement available",
                    "last_capture_time": last_measurement_time
                })
            # Check if the last measurement is older than 2020
            elif last_measurement_time.year < 2020:
                station_data_availability.append({
                    "site_no": station_id,
                    "data_availability": "No measurement available",
                    "last_capture_time": last_measurement_time
                })
            # Handle the case where the last measurement is not recent
            else:
                station_data_availability.append({
                    "site_no": station_id,
                    "data_availability": "Recent measurement unavailable",
                    "last_capture_time": last_measurement_time
                })
        else:
            # If 'datetime' is not found, the station is inactive
            station_data_availability.append({
                "site_no": station_id,
                "data_availability": "Inactive station",
                "last_capture_time": None
            })

    # Handle the case where the station is not found (404 error)
    elif response.status_code == 404:
        station_data_availability.append({
            "site_no": station_id,
            "data_availability": "Not found (404)",
            "last_capture_time": None
        })
    else:
        # Handle other error codes
        station_data_availability.append({
            "site_no": station_id,
            "data_availability": f"{response.status_code} Bad Request",
            "last_capture_time": None
        })

    return station_data_availability


def select_shallowest_measurement_id(raw_data, parameter_code: str = "63680"):
    """
        extracts the TS_ID corresponding to the minimum depth from a USGS data file.


    args:
        raw_data (_type_): USGS raw data
        parameter_code (str): Parameter code to filter the data

    returns:
        int: min_depth if available, otherwise None.

    """
    depth_map = {}
    file = StringIO(raw_data)

    # Read the file first to find the header row
    # with open(raw_data, "r", encoding="utf-8") as file:

    for line in file:
        # Only process comment lines that might contain depth information

        if line.startswith("#"):
            match = re.search(
                fr"(\d+)\s+{parameter_code}.*\[(\d+) ft.\]", line)
            # match = re.search(fr"(\d+)\s+{parameter_code}.*\[(\d+)\s*ft\]", line)

            if match:
                ts_id, depth = match.groups()
                depth_map[int(depth)] = int(ts_id)

    if depth_map:
        min_depth = min(depth_map.keys())
        ts_id = depth_map[min_depth]
        # Sort depths for better readability
        all_depths = sorted(depth_map.keys())
        print(f"✔ Found minimum depth: {min_depth} ft (ID: {ts_id})")
        print(
            f"📌 This station has data at depths: {', '.join(map(str, all_depths))} ft.")

        return ts_id

    else:
        print("⚠ No multiple depths found. This station has data for only one depth.")
        ts_id = None
        return ts_id

def convert_to_utc(row):
    if row['tz_cd'] in TIMEZONE_MAPPING:
        tz_offset= TIMEZONE_MAPPING.get(row['tz_cd'], "+00:00")['utc_offset']  ## Default to UTC if not found
    else:
        print(f"Timezone not found for {row['tz_cd']}")
        tz_offset = "+00:00"
    ## this is horly "%Y-%m-%d %H:%M" structred data for usgs data
    dt = datetime.strptime(row["datetime"], "%Y-%m-%d %H:%M")
    hours, minutes = map(int, tz_offset.split(":"))
    offset = pd.Timedelta(hours=hours, minutes=minutes)
    dt_utc = dt - offset
    return dt_utc
# ## Example
# data["datetime_utc"] = data.apply(convert_to_utc, axis=1)