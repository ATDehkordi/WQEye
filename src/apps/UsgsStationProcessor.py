import requests
from src.utils.usgs_func import (
    convert_to_utc, get_bbox_from_region_usgs, get_station_data_availability, get_station_list_within_bbox, select_shallowest_measurement_id)
from settings import DATA_AVAILABLE_API_URL, PARAMETER_METADATA

import ee
import pandas as pd
from io import StringIO
import geemap


class UsgsStationProcessor:

    """
        a processor for querying USGS station data within a specified region.
    """

    def __init__(self, region: ee.Geometry.Polygon, parameter_code: str = "63680"):
        """
        initializes the processor for USGS stations.

        args:
            region (optional): The geographical region to define the bounding box.
            parameter_code (optional): The parameter code for querying data (e.g., turbidity: 63680).
        """
        self.region = region
        self.parameter_code = str(parameter_code)
        self.m = geemap.Map()

    def fetch_station(self) -> pd.DataFrame:
        """
            fetches a list of stations within the specified bounding box.
        """
        bbox = self._get_bbox()
        return self._fetch_station_list(bbox)

    def check_data_availability(self, station_list: pd.DataFrame) -> pd.DataFrame:
        """
            checks the data availability for each station in the list.
        """

        results = []
        for _, station in station_list.iterrows():
            station_id = station["site_no"]
            results.extend(self._fetch_station_availability(station_id))
        return pd.DataFrame(results)

    def download_usgs_data(self, site_no, date_range: tuple[str, str], parameter_code: str = "63680", save_raw_data: bool = False):
        """
            downloads USGS data for the specified station and date range.
        """
        # Define the query parameters
        query_params = {
            "sites": site_no,
            "startDT": date_range[0],
            "endDT": date_range[1],
            "parameterCd": parameter_code,
            "format": "rdb"
        }
        response = requests.get(DATA_AVAILABLE_API_URL, params=query_params)

        if response.status_code == 200:

            if save_raw_data:
                save_path = f"{site_no}_{parameter_code}.txt"
                # Save response text directly to a .txt file
                with open(save_path, "w", encoding="utf-8") as file:
                    file.write(response.text)

            ts_id = select_shallowest_measurement_id(
                raw_data=response.text, parameter_code=parameter_code)
            print("An internal number representing a time series: ", ts_id)

            ## Convert response text to DataFrame, skipping comment lines starting with #
            data = pd.read_csv(StringIO(response.text), sep='\t',
                               comment='#', low_memory=False, header=0)

            return self._usgs_data_cleaning(data, parameter_code, ts_id)

        elif response.status_code == 503:
            print(f"""Error {response.status_code}: The server is currently unable to handle 
            the request due to a temporary overloading or maintenance of the server. 
            The implication is that this is a temporary condition which will be alleviated 
            after some delay.""")
        else:
            print(
                f"Failed to retrieve data. Status code: {response.status_code}")

    def fetch_and_check_station_availability(self):
        station_list = self.fetch_station()
        availability_results = self.check_data_availability(station_list)
        station_list_available_df = pd.merge(
            station_list, availability_results, on='site_no')

        return station_list_available_df

    # private methods

    def _get_bbox(self) -> str:
        return get_bbox_from_region_usgs(self.region)

    def _fetch_station_list(self, bbox: str) -> pd.DataFrame:
        return get_station_list_within_bbox(bbox, self.parameter_code)

    def _fetch_station_availability(self, station_id: str) -> dict:
        return get_station_data_availability(station_id, self.parameter_code)

    def _usgs_data_cleaning(self, df, parameter_code: str = "63680", ts_id=None):
        # Cleaning the data here
        # print(df)
        df = df.drop(index=0)
        df = df.dropna()
        # df = df[~df.apply(lambda row: row.astype(str).str.contains('Eqp').any(), axis=1)]
        # df = df[~df.apply(lambda row: row.astype(str).str.contains('***', regex=False).any(), axis=1)]
        # print('after')
        # print(df)
        # Converting local time to datetime
        df["datetime_utc"] = df.apply(convert_to_utc, axis=1)
        # convert data into datetime
        df['datetime'] = pd.to_datetime(
            df['datetime'], format='%Y-%m-%d %H:%M')

        param_name = PARAMETER_METADATA.get(parameter_code, {}).get("SRSName", parameter_code)

        # TS_ID is None Means this station just has 1 depth and column 4 represente data of parameters_code
        if ts_id == None:
            df = df.rename(columns={df.columns[4]: param_name})
            df[param_name] = pd.to_numeric(df[param_name], errors='coerce')
            df.dropna(subset=[param_name],inplace=True)
            # df = df.astype({param_name: 'float'})
            # return df[['site_no', 'datetime', 'tz_cd', param_name, 'datetime_utc']]
            return df[['site_no', param_name, 'datetime_utc']]

        else:
            select_col = [col for col in df.columns if col.startswith(
                str(ts_id) + "_" + str(parameter_code)) and not col.endswith("_cd")]
            df = df.rename(columns={select_col[0]: param_name})
            df[param_name] = pd.to_numeric(df[param_name], errors='coerce')
            df.dropna(subset=[param_name],inplace=True)
            # df = df.astype({param_name: 'float'})
            # return df[['site_no', 'datetime', 'tz_cd', param_name, 'datetime_utc']]
            return df[['site_no', param_name, 'datetime_utc']]


# # Example Usage
# from apps.UsgsStationProcessor import UsgsStationProcessor

# # Define the region of interest
# region = ee.Geometry.Polygon([[[-81.15861212177053, 32.15169970973244],
#           [-81.15861212177053, 32.06940257835562],
#           [-80.9895256654717, 32.06940257835562],
#           [-80.9895256654717, 32.15169970973244]]], None, False)

# # Initialize the processor
# Usgs=UsgsStationProcessor(region=region)

# # Fetch the station list within the region
# station_list = Usgs.fetch_and_check_station_availability()

# # Download USGS data for a specific station
# data = Usgs.download_usgs_data(site_no="02198955", date_range= ('2020-08-16', '2024-09-03'), parameter_code = "63680", save_radata = False)
