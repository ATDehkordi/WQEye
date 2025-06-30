import pandas as pd
import datetime
import ee
from settings import PARAMETER_METADATA


def match_up_usgs_and_rs_data(remote_sensing_data, usgs_station_data, parameter_code, threshold):

    param_name = PARAMETER_METADATA.get(parameter_code, {}).get("SRSName", parameter_code)
    print('******************',param_name)
    # Retrieve features from the remote sensing data as a dictionary
    # remote_sensing_features = remote_sensing_data.getInfo()['features']
    remote_sensing_features = remote_sensing_data

    matched_features = []  # Features with matched usgs  values
    unmatched_features = []  # Features where usgs data could not be matched

    for feature in remote_sensing_features:
        # Convert the timestamp of the feature from milliseconds to datetime
        # feature_timestamp = datetime.datetime.fromtimestamp(
        #     feature['properties']['system:time_start'] / 1000)

        feature_timestamp = datetime.datetime.utcfromtimestamp(
            feature['properties']['system:time_start'] / 1000)
        
        # Find the nearest in-situ data timestamp to the remote sensing feature's timestamp
        nearest_index = (
            usgs_station_data['datetime_utc'] - feature_timestamp).abs().idxmin()
        nearest_row = usgs_station_data.loc[nearest_index]

        # Calculate the time difference between feature and the nearest in-situ data
        time_diff_seconds = abs(
            (feature_timestamp - nearest_row['datetime_utc']).total_seconds())

        # Check if the time difference is within the acceptable threshold (20 mins or 1200 seconds)
        # if time_diff_seconds <= 1200:
        if time_diff_seconds <= threshold:
        
            # Add turbidity as a new property to the feature
            feature['properties'].update({
                'site_no': int(nearest_row['site_no']),
                # 'tz_cd': nearest_row['tz_cd'],
                'USGS_data_utc': nearest_row['datetime_utc'].strftime('%Y-%m-%d %H:%M:%S'),
                'system_time_start': str(feature['properties']['system:time_start']),
                'ImageAquisition_time': feature_timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                # f"nearest_{param_name}": float(nearest_row[param_name]),
                f"{param_name}": float(nearest_row[param_name])
            })
            matched_features.append(feature)
        else:
            unmatched_features.append(time_diff_seconds)

        # Clean up unsupported properties in the geometry if present
        feature_geometry = feature.get('geometry', {})
        feature_geometry.pop('geodesic', None)
        feature_geometry.pop('evenOdd', None)

    # Convert the updated list of matched features back into a FeatureCollection
    updated_remote_sensing_data = ee.FeatureCollection(matched_features)

    # Output the number of unmatched features (i.e., features that could not be matched with in-situ data)
    print(
        f"Number of remote sensing features without matching in-situ data: {len(unmatched_features)}")

    return updated_remote_sensing_data


def get_day_of_year(date):
    import datetime

    # Convert Unix timestamp to UTC
    # divide to 1000 because input date is miliseconds
    date_obj = datetime.datetime.fromtimestamp(date/1000)
    return date_obj.timetuple().tm_yday


def convert_features_to_dataframe(feature_collection, parameter_code):
    param_name = PARAMETER_METADATA.get(
        parameter_code, {}).get("SRSName", parameter_code)

    data = []
    fc_data = feature_collection.getInfo()

    for i in range(len(fc_data['features'])):
        feature = fc_data['features'][i]['properties']
        numeric_data = {k: v for k, v in feature.items(
        ) if isinstance(v, (int, float, str))}
        data.append(numeric_data)

    df = pd.DataFrame(data)
    # df['doy'] = df['system:time_start'].apply(lambda date: get_day_of_year(date))

    total_rows = len(df)    # Count the total number of rows in the data
    # Count the number of NaN values in the 'nearest_turbidity' column
    # nan_count = df[f"nearest_{param_name}"].isna().sum()
    nan_count = df[param_name].isna().sum()

    # Calculate the percentage of NaN values
    nan_percentage = (nan_count / total_rows) * 100

    # Print the results
    print(f"        Total rows in data: {total_rows}")
    print(
        f"        Number of NaN values in 'nearest_{param_name}': {nan_count}")
    print(
        f"        Percentage of NaN values in 'nearest_{param_name}': {nan_percentage:.2f}%")

    df = df.dropna(subset=param_name)
    return df
