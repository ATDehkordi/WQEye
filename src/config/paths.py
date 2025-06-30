import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data')
DEFAULT_STATION_PATH = os.path.join(DATA_DIR, 'stations.csv')


print(DEFAULT_STATION_PATH)

