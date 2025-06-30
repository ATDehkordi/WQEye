from pathlib import Path
import ee
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

STATION_LIST_API_URL = "https://waterservices.usgs.gov/nwis/site/"
DATA_AVAILABLE_API_URL = "https://nwis.waterservices.usgs.gov/nwis/iv/"


PARAMETER_METADATA = {
    "63680": {"SRSName": "turbidity", "parm_unit": "FNU",
              "abbr": "TURB",
              "description": "Turbidity, water, unfiltered, monochrome near infra-red LED light, 780-900 nm, detection angle 90 +-2.5 degrees, formazin nephelometric units (FNU)"
              },
    "00300": {"SRSName": "dissolved_oxygen", "parm_unit": "mg/l",
              "abbr": "DO",
              "description": "Dissolved oxygen, water, unfiltered, milligrams per liter"
              },
    "32316": {"SRSName": "Chl_a", "parm_unit": "ug/l",
              "abbr": "CHL",
              "description": "Chlorophyll fluorescence (fChl), water, in situ, concentration estimated from reference material, micrograms per liter as chlorophyll"
              },
    "00095": {"SRSName": "Specific_conductance", "parm_unit": "microsiemens per centimeter at 25 degrees Celsius",
              "abbr": "SC",
              "description": "Specific conductance, water, unfiltered, microsiemens per centimeter at 25 degrees Celsius"
              },
    "00010": {"SRSName": "Temperature", "parm_unit": "degrees Celsius",
              "abbr": "TEMP",
              "description": "Temperature, water, degrees Celsius"
              },
    "00400": {"SRSName": "PH", "parm_unit": "standard units",
              "abbr": "PH",
              "description": "pH, water, unfiltered, field, standard units"
              },
    "1": {"SRSName": "TSS", "parm_unit": "mg/L",
              "abbr": "TSS",
              "description": "total suspended sediments"
              },
    "2": {"SRSName": "CDOM", "parm_unit": "mg/L",
              "abbr": "CDOM",
              "description": "colored dissolved organic matters"
              },
    "3": {"SRSName": "SDD", "parm_unit": "mg/L",
              "abbr": "SDD",
              "description": "secchi disk depth"
              },
    "4": {"SRSName": "TP", "parm_unit": "mg/L",
              "abbr": "TP",
              "description": "total phosphorus"
              },
    "5": {"SRSName": "TN", "parm_unit": "mg/L",
              "abbr": "TN",
              "description": "total nitrogen"
              },
    "6": {"SRSName": "BOD", "parm_unit": "mg/L",
              "abbr": "BOD",
              "description": "biochemical oxygen demand"
              },
    "7": {"SRSName": "COD", "parm_unit": "mg/L",
              "abbr": "COD",
              "description": "chemical oxygen demand"
              }
}


PARAM_OPTIONS = [meta["SRSName"] for meta in PARAMETER_METADATA.values()]
PARAM_CODE_MAP = {meta["SRSName"]: code for code,
                  meta in PARAMETER_METADATA.items()}

US_STATES = {
    "AL": "Alabama",    "AK": "Alaska",    "AS": "American Samoa",    "AZ": "Arizona",    "AR": "Arkansas",
    "CA": "California",    "CO": "Colorado",    "CT": "Connecticut",    "DE": "Delaware",    "DC": "District of Columbia",
    "FM": "Federated States of Micronesia",    "FL": "Florida",    "GA": "Georgia",    "GU": "Guam",    "HI": "Hawaii",
    "ID": "Idaho",    "IL": "Illinois",    "IN": "Indiana",    "IA": "Iowa",    "KS": "Kansas",    "KY": "Kentucky",
    "LA": "Louisiana",    "ME": "Maine",    "MH": "Marshall Islands",    "MD": "Maryland",    "MA": "Massachusetts",
    "MI": "Michigan",    "MN": "Minnesota",    "MS": "Mississippi",    "MO": "Missouri",    "MT": "Montana",
    "NE": "Nebraska",    "NV": "Nevada",    "NH": "New Hampshire",    "NJ": "New Jersey",    "NM": "New Mexico",
    "NY": "New York",    "NC": "North Carolina",    "ND": "North Dakota",    "MP": "Northern Marianas",    "OH": "Ohio",
    "OK": "Oklahoma",    "OR": "Oregon",    "PW": "Palau",    "PA": "Pennsylvania",    "PR": "Puerto Rico",    "RI": "Rhode Island",
    "SC": "South Carolina",    "SD": "South Dakota",    "TN": "Tennessee",    "TX": "Texas",    "UT": "Utah",    "VT": "Vermont",
    "VA": "Virginia",    "VI": "Virgin Islands",    "WA": "Washington",    "WV": "West Virginia",    "WI": "Wisconsin",    "WY": "Wyoming"
}
TIMEZONE_MAPPING = {
    "ACST": {"name": "Central Australia Standard Time", "region": "Central Australia", "utc_offset": "+09:30"},
    "ACSST": {"name": "Central Australia Summer Time", "region": "Central Australia", "utc_offset": "+10:30"},
    "AEST": {"name": "Australia Eastern Standard Time", "region": "Eastern Australia", "utc_offset": "+10:00"},
    "AESST": {"name": "Australia Eastern Summer Time", "region": "Eastern Australia", "utc_offset": "+11:00"},
    "AFT": {"name": "Afghanistan Time", "region": "Afghanistan", "utc_offset": "+04:30"},
    "AKST": {"name": "Alaska Standard Time", "region": "Alaska", "utc_offset": "-09:00"},
    "AKDT": {"name": "Alaska Daylight Time", "region": "Alaska", "utc_offset": "-08:00"},
    "AST": {"name": "Atlantic Standard Time (Canada)", "region": "Atlantic (Canada)", "utc_offset": "-04:00"},
    "ADT": {"name": "Atlantic Daylight Time", "region": "Atlantic (Canada)", "utc_offset": "-03:00"},
    "AWST": {"name": "Australia Western Standard Time", "region": "Western Australia", "utc_offset": "+08:00"},
    "AWSST": {"name": "Australia Western Summer Time", "region": "Western Australia", "utc_offset": "+09:00"},
    "BT": {"name": "Baghdad Time", "region": "Baghdad", "utc_offset": "+03:00"},
    "CAST": {"name": "Central Australia Standard Time", "region": "Central Australia", "utc_offset": "+09:30"},
    "CADT": {"name": "Central Australia Daylight Time", "region": "Central Australia", "utc_offset": "+10:30"},
    "CCT": {"name": "China Coastal Time", "region": "China Coastal", "utc_offset": "+08:00"},
    "CET": {"name": "Central European Time", "region": "Central Europe", "utc_offset": "+01:00"},
    "CETDST": {"name": "Central European Daylight Time", "region": "Central Europe", "utc_offset": "+02:00"},
    "CST": {"name": "Central Standard Time", "region": "Central North America", "utc_offset": "-06:00"},
    "CDT": {"name": "Central Daylight Time", "region": "Central North America", "utc_offset": "-05:00"},
    "DNT": {"name": "Dansk Normal Time", "region": "Dansk", "utc_offset": "+01:00"},
    "DST": {"name": "Dansk Summer Time", "region": "Dansk", "utc_offset": "+01:00"},
    "EAST": {"name": "East Australian Standard Time", "region": "East Australia", "utc_offset": "+10:00"},
    "EASST": {"name": "East Australian Summer Time", "region": "East Australia", "utc_offset": "+11:00"},
    "EET": {"name": "Eastern Europe Standard Time", "region": "Eastern Europe, Russia Zone 1", "utc_offset": "+02:00"},
    "EETDST": {"name": "Eastern Europe Daylight Time", "region": "Eastern Europe", "utc_offset": "+03:00"},
    "EST": {"name": "Eastern Standard Time", "region": "Eastern North America", "utc_offset": "-05:00"},
    "EDT": {"name": "Eastern Daylight Time", "region": "Eastern North America", "utc_offset": "-04:00"},
    "FST": {"name": "French Summer Time", "region": "French", "utc_offset": "+01:00"},
    "FWT": {"name": "French Winter Time", "region": "French", "utc_offset": "+02:00"},
    "GMT": {"name": "Greenwich Mean Time", "region": "Great Britain", "utc_offset": "00:00"},
    "BST": {"name": "British Summer Time", "region": "Great Britain", "utc_offset": "+01:00"},
    "GST": {"name": "Guam Standard Time", "region": "Guam Standard Time, Russia Zone 9", "utc_offset": "+10:00"},
    "HST": {"name": "Hawaii Standard Time", "region": "Hawaii", "utc_offset": "-10:00"},
    "HDT": {"name": "Hawaii Daylight Time", "region": "Hawaii", "utc_offset": "-09:00"},
    "IDLE": {"name": "International Date Line, East", "region": "International Date Line, East", "utc_offset": "+12:00"},
    "IDLW": {"name": "International Date Line, West", "region": "International Date Line, West", "utc_offset": "-12:00"},
    "IST": {"name": "Israel Standard Time", "region": "Israel", "utc_offset": "+02:00"},
    "IT": {"name": "Iran Time", "region": "Iran", "utc_offset": "+03:30"},
    "JST": {"name": "Japan Standard Time", "region": "Japan Standard Time, Russia Zone 8", "utc_offset": "+09:00"},
    "JT": {"name": "Java Time", "region": "Java", "utc_offset": "+07:30"},
    "KST": {"name": "Korea Standard Time", "region": "Korea", "utc_offset": "+09:00"},
    "LIGT": {"name": "Melbourne, Australia", "region": "Melbourne", "utc_offset": "+10:00"},
    "MET": {"name": "Middle Europe Time", "region": "Middle Europe", "utc_offset": "+01:00"},
    "METDST": {"name": "Middle Europe Daylight Time", "region": "Middle Europe", "utc_offset": "+02:00"},
    "MEWT": {"name": "Middle Europe Winter Time", "region": "Middle Europe", "utc_offset": "+01:00"},
    "MEST": {"name": "Middle Europe Summer Time", "region": "Middle Europe", "utc_offset": "+02:00"},
    "MEZ": {"name": "Middle Europe Zone", "region": "Middle Europe", "utc_offset": "+01:00"},
    "MST": {"name": "Mountain Standard Time", "region": "Mountain North America", "utc_offset": "-07:00"},
    "MDT": {"name": "Mountain Daylight Time", "region": "Mountain North America", "utc_offset": "-06:00"},
    "MT": {"name": "Moluccas Time", "region": "Moluccas", "utc_offset": "+08:30"},
    "NFT": {"name": "Newfoundland Standard Time", "region": "Newfoundland", "utc_offset": "-03:30"},
    "NDT": {"name": "Newfoundland Daylight Time", "region": "Newfoundland", "utc_offset": "-02:30"},
    "NOR": {"name": "Norway Standard Time", "region": "Norway", "utc_offset": "+01:00"},
    "NST": {"name": "Newfoundland Standard Time", "region": "Newfoundland", "utc_offset": "-03:30"},
    "NZST": {"name": "New Zealand Standard Time", "region": "New Zealand", "utc_offset": "+12:00"},
    "NZDT": {"name": "New Zealand Daylight Time", "region": "New Zealand", "utc_offset": "+13:00"},
    "NZT": {"name": "New Zealand Time", "region": "New Zealand", "utc_offset": "+12:00"},
    "PST": {"name": "Pacific Standard Time", "region": "Pacific North America", "utc_offset": "-08:00"},
    "PDT": {"name": "Pacific Daylight Time", "region": "Pacific North America", "utc_offset": "-07:00"},
    "SAT": {"name": "South Australian Standard Time", "region": "South Australia", "utc_offset": "+09:30"},
    "SADT": {"name": "South Australian Daylight Time", "region": "South Australia", "utc_offset": "+10:30"},
    "SET": {"name": "Seychelles Time", "region": "Seychelles", "utc_offset": "+01:00"},
    "SWT": {"name": "Swedish Winter Time", "region": "Swedish", "utc_offset": "+01:00"},
    "SST": {"name": "Swedish Summer Time", "region": "Swedish", "utc_offset": "+02:00"},
    "UTC": {"name": "Universal Coordinated Time", "region": "Universal Coordinated Time", "utc_offset": "00:00"},
    "WAST": {"name": "West Australian Standard Time", "region": "West Australia", "utc_offset": "+07:00"},
    "WADT": {"name": "West Australian Daylight Time", "region": "West Australia", "utc_offset": "+08:00"},
    "WAT": {"name": "West Africa Time", "region": "West Africa", "utc_offset": "-01:00"},
    "WET": {"name": "Western Europe", "region": "Western Europe", "utc_offset": "00:00"},
    "WETDST": {"name": "Western Europe Daylight Time", "region": "Western Europe", "utc_offset": "+01:00"},
    "WST": {"name": "West Australian Standard Time", "region": "West Australian", "utc_offset": "+08:00"},
    "WDT": {"name": "West Australian Daylight Time", "region": "West Australian", "utc_offset": "+09:00"},
    "ZP-11": {"name": "UTC -11 hours", "region": "UTC -11 hours", "utc_offset": "-11:00"},
    "ZP-2": {"name": "UTC -2 hours", "region": "Zone UTC -2 Hours", "utc_offset": "-02:00"},
    "ZP-3": {"name": "UTC -3 hours", "region": "Zone UTC -3 Hours", "utc_offset": "-03:00"},
    "ZP11": {"name": "UTC +11 hours", "region": "Zone UTC +11 Hours", "utc_offset": "+11:00"},
    "ZP4": {"name": "UTC +4 hours", "region": "Zone UTC +4 Hours", "utc_offset": "+04:00"},
    "ZP5": {"name": "UTC +5 hours", "region": "Zone UTC +5 Hours", "utc_offset": "+05:00"},
    "ZP6": {"name": "UTC +6 hours", "region": "Zone UTC +6 Hours", "utc_offset": "+06:00"}
}  # https://help.waterdata.usgs.gov/code/tz_query?fmt=html -----> Base URL File


sensor_band_dict = {
    "S2": [  # Sentinel-2
        "B1", "B2", "B3", "B4", "B5", "B6", "B7",
        "B8", "B8A", "B9", "B11", "B12"
    ],
    "L8": [  # Landsat 8
        "SR_B1", "SR_B2", "SR_B3", "SR_B4",
        "SR_B5", "SR_B6", "SR_B7"
    ],
    "L9": [  # Landsat 9
        "SR_B1", "SR_B2", "SR_B3", "SR_B4",
        "SR_B5", "SR_B6", "SR_B7"
    ]
}


SENSORS_CONFIG = {
    "S2": {
        "name": "Sentinel-2",
        "icon": "",
        "bands": ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
    },
    "L8": {
        "name": "Landsat-8",
        "icon": "",
        "bands": ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']
    },
    "L9": {
        "name": "Landsat-9",
        "icon": "",
        "bands": ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']
    },
    "L89": {
        "name": "Landsat 8 & 9",
        "icon": "",
        "bands": ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']
    }
}


def initialize_gee():
    #  initialize Google Earth Engine
    service_account = "water-quality@waterquality-440405.iam.gserviceaccount.com"
    credentials = ee.ServiceAccountCredentials(
        service_account, "waterquality.json")
    ee.Initialize(credentials)
    print("GEE authentication successfully completed")
