import ee

collection_info = ee.Dictionary({
    'S2': ee.Dictionary({
        'id': 'COPERNICUS/S2_SR_HARMONIZED',
        'cloud_field': 'CLOUD_COVERAGE_ASSESSMENT',
        'spacecraft_field': 'SPACECRAFT_NAME',
        'tile_name': 'MGRS_TILE',
        'qa_band': 'QA60',
        'cloudBitMask': 1024,
        'cirrusBitMask': 2056,
        'band_prefix': 'B.*',
        'scale_factor': ee.Dictionary({'multiplier': 0.0001})
    }),
    'L8': ee.Dictionary({
        'id': 'LANDSAT/LC08/C02/T1_L2',
        'cloud_field': 'CLOUD_COVER',
        'spacecraft_field': 'SPACECRAFT_ID',
        'tile_name': ['WRS_PATH', 'WRS_ROW'],
        'qa_band': 'QA_PIXEL',
        'cloudBitMask': 8,
        'cirrusBitMask': 16,
        'band_prefix': 'SR_B.*',
        'scale_factor': ee.Dictionary({'multiplier': 0.0000275, 'offset': -0.2})
    }),
    'L9': ee.Dictionary({
        'id': 'LANDSAT/LC09/C02/T1_L2',
        'cloud_field': 'CLOUD_COVER',
        'spacecraft_field': 'SPACECRAFT_ID',
        'tile_name': ['WRS_PATH', 'WRS_ROW'],
        'qa_band': 'QA_PIXEL',
        'cloudBitMask': 8,
        'cirrusBitMask': 16,
        'band_prefix': 'SR_B.*',
        'scale_factor': ee.Dictionary({'multiplier': 0.0000275, 'offset': -0.2})
    })
})
