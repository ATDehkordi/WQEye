from src.config.collection_info import collection_info
import ee
import numpy as np


def find_most_common_landsat_tile(image_collections: ee.ImageCollection, path_field: str = 'WRS_PATH', row_field: str = 'WRS_ROW'):

    def extract_tile(img):
        return ee.Feature(None, {
            path_field: img.get(path_field),
            row_field: img.get(row_field)
        })

    path_row_fc = image_collections.map(
        extract_tile).distinct([path_field, row_field])

    def count_images(feature):
        WRS_PATH = feature.get(path_field)
        WRS_ROW = feature.get(row_field)

        filtered = image_collections.filter(
            ee.Filter.And(
                ee.Filter.eq(path_field, WRS_PATH),
                ee.Filter.eq(row_field, WRS_ROW)
            )
        )

        count = filtered.size()
        return feature.set('count', count)

    path_row_count = path_row_fc.map(count_images)
    most_common = path_row_count.sort('count', False).first()
    max_tile_name = (most_common.get(path_field).getInfo(), most_common.get(
        row_field).getInfo(), most_common.get('count').getInfo())

    return max_tile_name


def get_most_common_tile(image_collection: ee.ImageCollection) -> ee.ImageCollection:

    # Extract the satellite id from the image collection like 'S2', 'L8', 'L9'
    collection_name = extract_spacecraft_name(image_collection)

    collection_sattelite = ee.Dictionary(collection_info.get(collection_name))
    tile_name = collection_sattelite.get('tile_name').getInfo()

    if isinstance(tile_name, str):
        tile_ids = image_collection.aggregate_array(tile_name).getInfo()

        unique_tiles, counts = np.unique(tile_ids, return_counts=True)
        max_tile_index = np.argmax(counts)
        max_tile_name = unique_tiles[max_tile_index]

        # filter with max tile name
        filterd_by_max_tile_collection = image_collection.filter(
            ee.Filter.eq(tile_name, max_tile_name))
        print(
            f"Collection '{collection_name.getInfo()}' has max tile name '{max_tile_name}' with {counts[max_tile_index]} occurrences out of {image_collection.size().getInfo()} detected.")

        return filterd_by_max_tile_collection
    elif isinstance(tile_name, list) and len(tile_name) == 2:
        path_field, row_field = tile_name

        max_tile_name = find_most_common_landsat_tile(
            image_collection, path_field=path_field, row_field=row_field)

        filtered_by_max_tile_collection = image_collection.filter(
            ee.Filter.And(
                ee.Filter.eq(path_field, max_tile_name[0]),
                ee.Filter.eq(row_field, max_tile_name[1])
            )
        )

        print(
            f"Collection '{collection_name.getInfo()}' has max tile '{max_tile_name}' with with {max_tile_name[2]} occurrences out of {filtered_by_max_tile_collection.size().getInfo()} detected.")
        return filtered_by_max_tile_collection


def extract_spacecraft_name(input_object: ee.ComputedObject):
    """
    Extracts the short identifier (e.g., 'S2', 'L8') for an Image or ImageCollection.

    Args:
        input_object (ee.ComputedObject): An Earth Engine Image or ImageCollection.

    Returns:
        str: The short identifier (e.g., 'S2', 'L8', 'L9') or None if not found.
    """

    # Extract the collection part of the image_id
    image_id = input_object.get('system:id')
    collection_id = ee.String(image_id)

    # Get the keys (names) of the collection info
    keys = collection_info.keys()

    # Function to check if a collection's 'id' matches the extracted collection_id
    def check_collection(key):
        collection = ee.Dictionary(collection_info.get(key))
        collection_id_check = ee.String(collection.get('id'))

        # Return the key if the collection id matches
        return ee.Algorithms.If(collection_id.index(collection_id_check).neq(ee.Number(-1)), key, None)

    # Apply the check function to each key
    result = keys.map(check_collection)

    # Filter out the None values and return the first match
    filtered_result = result.filter(ee.Filter.neq('item', None))

    # Return the first match, if available
    return filtered_result.get(0)  # Get the first element


def compute_clear_pixel_percentage(image: ee.Image, region: ee.Geometry) -> ee.Image:
    """    
    Computes the percentage of clear pixels in an image within a specified region.


    Args:
        image (ee.Image): The Earth Engine image for which the clear pixel percentage will be computed.
        region (ee.Geometry): The region of interest where the computation will be performed.

    Returns:
        ee.Image: The input image with additional properties:
                  - 'numofpixels': Total number of pixels in the region.
                  - 'validpixels': Number of clear (valid) pixels in the region.
                  - 'valid_percentage': Percentage of clear pixels in the region.
    """
    # Extract spacecraft name
    spacecraft_name = extract_spacecraft_name(image)
    spacecraft_info = ee.Dictionary(collection_info.get(spacecraft_name))

    qa_band = spacecraft_info.get('qa_band')  # No .getInfo()
    cloud_bitmask = spacecraft_info.get('cloudBitMask')  # No .getInfo()
    cirrus_bitmask = spacecraft_info.get('cirrusBitMask')  # No .getInfo()

    bands = image.bandNames()
    selected_band = ee.String(bands.get(0))

    # Generate cloud mask
    qa = image.select([qa_band]).clip(region)

    cloudmask = (
        qa.bitwiseAnd(ee.Number(cloud_bitmask)).eq(0)
        .bitwiseAnd(qa.bitwiseAnd(ee.Number(cirrus_bitmask)).eq(0))
    )

    nocloudpixels = image.select(selected_band).clip(
        region).updateMask(cloudmask)

    numofpixels = image.select(selected_band).clip(region).reduceRegion(
        reducer=ee.Reducer.count(),
        geometry=region,
        scale=60
    )

    validpixels = nocloudpixels.reduceRegion(
        reducer=ee.Reducer.count(),
        geometry=region,
        scale=60
    )

    valid_percentage = ee.Number(validpixels.get(selected_band)).divide(
        ee.Number(numofpixels.get(selected_band))
    )

    return (
        image.set('NUMBER_OF_PIXEL', ee.Number(numofpixels.get(selected_band)))
        .set('VALID_PIXEL', ee.Number(validpixels.get(selected_band)))
        .set('VALID_PERCENTAGE', ee.Number(valid_percentage))
    )


def add_date_property(image: ee.Image) -> ee.Image:  # type: ignore
    """Extracts the date from the 'system:time_start' property of an image and sets it as a new property
            called 'DATE' in 'YYYYMMdd' format.
    Args:
        image (ee.Image): The Earth Engine Image from which the date will be extracted.

    Returns:
        ee.Image: The input image with an additional property 'DATE' containing the formatted date.
    """
    # Function to extract and set date as a property
    date = ee.Date(image.get('system:time_start')).format(
        'YYYYMMdd')  # type: ignore
    return image.set('DATE', date)  # type: ignore


def create_feature_from_date(date, count):
    return ee.Feature(None, {
        'DATE': ee.String(date),
        'count': ee.Number(count)
    })


# def filter_unique_date_images(image_collection: ee.ImageCollection) -> ee.ImageCollection:
#     """
#         Filters the image collection to keep only the images from dates with a single image.


#     Returns:
#         ee.ImageCollection: An ImageCollection containing only images from unique dates.
#     """

#     # Add date property to each image in the collection
#     image_collection = image_collection.map(add_date_property)
#     # Calculate a histogram of dates in the ImageCollection
#     date_histogram = image_collection.aggregate_array(
#         'DATE').reduce(ee.Reducer.frequencyHistogram())

#     # Map the dictionary into a FeatureCollection
#     date_count_fc = ee.FeatureCollection(ee.Dictionary(
#         date_histogram).map(create_feature_from_date).values())

#     # Filter the FeatureCollection to keep only dates with a single image
#     unique_date_fc = date_count_fc.filter(ee.Filter.eq('count', 1))

#     # Convert single-image dates to a list for filtering the ImageCollection
#     unique_date_list = unique_date_fc.aggregate_array('DATE')

#     # Filter the original ImageCollection to keep only images with unique dates
#     filtered_image_collection = image_collection.filter(
#         ee.Filter.inList('DATE', unique_date_list))

#     return filtered_image_collection

def filter_unique_date_images(image_collection: ee.ImageCollection) -> ee.ImageCollection:
    """   
        Filters the image collection to keep only the images from dates with a single image. 

    Args:
        image_collection (ee.ImageCollection): The input ImageCollection to be filtered.

    Returns:
        ee.ImageCollection: An ImageCollection containing only images from unique dates.
    """

    # Add date property to each image in the collection
    image_collection = image_collection.map(add_date_property)

    # Extract distinct tile names from the image collection
    tile_names = image_collection.aggregate_array('tile_name').distinct()

    # Function to compute a date histogram (frequency count) per tile
    def compute_date_histogram(tile_name):
        # Filter the image collection for images with the specific tile_name
        filtered_collection = image_collection.filter(
            ee.Filter.eq('tile_name', tile_name))

        # Create a histogram of dates in the filtered collection
        date_histogram = filtered_collection.aggregate_array(
            'DATE').reduce(ee.Reducer.frequencyHistogram())

        # Convert the histogram into a FeatureCollection, with each date as a feature
        date_count_fc = ee.FeatureCollection(ee.Dictionary(
            date_histogram).map(create_feature_from_date).values())

        # Filter the FeatureCollection to keep only dates with a single image
        unique_date_fc = date_count_fc.filter(ee.Filter.eq('count', 1))

        # Convert the filtered FeatureCollection into a list of unique dates
        unique_date_list = unique_date_fc.aggregate_array('DATE')

        # Filter the original image collection to keep only images with unique dates
        filtered_image_collection = filtered_collection.filter(
            ee.Filter.inList('DATE', unique_date_list))

        return filtered_image_collection

    # Apply the date histogram computation for each tile name
    filter_per_tile = tile_names.map(compute_date_histogram)

    # Merge the filtered collections per tile into one final ImageCollection
    final_filtered_collection = ee.ImageCollection(filter_per_tile.iterate(
        lambda col1, col2: ee.ImageCollection(col1).merge(ee.ImageCollection(col2)), ee.ImageCollection([])))  # Start with an empty ImageCollection

    return final_filtered_collection


def pointwise_sampling(image: ee.Image, region: ee.Geometry) -> ee.FeatureCollection:
    """
        Performs pointwise sampling of a given Earth Engine image over a specified region.

    Args:
        image (ee.Image): The Earth Engine image to be sampled.
        region (ee.Geometry): The geometry defining the region of interest for sampling.

    Returns:
        ee.FeatureCollection: A feature collection containing sampled values from the image.
    """
    # Extract the collection name from the image
    collection_name = extract_spacecraft_name(image)
    collection_sattelite = ee.Dictionary(collection_info.get(collection_name))
    scale_factor_info = ee.Dictionary(collection_sattelite.get('scale_factor'))

    multiplier = ee.Number(scale_factor_info.get('multiplier'))
    offset = ee.Number(scale_factor_info.get('offset', 0))
    band_prefix = ee.String(collection_sattelite.get('band_prefix'))
    spacecraft_id = ee.String(collection_sattelite.get('spacecraft_field'))

    # Fetch metadata about the collection, including band prefix and scaling factors
    # collection = collection_info[collection_name]
    spacecraft_name = image.get(spacecraft_id)

    # Select the bands of interest from the image using the collection's band prefix
    bands = image.select(band_prefix)

    # Apply scaling factors (multiplicative and optional additive offset) to the bands
    bands = bands.multiply(multiplier).add(offset)

    # Perform pointwise sampling of the scaled bands over the specified region
    sampledvalues = bands.sampleRegions(
        collection=ee.FeatureCollection([region]),
        scale=10,
        geometries=True
    ).map(lambda feature: feature
          .set('system:time_start', image.get('system:time_start'))
          .set('spacecraft_name', spacecraft_name)
          .set('tile_name', image.get('tile_name'))


          )
    return sampledvalues


def generate_neighborhood_array(image: ee.Image, neighborhood_dimension: int) -> ee.Image:
    """
    Generates a neighborhood array from an input image using a specified neighborhood dimension.
    """
    # Create a neighborhood array using a square kernel
    # The neighborhood array represents the values of the pixels within the specified neighborhood dimension
    # The neighborhood array has the same dimensions as the input image
    # For example, a neighborhood array of size 3x3 will contain the values of the pixels within a 3x3 square neighborhood around each pixel in the input image

    # Create a square kernel with the specified neighborhood dimension
    kernel = ee.Kernel.square(neighborhood_dimension)
    return image.neighborhoodToArray(kernel)


def neighborhoodwise_sampling(image: ee.Image, region: ee.Geometry, neighborhood_dimension: int) -> ee.FeatureCollection:
    """
    Performs neighborhood-wise sampling of an image around a given location.

    Args:
        image (ee.Image): The Earth Engine image to be sampled.
        region (ee.Geometry): The geometry defining the region of interest for sampling.
        neighborhood_dimension (int): The size of the square kernel for neighborhood reduction.

    Returns:
        ee.FeatureCollection: FeatureCollection containing sampled mean values
    """

    collection_name = extract_spacecraft_name(image)
    collection_sattelite = ee.Dictionary(collection_info.get(collection_name))
    scale_factor_info = ee.Dictionary(collection_sattelite.get('scale_factor'))

    multiplier = ee.Number(scale_factor_info.get('multiplier'))
    offset = ee.Number(scale_factor_info.get('offset', 0))
    band_prefix = ee.String(collection_sattelite.get('band_prefix'))
    spacecraft_id = ee.String(collection_sattelite.get('spacecraft_field'))

    # Fetch metadata about the collection, including band prefix and scaling factors
    # collection = collection_info[collection_name]
    spacecraft_name = image.get(spacecraft_id)

    # Fetch metadata about the collection, including band prefix and scaling factors
    # collection = collection_info[collection_name]

    # Select the bands of interest from the image using the collection's band prefix
    bands = image.select(band_prefix)

    # Apply scaling factors (multiplicative and optional additive offset) to the bands
    bands = bands.multiply(multiplier).add(offset)

    # Compute neighborhood mean values using a square kernel
    neighbor_image = bands.reduceNeighborhood(
        reducer=ee.Reducer.mean(),
        kernel=ee.Kernel.square(neighborhood_dimension)
    )

    # Sample the neighborhood mean values at the region of interest
    sampled_values_mean = neighbor_image.sampleRegions(
        collection=ee.FeatureCollection([region]),
        scale=10,
        geometries=True
    ).map(lambda feature: feature
          .set('system:time_start', image.get('system:time_start'))
          .set('spacecraft_name', spacecraft_name)
          .set('tile_name', image.get('tile_name')))
    return sampled_values_mean


def point_geometry_converter(point):

    if isinstance(point, ee.geometry.Geometry):
        # print ('Type : ee.geometry.Geometry')
        return point

    elif isinstance(point, list) and len(point) == 2:
        # print ('Type : list')
        return ee.Geometry.Point(float(point[0]), float(point[1]))

    elif isinstance(point, tuple) and len(point) == 2:
        # print ('Type : tuple')
        return ee.Geometry.Point(float(point[0]), float(point[1]))

    else:
        raise ValueError(
            "Input must be an ee.Geometry.Point, list, or tuple with two numeric values.")


def add_tile_property(image_collection: ee.ImageCollection) -> ee.ImageCollection:
    """Adds the tile name as a property to each image in the collection."""

    # Extract the satellite ID (e.g., 'S2', 'L8', 'L9')
    collection_name = extract_spacecraft_name(image_collection)

    # Get collection-specific tile field names
    collection_satellite = ee.Dictionary(collection_info.get(collection_name))
    tile_name = collection_satellite.get('tile_name').getInfo()

    # Function to add tile property to each image
    def add_tile(img):
        if isinstance(tile_name, str):  # Sentinel-2 case (single tile field)
            return img.set('tile_name', img.get(tile_name))
        elif isinstance(tile_name, list) and len(tile_name) == 2:  # Landsat case (path & row)
            path_field, row_field = tile_name
            return img.set('tile_name', ee.String(img.get(path_field)).cat('_').cat(ee.String(img.get(row_field))))
        else:
            return img  # Return unchanged if tile name is invalid

    # Apply function to all images
    return image_collection.map(add_tile)


def count_images_per_tile(image_collection: ee.ImageCollection):
    """Counts the number of images per tile in the collection and prints the results."""

    tile_counts = image_collection.aggregate_histogram('tile_name')
    print(f"Number of images by tile: {tile_counts.getInfo()}")
