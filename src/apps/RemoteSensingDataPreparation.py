import ee

from src.apps.ImageCollectionGetter import ImageCollectionGetter
from src.utils.rs_utils_func import (add_tile_property, count_images_per_tile, get_most_common_tile, compute_clear_pixel_percentage,
                                 add_date_property, filter_unique_date_images, pointwise_sampling,
                                 generate_neighborhood_array, neighborhoodwise_sampling, point_geometry_converter)


class RemoteSensingDataPreparation():
    def __init__(self, collection_name: str, date_range: tuple[str, str], station_point: ee.Geometry.Point, neighborhood_dimension: int,
                region: ee.Geometry.Polygon, cloud_cover: int = 20, buffer_distance: int = 1000):
        
        self.collection_name = collection_name
        self.date_range = date_range
        self.station_point = point_geometry_converter(station_point)
        self.neighborhood_dimension = neighborhood_dimension
        self.region = region if region else self.station_point.buffer(buffer_distance).bounds()  # create a 1x1 km buffer around the station_point
        self.cloud_cover = cloud_cover
        self.collection = None
        self.results = {}

    def get_image_collection(self):
        rs = ImageCollectionGetter(
            collection_name=self.collection_name,
            data_range=self.date_range,
            point=self.station_point,
            cloud_cover=self.cloud_cover
        )
        self.collection = rs.get_collection()
        return self.collection

    def preprocess_collections(self):

        if self.collection is not None:

            # self.collection = get_most_common_tile(self.collection)
            self.collection = add_tile_property(self.collection)
            count_images_per_tile(self.collection)

            self.collection = self.collection.map(
                lambda img: compute_clear_pixel_percentage(img, self.region)
            ).filterMetadata('VALID_PERCENTAGE', 'greater_than', 0.95)
            print(f'The Number of initial cloud-free images over the current station: ',
                  self.collection.size().getInfo())

            self.unique_date_collection = filter_unique_date_images(
                self.collection)
            print(
                f'The Number of Images contain only images from unique dates: {self.unique_date_collection.size().getInfo()} ')

        else:
            raise ValueError(
                "Image collection is None. Ensure that get_image_collection() is called before preprocess_collections().")

    def extract_required_data(self):

        neighbour_array = self.unique_date_collection.map(
            lambda img: generate_neighborhood_array(img, self.neighborhood_dimension))

        self.sampled_values_from_neighborhood = neighbour_array.map(
            lambda img: pointwise_sampling(img, self.station_point)).flatten()

        self.sampled_values_at_station = self.unique_date_collection.map(
            lambda img: pointwise_sampling(img, self.station_point)).flatten()

        self.mean_values_from_neighborhoods = self.unique_date_collection.map(
            lambda img: neighborhoodwise_sampling(img, self.station_point, self.neighborhood_dimension)).flatten()

    def run_pipeline(self):
        self.get_image_collection()
        self.preprocess_collections()
        self.extract_required_data()

        self.results['sampled_values_at_station'] = self.sampled_values_at_station
        self.results['sampled_values_from_neighborhood'] = self.sampled_values_from_neighborhood
        self.results['mean_values_from_neighborhoods'] = self.mean_values_from_neighborhoods

        return self.results

# Example Usage
# from apps.RemoteSensingDataPreparation import RemoteSensingDataPreparation
# date_range = ('2013-08-16', '2024-09-03')
# collection_name= 'S2'
# station_point = ee.Geometry.Point([-81.0069444, 32.10301011834791])
# neighborhood_dimension=1

# region = ee.Geometry.Polygon([[[-81.15861212177053, 32.15169970973244],
#           [-81.15861212177053, 32.06940257835562],
#           [-80.9895256654717, 32.06940257835562],
#           [-80.9895256654717, 32.15169970973244]]], None, False)

# rs= RemoteSensingDataPreparation(collection_name=collection_name, station_point=station_point, date_range=date_range, neighborhood_dimension=neighborhood_dimension, region=region)
# collection= rs.run_pipeline()
