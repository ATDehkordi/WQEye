import ee
from src.config.collection_info import collection_info


class ImageCollectionGetter():
    def __init__(self, collection_name:str, point: ee.Geometry.Point, data_range: tuple, cloud_cover: float = 10):
        """_summary_

        Args:
            
            collection_name (str): in this version of code just select S2, L8, L9
            data_range (tuple): (start_date, end_date)
            region (ee.Geometry.Point): _description_
            cloud_cover (float): _description_
        """

        self.collection_name = collection_name.upper()
        self.data_range = data_range
        self.point = point
        self.cloud_cover = cloud_cover
        self.collection = None
        self.collection_sattelite = ee.Dictionary(collection_info.get(self.collection_name))

    def filter_by_region(self):
        """Filter the collection by the specified geographic region."""
        
        self.collection = (
            ee.ImageCollection(self.collection_sattelite.get('id').getInfo())
            .filterBounds(self.point)
        )
        
    def filter_by_date(self):
        """Filter the collection by the specified date range."""
        self.collection = self.collection.filterDate(*self.data_range)

    def filter_by_cloud_cover(self):
        """Filter the collection by cloud cover percentage."""
        self.collection = self.collection.filter(ee.Filter.lt(self.collection_sattelite.get('cloud_field'), self.cloud_cover))

    def get_collection(self) -> ee.ImageCollection:
        """Return the filtered collection."""

        if not self.collection:
            self.filter_by_region()
        self.filter_by_date()
        self.filter_by_cloud_cover()
        return self.collection

