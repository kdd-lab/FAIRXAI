from fairxai.data.descriptor.timeserie_descriptor import TimeSeriesDatasetDescriptor
from fairxai.logger import logger
from . import Dataset


class TimeSeriesDataset(Dataset):
    """
    Represents a dataset specifically designed for time series data.

    This class provides an interface to store, process, and update descriptors for time
    series datasets. It is designed to accommodate time series data and any associated
    metadata with methods to seamlessly integrate descriptive updates.

    Attributes:
        data: The time series data stored in the dataset.
        class_name: An optional name or identifier for the dataset's class/category.
        descriptor: The descriptor for the dataset, initialized as None.

    Methods:
        update_descriptor(): Generates and updates the descriptor for the dataset.
    """

    def __init__(self, data, class_name=None):
        """
        Represents an initializer for an object containing data and an optional class name.
        This class allows setting up attributes for further handling within the instance.

        Attributes:
        data: Contains the primary data for the instance. Its type depends on its usage.
        class_name: Optional; represents the name of the class as a string, if applicable.
        descriptor: Holds additional metadata or information, initialized as None.

        Parameters:
        data: The primary data for the object.
        class_name: Optional; the name of the class as a string.

        """
        super().__init__()
        self.data = data
        self.class_name = class_name
        self.descriptor = None

    def update_descriptor(self):
        """
        Updates the descriptor for the timeseries dataset.

        The method generates a descriptor for the timeseries dataset by using
        the `TimeSeriesDatasetDescriptor` class and sets it within the object.
        The generated descriptor is also returned.

        Returns:
            dict: The generated descriptor for the timeseries dataset.
        """
        logger.info("Descriptor generation for timeseries dataset")
        self.descriptor = TimeSeriesDatasetDescriptor(self.data).describe()
        return self.descriptor
