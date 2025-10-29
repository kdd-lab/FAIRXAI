from typing import Any

from fairxai.data.dataset import TabularDataset
from fairxai.data.dataset.image_dataset import ImageDataset
from fairxai.data.dataset.text_dataset import TextDataset
from fairxai.data.dataset.timeserie_dataset import TimeSeriesDataset


class DatasetFactory:
    """
    Responsible for creating dataset instances of various types using a registry pattern.

    This factory class abstracts the creation of dataset objects based on a specified type.
    It uses a registry mapping to identify and instantiate the appropriate dataset class for
    handling specific dataset types such as tabular, image, text, and timeseries datasets.
    """

    # Registry mapping dataset type strings to dataset classes
    _registry = {
        "tabular": TabularDataset,
        "image": ImageDataset,
        "text": TextDataset,
        "timeseries": TimeSeriesDataset
    }

    @classmethod
    def create(cls, data: Any, dataset_type: str, class_name: str = None):
        """
        Creates and returns an instance of a dataset class based on the provided dataset type.

        This method serves as a factory for creating instances of registered dataset types.
        It checks whether the specified dataset type is supported, and if so, initializes an
        instance of the corresponding dataset class.

        Parameters:
            data: Any
                The data to be processed by the dataset class.
            dataset_type: str
                The type of dataset to create. Must be a registered dataset type.
            class_name: str, optional
                The target column name for tabular datasets, defaults to None.

        Returns:
            object
                An instance of the dataset class corresponding to the specified dataset type.

        Raises:
            ValueError
                If the provided dataset type is not supported, or not found in the registry.
        """
        dataset_type = dataset_type.lower()
        if dataset_type not in cls._registry:
            raise ValueError(f"Unsupported dataset type '{dataset_type}'. "
                             f"Supported types are: {list(cls._registry.keys())}")

        dataset_class = cls._registry[dataset_type]
        return dataset_class(data, class_name)
