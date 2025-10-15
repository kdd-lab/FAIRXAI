from typing import Any

from fairxai.dataset import TabularDataset
from fairxai.dataset.image_dataset import ImageDataset
from fairxai.dataset.text_dataset import TextDataset
from fairxai.dataset.timeserie_dataset import TimeSeriesDataset


class DatasetFactory:
    """
    A factory class for creating dataset instances.

    The DatasetFactory class provides an interface to create dataset instances
    based on the specified type. It supports various dataset types such as
    tabular, image, text, and timeseries, and facilitates selecting the correct
    dataset class for instantiation.

    Methods
    -------
    create(data, dataset_type, class_name=None)
        Creates and returns an instance of the specified dataset type.

    """
    _registry = {
        "tabular": TabularDataset,
        "image": ImageDataset,
        "text": TextDataset,
        "timeseries": TimeSeriesDataset
    }

    @classmethod
    def create(cls, data: Any, dataset_type: str, class_name: str = None):
        """
        Creates an instance of a dataset class from the registered dataset types. This method
        allows dynamic creation of dataset instances based on the provided dataset type and
        associated registry.

        Parameters:
            data (Any): The data to be used by the dataset instance.
            dataset_type (str): A string representing the type of dataset. It must
                match a value in the registered dataset type.
            class_name (str, optional): The name of the class to be used by this
                dataset instance. Defaults to None.

        Raises:
            ValueError: If the specified dataset type is not registered.

        Returns:
            The instance of the dataset class corresponding to the specified
            dataset type.
        """
        dataset_type = dataset_type.lower()
        if dataset_type not in cls._registry:
            raise ValueError(f"Unsupported dataset type '{dataset_type}'")

        dataset_class = cls._registry[dataset_type]
        return dataset_class(data, class_name)
