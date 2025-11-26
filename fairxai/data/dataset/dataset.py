__all__ = ["Dataset"]

from abc import ABC, abstractmethod


class Dataset(ABC):
    """
    Generic abstract class to handle datasets of different modalities
    (tabular, image, text, timeseries).

    Attributes:
        data: The raw dataset (DataFrame, list, or array depending on modality)
        descriptor: Dictionary describing dataset structure and statistics
        class_name: Optional target column name (for tabular datasets)
        target: Optional, contains the target column values
    """

    def __init__(self, data=None, class_name: str = None):
        self.data = data
        self.descriptor = None
        self.class_name = class_name
        self._target = None

    # -------------------------------------------------------------------------
    # Abstract methods
    # -------------------------------------------------------------------------
    @abstractmethod
    def update_descriptor(self, *args, **kwargs):
        """
        Must create and assign the dataset descriptor.
        Each subclass should call its specific BaseDatasetDescriptor.describe().
        """
        raise NotImplementedError

    # -------------------------------------------------------------------------
    # Descriptor management
    # -------------------------------------------------------------------------
    def set_descriptor(self, descriptor: dict):
        """
        Assign a descriptor dictionary to the dataset.
        """
        self.descriptor = descriptor

    def set_class_name(self, class_name: str):
        """
        Define the target column name.
        Optionally extracts the target values from the dataset (for tabular).
        """
        self.class_name = class_name
