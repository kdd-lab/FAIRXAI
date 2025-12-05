from typing import Any, List, Optional

from fairxai.data.dataset import TabularDataset
from fairxai.data.dataset.image_dataset import ImageDataset
from fairxai.data.dataset.text_dataset import TextDataset
from fairxai.data.dataset.timeserie_dataset import TimeSeriesDataset


class DatasetFactory:
    """
    Factory class responsible for creating dataset instances (tabular, image, text, timeseries)
    using a registry pattern and dataset-specific initialization parameters.
    """

    _registry = {
        "tabular": TabularDataset,
        "image": ImageDataset,
        "text": TextDataset,
        "timeseries": TimeSeriesDataset,
    }

    @classmethod
    def create(
            cls,
            data: Any,
            dataset_type: str,
            class_name: Optional[str] = None,
            categorical_columns: Optional[List[str]] = None,
            ordinal_columns: Optional[List[str]] = None
    ):
        """
        Create and return a dataset instance based on the specified type.

        For tabular datasets, additional arguments such as categorical and ordinal columns
        can be provided to correctly configure the dataset descriptor.

        Parameters:
            data (Any): Input data.
            dataset_type (str): One of ["tabular", "image", "text", "timeseries"].
            class_name (str, optional): Target/label column name (for supervised datasets).
            categorical_columns (list[str], optional): Columns to treat as categorical (tabular only).
            ordinal_columns (list[str], optional): Columns to treat as ordinal (tabular only).

        Returns:
            Dataset: An instance of the appropriate dataset subclass.

        Raises:
            ValueError: If the dataset_type is unsupported.
        """
        dataset_type = dataset_type.lower()
        if dataset_type not in cls._registry:
            raise ValueError(
                f"Unsupported dataset type '{dataset_type}'. "
                f"Supported types are: {list(cls._registry.keys())}"
            )

        dataset_class = cls._registry[dataset_type]

        if dataset_type == "tabular":
            # Explicitly handle tabular datasets (requires column type hints)
            return dataset_class(data=data, class_name=class_name, categorical_columns=categorical_columns,
                                 ordinal_columns=ordinal_columns)

        # For all other dataset types, use the standard constructor
        return dataset_class(data, class_name=class_name)

    @classmethod
    def get_class(cls, dataset_type: str):
        """
        Return the dataset class corresponding to the dataset_type string.

        Parameters:
            dataset_type (str): "tabular", "image", "text", or "timeseries"

        Returns:
            Dataset subclass (type)

        Raises:
            ValueError: if dataset_type is unsupported
        """
        dataset_type = dataset_type.lower()
        if dataset_type not in cls._registry:
            raise ValueError(
                f"Unsupported dataset type '{dataset_type}'. "
                f"Supported types are: {list(cls._registry.keys())}"
            )
        return cls._registry[dataset_type]
