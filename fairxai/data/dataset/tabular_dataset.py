import pandas as pd

from dataset import Dataset

__all__ = ["TabularDataset"]

from fairxai.data.descriptor.tabular_descriptor import TabularDatasetDescriptor
from fairxai.logger import logger


class TabularDataset(Dataset):
    """
    Handles tabular datasets, providing feature statistics and metadata.

    Attributes:
        data (pd.DataFrame): Raw dataset.
        descriptor (dict): Dictionary describing features.
        class_name (str, optional): Name of the target column.
        target (pd.Series, optional): Target column values.
    """

    def __init__(self, data: pd.DataFrame, class_name: str = None,
                 categorical_columns: list = None, ordinal_columns: list = None):
        super().__init__(data=data, class_name=class_name)

        # Move the target column to the end (optional)
        if class_name is not None and class_name in self.data.columns:
            cols = [c for c in self.data.columns if c != class_name] + [class_name]
            self.data = self.data[cols]

        # Store optional column type hints
        self.categorical_columns = categorical_columns or []
        self.ordinal_columns = ordinal_columns or []

        # Compute descriptor
        self.update_descriptor()

        # Extract target if defined
        if self.class_name in self.data.columns:
            self.target = self.data[self.class_name]
        else:
            self.target = None

    def update_descriptor(self, categorical_columns: list = None, ordinal_columns: list = None):
        """
        Computes the dataset descriptor using TabularDatasetDescriptor.
        """
        categorical_columns = categorical_columns or self.categorical_columns
        ordinal_columns = ordinal_columns or self.ordinal_columns

        descriptor = TabularDatasetDescriptor(
            data=self.data,
            categorical_columns=categorical_columns,
            ordinal_columns=ordinal_columns
        ).describe()

        self.set_descriptor(descriptor)
        logger.info("Tabular dataset descriptor created.")
        return self.descriptor

    # -------------------------------------------------------------------------
    # Convenience methods
    # -------------------------------------------------------------------------
    @classmethod
    def from_csv(cls, filename: str, class_name: str = None, dropna: bool = True,
                 categorical_columns: list = None, ordinal_columns: list = None):
        """
        Load a CSV file into a TabularDataset.
        """
        df = pd.read_csv(filename, skipinitialspace=True, na_values='?', keep_default_na=True)
        if dropna:
            df.dropna(inplace=True)

        dataset = cls(df, class_name=class_name,
                      categorical_columns=categorical_columns,
                      ordinal_columns=ordinal_columns)
        dataset.filename = filename
        logger.info(f"{filename} imported as TabularDataset.")
        return dataset

    @classmethod
    def from_dict(cls, data: dict, class_name: str = None,
                  categorical_columns: list = None, ordinal_columns: list = None):
        """
        Create a TabularDataset from a dictionary of arrays or lists.
        """
        df = pd.DataFrame(data)
        return cls(df, class_name=class_name,
                   categorical_columns=categorical_columns,
                   ordinal_columns=ordinal_columns)

    def get_class_values(self):
        """
        Return the list of distinct target values.

        Raises:
            Exception: If class_name is not defined.
        """
        if not self.class_name:
            raise Exception("ERR: class_name is None. Use set_class_name('<column name>') first.")

        if self.target is not None:
            if hasattr(self.target, 'unique'):
                return list(self.target.unique())
            else:
                return list(set(self.target))

        raise KeyError(f"Target column '{self.class_name}' not found in dataset or descriptor.")

    def get_feature_names(self):
        """
        Return all feature names (numeric + categorical + ordinal).
        """
        if not self.descriptor:
            return []

        names = []
        for section in ['numeric', 'categorical', 'ordinal']:
            names.extend(self.descriptor.get(section, {}).keys())
        return names

    def get_number_of_features(self):
        """
        Return the total number of features.
        """
        return len(self.get_feature_names())

    def get_feature_name(self, index: int):
        """
        Retrieve a feature name by its index.

        Raises:
            IndexError: If no feature matches the given index.
        """
        if not self.descriptor:
            raise IndexError("Descriptor is not set.")

        for section in ['numeric', 'categorical', 'ordinal']:
            for name, info in self.descriptor.get(section, {}).items():
                if info.get('index') == index:
                    return name
        raise IndexError(f"No feature found with index {index}")
