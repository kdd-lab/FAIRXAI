import pandas as pd

__all__ = ["TabularDataset"]

from fairxai.data.dataset import Dataset
from fairxai.data.descriptor.tabular_descriptor import TabularDatasetDescriptor
from fairxai.logger import logger


class TabularDataset(Dataset):
    """
    Represents a tabular dataset with capabilities for managing categorical
    and ordinal columns, setting a target column, and computing descriptive
    metadata for the dataset.

    This class allows initialization from various data sources such as a DataFrame,
    CSV, or dictionary. It provides functionality to maintain, update, and describe
    dataset attributes. Features include moving the target column, extracting
    distinct target values, retrieving feature names, and more.

    Attributes:
        categorical_columns: list
            A list of column names in the dataset that are categorized as
            categorical.
        ordinal_columns: list
            A list of column names in the dataset that are categorized as ordinal.
    """

    def __init__(self, data: pd.DataFrame, class_name: str = None,
                 categorical_columns: list = None, ordinal_columns: list = None):
        """
        Initializes an object with a given dataset, specifies the class/target column,
        and optionally sets categorical and ordinal column attributes. Provides the
        capability to compute a descriptor and handle the target column separately.

        Parameters:
            data (pd.DataFrame): The dataset to be utilized.
            class_name (str, optional): The name of the target column. Defaults to None.
            categorical_columns (list, optional): A list of categorical column names.
                                                  Defaults to None.
            ordinal_columns (list, optional): A list of ordinal column names.
                                               Defaults to None.

        Raises:
            ValueError: If an error occurs while computing the descriptor.
        """
        super().__init__(data=data, class_name=class_name)

        # Move the target column to the end (optional)
        if class_name is not None and class_name in self.data.columns:
            cols = [c for c in self.data.columns if c != class_name] + [class_name]
            self.data = self.data[cols]

        # Store optional column type hints
        self.categorical_columns = categorical_columns or []
        self.ordinal_columns = ordinal_columns or []

        # Compute descriptor
        try:
            self.update_descriptor()
        except ValueError as e:
            raise ValueError(e)

        # Extract target if defined
        if self.class_name in self.data.columns:
            self.target = self.data[self.class_name]
        else:
            self.target = None

    def update_descriptor(self, categorical_columns: list = None, ordinal_columns: list = None):
        """
        Updates the tabular dataset descriptor based on the provided categorical and ordinal
        columns. If no columns are provided, existing categorical and ordinal columns will
        be used by default. This method computes the descriptor for the dataset and updates
        the current instance descriptor.

        Args:
            categorical_columns: list, optional
                A list of columns to be treated as categorical. If not provided, the stored
                categorical columns will be used.
            ordinal_columns: list, optional
                A list of columns to be treated as ordinal. If not provided, the stored
                ordinal columns will be used.

        Raises:
            ValueError: Raised if there is an error computing the descriptor due to invalid
            column values or other data-related issues.

        Returns:
            The computed and updated dataset descriptor.
        """
        categorical_columns = categorical_columns or self.categorical_columns
        ordinal_columns = ordinal_columns or self.ordinal_columns
        try:
            descriptor = TabularDatasetDescriptor(
                data=self.data,
                categorical_columns=categorical_columns,
                ordinal_columns=ordinal_columns
            ).describe()
        except ValueError as e:
            logger.error(f"Error computing descriptor: {e}")
            raise ValueError(e)

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

    def get_feature_names(self, include_target: bool = False):
        """
        Return all feature names (numeric + categorical + ordinal).

        By default, excludes the target column from the feature list.
        Set include_target=True to include it.

        Returns:
            list[str]: List of feature names.
        """
        names = []
        for section in ['numeric', 'categorical', 'ordinal']:
            names.extend(self.descriptor.get(section, {}).keys())

        # Remove the target column if present and not requested
        if not include_target and self.class_name in names:
            names.remove(self.class_name)

        return names

    def get_number_of_features(self, include_target: bool = False):
        """
        Return the total number of features.

        By default, excludes the target column from the count.
        """
        return len(self.get_feature_names(include_target=include_target))

    def get_feature_name(self, index: int):
        """
        Retrieve a feature name by its index.

        Raises:
            IndexError: If no feature matches the given index.
        """

        for section in ['numeric', 'categorical', 'ordinal']:
            for name, info in self.descriptor.get(section, {}).items():
                if info.get('index') == index:
                    return name
        raise IndexError(f"No feature found with index {index}")
