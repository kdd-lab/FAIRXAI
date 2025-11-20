import os
from typing import Optional, List, Any, Union
import pandas as pd
from pandas import DataFrame, Series

from fairxai.data.dataset import Dataset
from fairxai.data.descriptor.tabular_descriptor import TabularDatasetDescriptor
from fairxai.logger import logger


class TabularDataset(Dataset):
    """
    Represents a tabular dataset with capabilities for managing categorical
    and ordinal columns, setting a target column, and computing descriptive
    metadata for the dataset.

    This class allows initialization from:
        - a pandas DataFrame
        - a CSV file path
        - a dictionary of arrays/lists

    Attributes:
        categorical_columns: list of column names considered categorical
        ordinal_columns: list of column names considered ordinal
    """

    def __init__(self, data: Any, class_name: str = None,
                 categorical_columns: Optional[List[str]] = None,
                 ordinal_columns: Optional[List[str]] = None):
        """
        Initializes TabularDataset from any supported data type.

        data can be:
            - DataFrame
            - CSV file path
            - dict of arrays/lists
            - list of dicts (records)
        """
        # --- Convert input to DataFrame ---
        if isinstance(data, pd.DataFrame):
            df = data
        elif isinstance(data, str) and os.path.exists(data) and data.lower().endswith(".csv"):
            df = pd.read_csv(data, skipinitialspace=True, na_values='?', keep_default_na=True)
        elif isinstance(data, dict):
            df = pd.DataFrame(data)
        elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            df = pd.DataFrame(data)
        else:
            raise TypeError(
                f"Cannot build TabularDataset from type '{type(data)}'. "
                "Supported types: DataFrame, CSV path, dict, list[dict]."
            )

        super().__init__(data=df, class_name=class_name)

        # Move target column to the end
        if class_name is not None and class_name in self.data.columns:
            cols = [c for c in self.data.columns if c != class_name] + [class_name]
            self.data = self.data[cols]

        self.categorical_columns = categorical_columns or []
        self.ordinal_columns = ordinal_columns or []

        # Compute descriptor
        try:
            self.update_descriptor()
        except ValueError as e:
            raise ValueError(e)

        # Extract target
        if self.class_name in self.data.columns:
            self.target = self.data[self.class_name]
        else:
            self.target = None

    # -------------------------
    # Descriptor update
    # -------------------------
    def update_descriptor(self, categorical_columns: Optional[List[str]] = None,
                          ordinal_columns: Optional[List[str]] = None):
        """
        Updates the tabular dataset descriptor.
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

    # -------------------------
    # Convenience constructors
    # -------------------------
    @classmethod
    def from_csv(cls, filename: str, class_name: str = None, dropna: bool = True,
                 categorical_columns: Optional[List[str]] = None,
                 ordinal_columns: Optional[List[str]] = None):
        """
        Load CSV file into TabularDataset
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
                  categorical_columns: Optional[List[str]] = None,
                  ordinal_columns: Optional[List[str]] = None):
        """
        Create a TabularDataset from a dictionary of arrays or lists
        """
        df = pd.DataFrame(data)
        return cls(df, class_name=class_name,
                   categorical_columns=categorical_columns,
                   ordinal_columns=ordinal_columns)

    # -------------------------
    # Feature / target helpers
    # -------------------------
    def get_class_values(self):
        if not self.class_name:
            raise Exception("ERR: class_name is None. Use set_class_name('<column name>') first.")

        if self.target is not None:
            if hasattr(self.target, 'unique'):
                return list(self.target.unique())
            else:
                return list(set(self.target))

        raise KeyError(f"Target column '{self.class_name}' not found in dataset or descriptor.")

    def get_feature_names(self, include_target: bool = False):
        names = []
        for section in ['numeric', 'categorical', 'ordinal']:
            names.extend(self.descriptor.get(section, {}).keys())

        if not include_target and self.class_name in names:
            names.remove(self.class_name)

        return names

    def get_number_of_features(self, include_target: bool = False):
        return len(self.get_feature_names(include_target=include_target))

    def get_feature_name(self, index: int):
        for section in ['numeric', 'categorical', 'ordinal']:
            for name, info in self.descriptor.get(section, {}).items():
                if info.get('index') == index:
                    return name
        raise IndexError(f"No feature found with index {index}")
