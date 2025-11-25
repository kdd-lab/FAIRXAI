import os
from typing import Optional, List, Any, Union

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from fairxai.data.dataset import Dataset
from fairxai.data.descriptor.tabular_descriptor import TabularDatasetDescriptor
from fairxai.logger import logger


class TabularDataset(Dataset):
    """
    Tabular dataset container used as the single reference dataset for explainers.

    Key characteristics / design decisions:
      - Accepts DataFrame, CSV path, dict, or list[dict] as input.
      - Extracts the target (if provided) into `self._target` and removes it
        from `self._data` so that `self._data` always contains only feature columns.
      - Descriptor is computed on features-only DataFrame to provide stable
        feature indices for explainers.
      - Implements __len__ and __getitem__ to be indexable by integer.
      - __getitem__ returns a 1-D numpy array (feature vector), which is broadly
        compatible with explainer adapters that call `np.asarray(instance)`.
    """

    def __init__(self, data: Union[DataFrame, str, dict, List[dict]], class_name: Optional[str] = None,
                 categorical_columns: Optional[List[str]] = None, ordinal_columns: Optional[List[str]] = None,
                 dropna: bool = False):
        """
        Initialize TabularDataset.

        Args:
            data: pandas.DataFrame | path-to-csv | dict | list[dict]
            class_name: optional target column name; if provided and present,
                        target will be extracted into self._target and removed
                        from the features DataFrame.
            categorical_columns: optional list of column names to treat as categorical
            ordinal_columns: optional list of column names to treat as ordinal
            dropna: whether to drop rows with NA when loading from CSV (applies to CSV only)
        """
        # --- Normalize input into a pandas DataFrame ---
        super().__init__(data, class_name)
        if isinstance(data, pd.DataFrame):
            df = data
        elif isinstance(data, str) and os.path.exists(data) and data.lower().endswith(".csv"):
            df = pd.read_csv(data, skipinitialspace=True, na_values="?", keep_default_na=True)
            if dropna:
                df = df.dropna()
        elif isinstance(data, dict):
            df = pd.DataFrame(data)
        elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            df = pd.DataFrame(data)
        else:
            raise TypeError(
                f"Unsupported data type for TabularDataset: {type(data)}. "
                "Supported: DataFrame, CSV path, dict, list[dict]."
            )

        # Store passed column hints (but we'll sanitize later)
        self._categorical_columns = list(categorical_columns) if categorical_columns else []
        self._ordinal_columns = list(ordinal_columns) if ordinal_columns else []
        self.class_name = class_name  # keep original target name (may be None)

        # --- Extract target (if present) and keep features-only DataFrame ---
        if self.class_name is not None and self.class_name in df.columns:
            # Extract target series and remove column from df to keep feature columns stable
            self._target: Optional[Series] = df[self.class_name].copy()
            df = df.drop(columns=[self.class_name])
        else:
            self._target = None

        # Features-only DataFrame stored here
        self._data: DataFrame = df

        # compute descriptor on features-only DataFrame
        # But first sanitize categorical/ordinal hints: they must reference feature columns only
        self._sanitize_column_hints()
        try:
            self.update_descriptor()
        except ValueError as e:
            logger.error(f"Error computing descriptor: {e}")
            raise

    # -------------------------
    # Internal helpers
    # -------------------------
    def _sanitize_column_hints(self):
        """
        Remove target column from categorical/ordinal hints if present and log a warning.
        """
        if self.class_name:
            if self.class_name in self._categorical_columns:
                logger.warning(f"Target column '{self.class_name}' found in categorical_columns; removing it for descriptor.")
                self._categorical_columns = [c for c in self._categorical_columns if c != self.class_name]
            if self.class_name in self._ordinal_columns:
                logger.warning(f"Target column '{self.class_name}' found in ordinal_columns; removing it for descriptor.")
                self._ordinal_columns = [c for c in self._ordinal_columns if c != self.class_name]

    # -------------------------
    # Descriptor & metadata
    # -------------------------
    def update_descriptor(self, categorical_columns: Optional[List[str]] = None,
                          ordinal_columns: Optional[List[str]] = None):
        """
        Compute and set the dataset descriptor based on feature DataFrame.
        Adds an extra 'target' entry containing the target series if present.

        Args:
            categorical_columns: Optional list of categorical columns (feature-only)
            ordinal_columns: Optional list of ordinal columns (feature-only)

        Returns:
            descriptor dict
        """
        # Use internal hints if arguments not provided
        categorical_columns = categorical_columns if categorical_columns is not None else self._categorical_columns
        ordinal_columns = ordinal_columns if ordinal_columns is not None else self._ordinal_columns

        # Keep only columns actually in the features DataFrame
        categorical_columns = [c for c in (categorical_columns or []) if c in self._data.columns]
        ordinal_columns = [c for c in (ordinal_columns or []) if c in self._data.columns]

        # Compute descriptor on feature-only DataFrame
        descriptor = TabularDatasetDescriptor(
            data=self._data,
            categorical_columns=categorical_columns,
            ordinal_columns=ordinal_columns
        ).describe(target=self._target)

        return self.descriptor

    # -------------------------
    # Indexing & length
    # -------------------------
    def __len__(self):
        """Return number of feature rows."""
        return len(self._data)

    def __getitem__(self, idx: int):
        """
        Return feature vector for index `idx` as a 1-D numpy array.

        This format is broadly compatible with explainer adapters that call
        `np.asarray(instance)` or expect numeric arrays.
        """
        # allow negative indexing as pandas does
        row = self._data.iloc[idx]
        # return 1-D numpy array (values in column order)
        return row.values

    # -------------------------
    # Convenience properties
    # -------------------------
    @property
    def X(self) -> DataFrame:
        """Return the features-only DataFrame (unchanged)."""
        return self._data

    @property
    def y(self) -> Optional[Series]:
        """Return the target Series (if present), otherwise None."""
        return self._target

    @property
    def target(self) -> Optional[Series]:
        """Alias for y."""
        return self.y

    @property
    def features(self) -> List[str]:
        """Return feature names according to current descriptor (numeric+categorical+ordinal)."""
        # If descriptor exists, use its ordering; otherwise fallback to DataFrame columns
        if hasattr(self, "descriptor") and self.descriptor:
            names = []
            for section in ["numeric", "categorical", "ordinal"]:
                names.extend(self.descriptor.get(section, {}).keys())
            return names
        return list(self._data.columns)

    # -------------------------
    # Feature helpers
    # -------------------------
    def get_feature_names(self, include_target: bool = False):
        """
        Return list of feature names. include_target is ignored because target
        is stored separately and not part of features DataFrame.
        """
        return self.features

    def get_number_of_features(self, include_target: bool = False):
        return len(self.get_feature_names())

    def get_feature_name(self, index: int):
        """
        Retrieve a feature name by its index according to the descriptor.

        Raises:
            IndexError if not found.
        """
        for section in ["numeric", "categorical", "ordinal"]:
            for name, info in self.descriptor.get(section, {}).items():
                if info.get("index") == index:
                    return name
        # Fallback: if descriptor missing or index not found, map using DataFrame columns
        cols = list(self._data.columns)
        if 0 <= index < len(cols):
            return cols[index]
        raise IndexError(f"No feature found with index {index}")

    def get_class_values(self):
        """
        Return the list of distinct target values.

        Raises:
            Exception: if class_name is None or target missing.
        """
        if not self.class_name:
            raise Exception("ERR: class_name is None. Use set_class_name('<column name>') first.")

        if self._target is not None:
            if hasattr(self._target, "unique"):
                return list(self._target.unique())
            else:
                return list(set(self._target))

        raise KeyError(f"Target column '{self.class_name}' not found in dataset or descriptor.")

    # -------------------------
    # Mutators / utilities
    # -------------------------
    def set_class_name(self, class_name: str):
        """
        Set or change the dataset's class_name (target). This will:
          - move the existing column (if present) from features into target, or
          - if the column is not present in features but exists elsewhere, raise.
        NOTE: changing class_name after initialization is allowed but should be used with care.
        """
        if class_name == self.class_name:
            return

        # If new class exists in features, extract it
        if class_name in self._data.columns:
            self._target = self._data[class_name].copy()
            self._data = self._data.drop(columns=[class_name])
            self.class_name = class_name
            # sanitize hints and recompute descriptor
            self._sanitize_column_hints()
            self.update_descriptor()
            return

        raise KeyError(f"Column '{class_name}' not found among features; cannot set as target.")
