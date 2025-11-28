from __future__ import annotations
import os
from typing import Optional, List, Union, Dict, Any
import pandas as pd
from pandas import DataFrame, Series
import numpy as np

from fairxai.data.dataset import Dataset
from fairxai.data.descriptor.tabular_descriptor import TabularDatasetDescriptor
from fairxai.logger import logger


class TabularDataset(Dataset):
    """
    Tabular dataset container for explainers.

    Supports initialization from:
        - pandas DataFrame
        - CSV file path
        - dict or list of dict

    The dataset supports two serialization modes:

    - **CSV-based dataset**:
      Only the CSV path is stored. Data is reloaded from the original CSV on load.

    - **Memory-based dataset**:
      Raw DataFrame is saved as CSV inside the project folder for persistence.

    Features and target are separated:
        - self._data: features-only DataFrame
        - self._target: target Series (if class_name provided)
    """

    def __init__(
        self,
        data: Union[DataFrame, str, dict, List[dict]],
        class_name: Optional[str] = None,
        categorical_columns: Optional[List[str]] = None,
        ordinal_columns: Optional[List[str]] = None,
        dropna: bool = False
    ) -> None:
        """
        Initialize TabularDataset.

        Parameters
        ----------
        data : DataFrame | str | dict | list[dict]
            Source data
        class_name : str | None
            Target column name, if present
        categorical_columns : list[str] | None
            Column names to treat as categorical
        ordinal_columns : list[str] | None
            Column names to treat as ordinal
        dropna : bool
            If reading CSV, drop rows with missing values
        """
        super().__init__(data, class_name)

        self.source_type: str  # "csv" or "memory"
        self.csv_path: Optional[str] = None
        self._categorical_columns: List[str] = list(categorical_columns) if categorical_columns else []
        self._ordinal_columns: List[str] = list(ordinal_columns) if ordinal_columns else []
        self.class_name: Optional[str] = class_name

        # --------------------------
        # CSV-based dataset
        # --------------------------
        if isinstance(data, str) and os.path.exists(data) and data.lower().endswith(".csv"):
            self.source_type = "csv"
            self.csv_path = data
            df = pd.read_csv(data, skipinitialspace=True, na_values="?", keep_default_na=True)
            if dropna:
                df = df.dropna()

        # --------------------------
        # Memory-based dataset
        # --------------------------
        # FIXME: qui devo chiamare save_memory_data(), ma devo trovare un modo per passargli il path giusto!
        elif isinstance(data, pd.DataFrame):
            self.source_type = "memory"
            df = data
        elif isinstance(data, dict):
            self.source_type = "memory"
            df = pd.DataFrame(data)
        elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            self.source_type = "memory"
            df = pd.DataFrame(data)
        else:
            raise TypeError(
                f"Unsupported data type for TabularDataset: {type(data)}. "
                "Supported: DataFrame, CSV path, dict, list[dict]."
            )

        # --------------------------
        # Extract target
        # --------------------------
        if self.class_name is not None and self.class_name in df.columns:
            self._target: Optional[Series] = df[self.class_name].copy()
            df = df.drop(columns=[self.class_name])
        else:
            self._target = None

        self._data: DataFrame = df

        # Remove target from hints if present
        self._sanitize_column_hints()

        # Compute descriptor
        try:
            self.update_descriptor()
        except ValueError as e:
            logger.error(f"Error computing descriptor: {e}")
            raise

    # --------------------------
    # Serialization
    # --------------------------
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize dataset metadata for project persistence.

        Returns
        -------
        dict
            Metadata describing dataset type, source, and column hints.
        """
        return {
            "type": "tabular",
            "source_type": self.source_type,
            "csv_path": self.csv_path,
            "class_name": self.class_name,
            "categorical_columns": self._categorical_columns,
            "ordinal_columns": self._ordinal_columns,
            "columns": list(self._data.columns)
        }

    @classmethod
    def from_dict(cls, meta: Dict[str, Any], project_path: Optional[str] = None) -> TabularDataset:
        """
        Reconstruct a TabularDataset from metadata.

        Parameters
        ----------
        meta : dict
            Serialized metadata
        project_path : str | None
            Project folder path; used to locate memory-based CSV if needed

        Returns
        -------
        TabularDataset
        """
        source = meta["source_type"]

        if source == "csv":
            return cls(data=meta["csv_path"],
                       class_name=meta.get("class_name"),
                       categorical_columns=meta.get("categorical_columns"),
                       ordinal_columns=meta.get("ordinal_columns"))
        elif source == "memory":
            # Memory-based dataset: load CSV inside project folder
            if project_path is None:
                raise ValueError("project_path must be provided to load memory-based dataset")
            csv_file = os.path.join(project_path, "dataset", "tabular.csv")
            if not os.path.exists(csv_file):
                raise FileNotFoundError(f"Missing memory dataset file: {csv_file}")
            df = pd.read_csv(csv_file)
            return cls(data=df,
                       class_name=meta.get("class_name"),
                       categorical_columns=meta.get("categorical_columns"),
                       ordinal_columns=meta.get("ordinal_columns"))
        else:
            raise ValueError(f"Unknown source_type: {source}")

    # --------------------------
    # Save memory-based dataset
    # --------------------------
    def save_memory_data(self, dest_folder: str) -> None:
        """
        Save memory-based dataset to CSV inside project folder.

        Parameters
        ----------
        dest_folder : str
            Destination folder path
        """
        if self.source_type != "memory":
            return

        os.makedirs(dest_folder, exist_ok=True)
        csv_file = os.path.join(dest_folder, "tabular.csv")
        self._data.to_csv(csv_file, index=False)
        # Also save target column separately if needed
        if self._target is not None and self.class_name:
            target_file = os.path.join(dest_folder, f"{self.class_name}.csv")
            self._target.to_csv(target_file, index=False)
        logger.info(f"Saved memory-based tabular dataset to {csv_file}")

    # --------------------------
    # Internal helpers
    # --------------------------
    def _sanitize_column_hints(self) -> None:
        """Remove target column from categorical/ordinal hints if present."""
        if self.class_name:
            if self.class_name in self._categorical_columns:
                logger.warning(
                    f"Target column '{self.class_name}' found in categorical_columns; removing it."
                )
                self._categorical_columns = [c for c in self._categorical_columns if c != self.class_name]
            if self.class_name in self._ordinal_columns:
                logger.warning(
                    f"Target column '{self.class_name}' found in ordinal_columns; removing it."
                )
                self._ordinal_columns = [c for c in self._ordinal_columns if c != self.class_name]

    # --------------------------
    # Descriptor
    # --------------------------
    def update_descriptor(
        self,
        categorical_columns: Optional[List[str]] = None,
        ordinal_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Compute dataset descriptor based on features-only DataFrame."""
        categorical_columns = categorical_columns or self._categorical_columns
        ordinal_columns = ordinal_columns or self._ordinal_columns

        # Keep only valid columns
        categorical_columns = [c for c in categorical_columns if c in self._data.columns]
        ordinal_columns = [c for c in ordinal_columns if c in self._data.columns]

        self.descriptor = TabularDatasetDescriptor(
            data=self._data,
            categorical_columns=categorical_columns,
            ordinal_columns=ordinal_columns
        ).describe(target=self._target, target_name=self.class_name)
        return self.descriptor

    # --------------------------
    # Indexing / length
    # --------------------------
    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> np.ndarray:
        """Return feature vector for index idx as 1-D numpy array."""
        row = self._data.iloc[idx]
        return row.values

    # --------------------------
    # Properties
    # --------------------------
    @property
    def X(self) -> DataFrame:
        return self._data

    @property
    def y(self) -> Optional[Series]:
        return self._target

    @property
    def target(self) -> Optional[Series]:
        return self._target

    @property
    def features(self) -> List[str]:
        """Return feature names according to descriptor."""
        if hasattr(self, "descriptor") and self.descriptor:
            names = []
            for section in ["numeric", "categorical", "ordinal"]:
                names.extend(self.descriptor.get(section, {}).keys())
            return names
        return list(self._data.columns)

    def get_feature_names(self) -> List[str]:
        return self.features

    def get_feature_name(self, index: int) -> str:
        for section in ["numeric", "categorical", "ordinal"]:
            for name, info in self.descriptor.get(section, {}).items():
                if info.get("index") == index:
                    return name
        # fallback
        cols = list(self._data.columns)
        if 0 <= index < len(cols):
            return cols[index]
        raise IndexError(f"No feature found with index {index}")

    def get_class_values(self) -> List[Any]:
        if not self.class_name:
            raise Exception("class_name is None.")
        if self._target is not None:
            return list(self._target.unique())
        raise KeyError(f"Target '{self.class_name}' not found in dataset.")

    def set_class_name(self, class_name: str) -> None:
        """Change target column name, moving column from features to target."""
        if class_name == self.class_name:
            return

        if class_name in self._data.columns:
            self._target = self._data[class_name].copy()
            self._data = self._data.drop(columns=[class_name])
            self.class_name = class_name
            self._sanitize_column_hints()
            self.update_descriptor()
            return

        raise KeyError(f"Column '{class_name}' not found among features.")
