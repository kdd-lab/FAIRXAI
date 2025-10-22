import pandas as pd

from .base_descriptor import BaseDatasetDescriptor


class TimeSeriesDatasetDescriptor(BaseDatasetDescriptor):
    """
    Descriptor for timeseries datasets.

    This class analyzes time series data stored in a pandas DataFrame and provides
    structured information about the dataset, including the number of series,
    total rows, and temporal range.
    """

    def describe(self) -> dict:
        """
        Analyzes the time series dataset and returns a description dictionary.

        Returns:
            dict: A dictionary containing:
                - type (str): Always "timeseries"
                - n_rows (int): Total number of rows in the dataset
                - n_series (int): Number of unique time series (based on 'id' column if present)
                - timestamps_range (tuple): Min and max timestamps (if 'timestamp' column exists)

        Raises:
            TypeError: If the data is not a pandas DataFrame
        """
        # Check that the data is a pandas DataFrame
        if not isinstance(self.data, pd.DataFrame):
            raise TypeError("TimeSeriesDatasetDescriptor richiede un pandas.DataFrame")

        # Initialize the description dictionary with basic information
        desc = {
            "type": "timeseries",
            "n_rows": len(self.data),
        }

        # Count the number of unique time series
        # If an 'id' column exists, count unique IDs; otherwise assume a single series
        if "id" in self.data.columns:
            desc["n_series"] = self.data["id"].nunique()
        else:
            desc["n_series"] = 1

        # Calculate the temporal range of the dataset
        # If a 'timestamp' column exists, store the min and max timestamps
        if "timestamp" in self.data.columns:
            min_timestamp = self.data["timestamp"].min()
            max_timestamp = self.data["timestamp"].max()
            desc["timestamps_range"] = f"({min_timestamp}, {max_timestamp})"

        return desc
