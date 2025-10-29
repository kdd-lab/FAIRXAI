import pandas as pd
import pytest
from fairxai.data.descriptor.timeserie_descriptor import TimeSeriesDatasetDescriptor


def test_describe_with_empty_dataframe():
    """
    Tests the behavior of the `describe` method when called on a TimeSeriesDatasetDescriptor
    object initialized with an empty DataFrame.

    Attributes:
    data (pd.DataFrame): The input DataFrame used to initialize the descriptor. In this
        test case, it is an empty DataFrame.
    descriptor (TimeSeriesDatasetDescriptor): The object being tested, initialized with
        the provided data DataFrame.

    Raises:
    AssertionError: If the output dictionary from the `describe` method does not contain
        the expected values, such as "type" being "timeseries", "n_rows" being 0, and
        "n_series" being 1.
    """
    # Given
    data = pd.DataFrame()
    descriptor = TimeSeriesDatasetDescriptor(data)

    # When
    result = descriptor.describe()

    # Then
    assert result["type"] == "timeseries"
    assert result["n_rows"] == 0
    assert result["n_series"] == 1


def test_describe_with_non_dataframe_data():
    """
    Tests that the `describe` method raises a `TypeError` when the input data is not a pandas DataFrame.

    Raises:
        TypeError: If the input data to `TimeSeriesDatasetDescriptor` is not a pandas.DataFrame.
    """
    # Given
    data = []

    # When / Then
    with pytest.raises(TypeError, match="TimeSeriesDatasetDescriptor requires a pandas.DataFrame"):
        TimeSeriesDatasetDescriptor(data).describe()


def test_describe_with_data_with_id_column():
    """
    Tests the describe method of the TimeSeriesDatasetDescriptor class to
    ensure it accurately summarizes the properties of a dataset with an
    'id' column.

    Raises:
        AssertionError: If the description does not match the expected
        values in terms of dataset type, number of rows, and number of
        unique series.
    """
    # Given
    data = pd.DataFrame({"id": [1, 1, 2, 2], "value": [10, 20, 30, 40]})
    descriptor = TimeSeriesDatasetDescriptor(data)

    # When
    result = descriptor.describe()

    # Then
    assert result["type"] == "timeseries"
    assert result["n_rows"] == 4
    assert result["n_series"] == 2


def test_describe_with_data_with_id_and_timestamp_columns():
    """
    Tests the `describe` method of the `TimeSeriesDatasetDescriptor` class when
    provided with data that includes `id` and `timestamp` columns. It ensures that
    the method correctly computes and returns the time series type, total number
    of rows, number of unique series, and the range of timestamps within the data.

    Raises:
        AssertionError: If the assertions for resulting type, row count, series
        count, or timestamp range fail.
    """
    # Given
    data = pd.DataFrame({
        "id": [1, 1, 2, 2],
        "timestamp": ["2023-01-01", "2023-01-02", "2023-01-01", "2023-01-02"],
        "value": [10, 20, 30, 40]
    })
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    descriptor = TimeSeriesDatasetDescriptor(data)

    # When
    result = descriptor.describe()

    # Then
    assert result["type"] == "timeseries"
    assert result["n_rows"] == 4
    assert result["n_series"] == 2
    assert result["timestamps_range"] == "(2023-01-01 00:00:00, 2023-01-02 00:00:00)"


def test_describe_with_data_without_id_or_timestamp_columns():
    """
    Tests the functionality of describing a DataFrame without specific columns 'id'
    or 'timestamp' to check if it can be classified as a time series dataset.

    Raises:
        AssertionError: If the type is not classified as "timeseries", the number of
        rows is not 4, or the number of series is not 1.
    """
    # Given
    data = pd.DataFrame({"value": [10, 20, 30, 40]})
    descriptor = TimeSeriesDatasetDescriptor(data)

    # When
    result = descriptor.describe()

    # Then
    assert result["type"] == "timeseries"
    assert result["n_rows"] == 4
    assert result["n_series"] == 1
