from numpy import number
from pandas import DataFrame, Series

from fairxai.logger import logger


class TabularDatasetDescriptor:
    """
    Handles the description of a tabular dataset by categorizing its columns into
    categorical, ordinal, and numeric types and providing summary statistics.

    This class requires **explicit declaration** of all non-numeric columns through
    the `categorical_columns` and `ordinal_columns` parameters. Columns not listed
    there and not recognized as numeric (based on their dtype) will raise a
    `ValueError` during the description process.

    It provides methods to describe the dataset, retrieve
    specific column types, and export the computed descriptions as a dictionary.

    Attributes:
        data (DataFrame): The main tabular dataset for analysis.
        categorical_columns (list): A list of column names which are considered
            categorical variables.
        ordinal_columns (list): A list of column names which are considered ordinal
            variables.

    Methods:
        describe():
            Describes the dataset by categorizing its columns and computing summary
            statistics for each type.

        get_numeric_columns():
            Returns the names of numeric columns.

        get_categorical_columns():
            Returns the list of categorical column names.

        get_ordinal_columns():
            Retrieves the list of ordinal column names.
    """

    def __init__(self, data: DataFrame, categorical_columns: list = None, ordinal_columns: list = None):
        self.data = data
        self.categorical_columns = categorical_columns or []
        self.ordinal_columns = ordinal_columns or []

        # internal state
        self.numeric = {}
        self.categorical = {}
        self.ordinal = {}

    def describe(self) -> dict:
        """
        Analyzes and describes each column in the dataset, categorizing them into
        numerical, categorical, and ordinal types, and provides detailed summaries
        of each type. The method iterates through all columns in the dataset,
        determines their types, and organizes their descriptions accordingly.

        Returns the summarized descriptions as a dictionary format.

        Raises:
            ValueError: Raised when an error occurs during column type determination, typically
                due to issues in data processing or invalid dataset configuration.


        Returns:
            dict: A dictionary containing structured descriptions of numeric, categorical,
                and ordinal features extracted from the dataset.
        """
        df = self.data
        self.numeric.clear()
        self.categorical.clear()
        self.ordinal.clear()

        try:

            for feature in df.columns:
                index = df.columns.get_loc(feature)
                col_type = self._get_column_type(feature, df)

                if col_type == 'categorical':
                    self.categorical[feature] = self._create_categorical_description(df[feature], index)
                elif col_type == 'ordinal':
                    self.ordinal[feature] = self._create_categorical_description(df[feature], index)
                else:
                    self.numeric[feature] = self._create_numeric_description(df[feature], index)
        except ValueError as e:
            logger.error(f"Error during column type determination: {e}")
            raise ValueError(e)

        return self._as_dict()

    def _get_column_type(self, feature: str, df: DataFrame) -> str:
        """
        Determines the type of column (categorical, ordinal, or numeric).

        Args:
            feature: Column name
            df: DataFrame containing the data

        Returns:
            Column type as string
        """
        if feature in self.categorical_columns:
            return "categorical"
        elif feature in self.ordinal_columns:
            return "ordinal"
        elif feature in df.select_dtypes(include=number).columns.tolist():
            return "numeric"
        else:
            raise ValueError(f"Unknown column type for column '{feature}'.")

    def _create_categorical_description(self, series: Series, index: int) -> dict:
        """
        Creates the description for a categorical or ordinal column.

        Args:
            series: Pandas series containing the column data
            index: Column index in the DataFrame

        Returns:
            Dictionary with column statistics
        """
        unique_values = series.unique()
        return {
            'index': index,
            'distinct_values': list(unique_values),
            'count': {x: int((series == x).sum()) for x in unique_values}
        }

    def _create_numeric_description(self, series: Series, index: int) -> dict:
        """
        Creates the description for a numeric column.

        Args:
            series: Pandas series containing the column data
            index: Column index in the DataFrame

        Returns:
            Dictionary with column statistics
        """
        return {
            'index': index,
            'min': series.min(),
            'max': series.max(),
            'mean': series.mean(),
            'std': series.std(),
            'median': series.median(),
            'q1': series.quantile(0.25),
            'q3': series.quantile(0.75)
        }

    # --- Utility methods ---
    def get_numeric_columns(self):
        """
        Returns the names of numeric columns.

        Returns:
            list: A list containing the names of numeric columns.
        """
        return list(self.numeric.keys())

    def get_categorical_columns(self):
        """
        Returns the list of categorical column names.

        Returns:
            List[str]: A list containing the names of categorical columns.
        """
        return list(self.categorical.keys())

    def get_ordinal_columns(self):
        """
        Retrieves the list of ordinal column names.

        Returns:
            list: A list of column names corresponding to ordinal data.
        """
        return list(self.ordinal.keys())

    def _as_dict(self):
        """
        Converts the object's attributes to a dictionary representation.

        Returns:
            dict: A dictionary containing the object's attributes mapped to
            their corresponding values.
        """
        return {
            'numeric': self.numeric,
            'categorical': self.categorical,
            'ordinal': self.ordinal
        }
