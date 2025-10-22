import numpy as np
from pandas import DataFrame

from fairxai.data.descriptor.base_descriptor import BaseDatasetDescriptor


class TabularDatasetDescriptor(BaseDatasetDescriptor):
    """
    Represents a descriptor for a tabular dataset, providing functionality
    to summarize and categorize dataset features, including numeric,
    categorical, and ordinal columns.

    This class is designed to process a dataset represented as a pandas
    DataFrame and classify its columns based on specified categorical or
    ordinal attribute lists. Additionally, it calculates statistics for
    numeric columns and generates a structured descriptor dictionary.

    Attributes:
        class_name: str or None
            The name of the target/class column, if defined.
        categorical_columns: list or None
            A list of column names explicitly defined as categorical.
        ordinal_columns: list or None
            A list of column names explicitly defined as ordinal.

    Methods:
        describe():
            Summarizes the dataset by categorizing and computing statistics
            for numeric, categorical, and ordinal columns.
    """

    def __init__(self, data: DataFrame, class_name: str = None,
                 categorical_columns: list = None, ordinal_columns: list = None):
        # Call the parent class constructor with the data
        super().__init__(data)
        # Store the name of the target/class column
        self.class_name = class_name
        # Store the list of categorical column names
        self.categorical_columns = categorical_columns
        # Store the list of ordinal column names
        self.ordinal_columns = ordinal_columns

    def describe(self) -> dict:
        """
        Generates a detailed description of the dataset by categorizing columns
        into numeric, categorical, or ordinal types and computing statistical
        or frequency-based metadata for each feature. This method processes all
        columns within the dataset and applies appropriate descriptive analysis
        based on the column's category or data type.

        Returns:
            dict: A comprehensive dictionary containing detailed descriptions
            for each column categorized into 'numeric', 'categorical', and
            'ordinal', along with their respective statistics such as min, max,
            mean, standard deviation, quartiles, distinct values, and value
            counts where applicable.

        Raises:
            Exception: If an unknown error occurs during the writing of the
            descriptor or the processing of columns.
        """
        # Create a reference to the DataFrame for cleaner code
        df = self.data
        # Initialize descriptor dictionary with empty categories
        descriptor = {'numeric': {}, 'categorical': {}, 'ordinal': {}}

        # Iterate through each column in the DataFrame
        for feature in df.columns:
            # Get the positional index of the current column
            index = df.columns.get_loc(feature)

            # CATEGORICAL
            # Check if the feature is explicitly defined as categorical
            if self.categorical_columns and feature in self.categorical_columns:
                desc = {
                    'index': index,
                    # Get all unique values in this column
                    'distinct_values': list(df[feature].unique()),
                    # Count occurrences of each unique value
                    'count': {x: len(df[df[feature] == x]) for x in df[feature].unique()}
                }
                descriptor['categorical'][feature] = desc

            # ORDINAL
            # Check if the feature is explicitly defined as ordinal
            elif self.ordinal_columns and feature in self.ordinal_columns:
                desc = {
                    'index': index,
                    # Get all unique values in this column
                    'distinct_values': list(df[feature].unique()),
                    # Count occurrences of each unique value
                    'count': {x: len(df[df[feature] == x]) for x in df[feature].unique()}
                }
                descriptor['ordinal'][feature] = desc

            # NUMERIC
            # Check if the feature has a numeric data type (uses NumPy number types)
            elif feature in df.select_dtypes(include=np.number).columns.tolist():
                desc = {
                    'index': index,
                    # Calculate minimum value
                    'min': df[feature].min(),
                    # Calculate maximum value
                    'max': df[feature].max(),
                    # Calculate mean (average)
                    'mean': df[feature].mean(),
                    # Calculate standard deviation
                    'std': df[feature].std(),
                    # Calculate median (50th percentile)
                    'median': df[feature].median(),
                    # Calculate first quartile (25th percentile)
                    'q1': df[feature].quantile(0.25),
                    # Calculate third quartile (75th percentile)
                    'q3': df[feature].quantile(0.75)
                }
                descriptor['numeric'][feature] = desc

            # OTHERWISE: fallback to categorical
            # If none of the above conditions match, treat as categorical
            else:
                desc = {
                    'index': index,
                    # Get all unique values in this column
                    'distinct_values': list(df[feature].unique()),
                    # Count occurrences of each unique value
                    'count': {x: len(df[df[feature] == x]) for x in df[feature].unique()}
                }
                descriptor['categorical'][feature] = desc

        # Gestione target
        # Process and move the target column to its own category
        descriptor = self._set_target_label(descriptor)
        return descriptor

    def _set_target_label(self, descriptor: dict) -> dict:
        """
        Sets the target label in the provided descriptor based on the specified target class name.
        This method identifies the target column in the input descriptor and moves it from its original
        category to a new "target" category. If the target column is not found, the method returns the
        descriptor unchanged. It raises an exception if the target column is in the "numeric" category.

        Parameters:
        descriptor (dict): A dictionary representing the metadata about features, categorized under
        numeric, categorical, or ordinal types.

        Returns:
        dict: The updated descriptor where the target column is categorized under the "target" key or
        the original descriptor if the target column is not found.

        Raises:
        Exception: If the target column is continuous (numeric) and not categorical or ordinal.
        """
        # If no class name is specified, return descriptor unchanged
        if not self.class_name:
            return descriptor

        # Iterate through each category (numeric, categorical, ordinal)
        for category in descriptor:
            # Iterate through features in the current category
            for feature in list(descriptor[category].keys()):
                # Check if this feature is the target column
                if feature == self.class_name:
                    # Prevent numeric columns from being used as target
                    if category == "numeric":
                        raise Exception(
                            "ERR: target column cannot be continuous. Please, set a categorical column as target. "
                            "You can discretize the target to force it."
                        )
                    # Create a separate "target" entry in the descriptor
                    descriptor["target"] = {feature: descriptor[category][feature]}
                    # Remove the target from its original category
                    descriptor[category].pop(feature)
                    return descriptor

        # Return descriptor if target column was not found
        return descriptor
