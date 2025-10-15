import pandas as pd
from pandas import DataFrame

from dataset import Dataset
from ..descriptor.tabular_descriptor import TabularDatasetDescriptor
from ..logger import logger

__all__ = ["TabularDataset", "Dataset"]


class TabularDataset(Dataset):
    """
    Represents a tabular dataset with features, class labels, and metadata descriptors.

    This class is designed to provide functionality for manipulating, describing,
    and working with tabular datasets, including specifying categorical and
    ordinal features. It supports creation from various data sources and includes
    methods to manage metadata about the dataset.

    Attributes:
        class_name: The name of the column containing class labels in the dataset.
        df: A pandas DataFrame containing the dataset's features and class labels.
        descriptor: A dictionary describing metadata of the dataset, including
            numeric, categorical, and ordinal features.

    Methods:
        update_descriptor(categorical_columns, ordinal_columns):
            Updates the descriptor for the dataset with given metadata.
        from_csv(filename, class_name, dropna):
            Creates a TabularDataset object from a CSV file.
        from_dict(data, class_name):
            Creates a TabularDataset object from a dictionary of data.
        set_class_name(class_name):
            Sets the name of the column containing class labels.
        get_class_values():
            Returns the values of the class label column.
        get_numeric_columns():
            Retrieves the names of numeric columns in the dataset.
        get_features_names():
            Returns a list of the feature names in the dataset.
        get_feature_name(index):
            Retrieves the feature name corresponding to the given index.
    """

    def __init__(self, data: DataFrame, class_name: str = None,
                 categorical_columns: list = None, ordinal_columns: list = None):
        """
        Initializes the class with the provided dataset and configuration for handling
        categorical, ordinal, and class columns.

        Attributes:
            class_name: str
                The name of the target column in the dataset, if applicable.
            df: DataFrame
                The dataset, with the target column moved to the last position if
                class_name is specified.
            descriptor: dict
                A dictionary to describe dataset column types, categorized into
                'numeric', 'categorical', and 'ordinal'.

        Arguments:
            data: DataFrame
                The input dataset to be handled.
            class_name: str, optional
                The name of the target column to be moved to the end of the dataset.
            categorical_columns: list, optional
                A list specifying which columns in the dataset are categorical.
            ordinal_columns: list, optional
                A list specifying which columns in the dataset are ordinal.
        """
        self.class_name = class_name
        self.df = data

        # move the target column to the end of the dataset for feature - target separation
        if class_name is not None:
            self.df = self.df[[c for c in self.df.columns if c != class_name] + [class_name]]

        self.descriptor = {'numeric': {}, 'categorical': {}, 'ordinal': {}}
        self.update_descriptor(categorical_columns=categorical_columns, ordinal_columns=ordinal_columns)

    def update_descriptor(self, categorical_columns: list = None, ordinal_columns: list = None):
        """
        Updates the descriptor for the tabular dataset based on defined categorical and ordinal columns.

        This method logs the process of creating the descriptor, utilizes the `TabularDatasetDescriptor`
        to generate the descriptor, and assigns the resulting descriptor to the instance.

        Args:
            categorical_columns: List of column names to be treated as categorical, or None.
            ordinal_columns: List of column names to be treated as ordinal, or None.

        Returns:
            dict: The updated descriptor generated for the tabular dataset.
        """
        logger.info("Descriptor creation for tabular dataset")

        descriptor = TabularDatasetDescriptor(
            self.df,
            class_name=self.class_name,
            categorical_columns=categorical_columns,
            ordinal_columns=ordinal_columns
        ).describe()

        self.descriptor = descriptor
        return self.descriptor

    @classmethod
    def from_csv(cls, filename: str, class_name: str = None, dropna: bool = True):
        """
        Read a comma-separated values (csv) file into Dataset object.
        :param [str] filename:
        :param class_name: optional
        :return:
        """
        df = pd.read_csv(filename, skipinitialspace=True, na_values='?', keep_default_na=True)
        if dropna:
            df.dropna(inplace=True)
        # check if the class_name correspond to a categorical column
        if class_name in df.select_dtypes(include=[np.number]).columns:
            # force the column to be categorical
            df[class_name] = df[class_name].astype(str)

        dataset_obj = cls(df, class_name=class_name)
        dataset_obj.filename = filename
        logger.info('{0} file imported'.format(filename))
        return dataset_obj

    @classmethod
    def from_dict(cls, data: dict, class_name: str = None):
        """
        From dicts of Series, arrays, or dicts.
        :param [dict] data:
        :param class_name: optional
        :return:
        """
        return cls(pd.DataFrame(data), class_name=class_name)

    def set_class_name(self, class_name: str):
        """
        Set the class name. Only the column name string
        :param [str] class_name:
        :return:
        """
        self.class_name = class_name

    def get_class_values(self):
        """
        Provides the class_name
        :return:
        """
        if self.class_name is None:
            raise Exception("ERR: class_name is None. Set class_name with set_class_name('<column name>')")
        return self.df[self.class_name].values

    def get_numeric_columns(self):
        numeric_columns = list(self.df._get_numeric_data().columns)
        return numeric_columns

    def get_features_names(self):
        return list(self.df.columns)

    def get_feature_name(self, index):
        for category in self.descriptor.keys():
            for name in self.descriptor[category].keys():
                if self.descriptor[category][name]['index'] == index:
                    return name
