from abc import abstractmethod




class Dataset():
    """
    Abstract class for datasets
    """
    @abstractmethod
    def set_dataset_type(self):
        """
        it sets the dataset type: Tabular | Image | Text | Timeseries
        """
