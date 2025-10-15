from . import Dataset
from ..descriptor.text_descriptor import TextDatasetDescriptor
from ..logger import logger


class TextDataset(Dataset):
    """
    Represents a dataset containing textual data.

    This class is used to handle and manage a text-based dataset. It allows for the updating of a text dataset's
    descriptor, which provides metadata or characterization of the dataset. The class can optionally include
    a name for the dataset's classification purpose.

    Attributes:
        data: The raw textual data to be managed by the dataset.
        class_name: Optional name or label for categorizing the dataset.
        descriptor: Metadata descriptor of the text dataset, populated after invoking the `update_descriptor` method.

    Methods:
        update_descriptor():
            Updates or generates the descriptor for the dataset and returns the resulting descriptor value.
    """

    def __init__(self, data, class_name=None):
        """
        Initializes the instance of a class.

        Parameters:
        data (any): The data associated with the instance.
        class_name (Optional[str]): The name of the class, default is None.
        """
        super().__init__()
        self.data = data
        self.class_name = class_name
        self.descriptor = None

    def update_descriptor(self):
        """
        Updates the descriptor for the text dataset by creating a description
        using the TextDatasetDescriptor and assigning it to the descriptor
        attribute.

        Returns:
            The updated descriptor of the text dataset as created by
            TextDatasetDescriptor.
        """
        logger.info("description creation for text dataset")
        descriptor = TextDatasetDescriptor(self.data).describe()
        self.set_descriptor(descriptor)
        return self.descriptor
