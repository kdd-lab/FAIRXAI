from . import Dataset
from ..descriptor.image_descriptor import ImageDatasetDescriptor
from ..logger import logger


class ImageDataset(Dataset):
    """
    Represents an image dataset.

    This class is used to manage and operate on datasets of images. It
    provides an interface to update and maintain descriptors for the
    dataset, which can be useful for various image processing tasks.
    """

    def __init__(self, data, class_name=None):
        """
        Represents an initialization of an object with data and an optional class name.

        Attributes:
            data: The main data associated with the object.
            class_name: An optional string that represents the class name associated with the data.

        """
        super().__init__()
        self.data = data
        self.class_name = class_name
        self.descriptor = None

    def update_descriptor(self):
        """
        Updates the descriptor for the image dataset after creating it using the
        ImageDatasetDescriptor. Logs the progress and sets the newly created
        descriptor as the current descriptor for the dataset.

        Returns:
            dict: The updated descriptor of the image dataset.
        """
        logger.info("Descriptor creation for image dataset")
        descriptor = ImageDatasetDescriptor(self.data).describe()
        self.set_descriptor(descriptor)
        return self.descriptor
