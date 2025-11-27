from typing import List, Union, Optional
import os
from PIL import Image
import numpy as np
from .dataset import Dataset
from fairxai.logger import logger
from ..descriptor.image_descriptor import ImageDatasetDescriptor


class ImageDataset(Dataset):
    """
    Represents an image dataset with images ready to be passed to a model.

    Automatically handles:
        - Folder path containing images
        - Single NumPy array
        - List of NumPy arrays

    Maintains a mapping from filenames to arrays for retrieval by name.
    """

    def __init__(
        self,
        data: Union[str, np.ndarray, List[np.ndarray]],
        class_name: Optional[str] = None
    ):
        """
        Initialize the ImageDataset.

        :param data: Folder path, single array, or list of arrays
        :param class_name: Optional target class name
        """
        super().__init__(data=None, class_name=class_name)
        self.filenames: List[str] = []

        if isinstance(data, str):
            # Load images from folder
            logger.info(f"Loading images from folder: {data}")
            self.data, self.filenames = self._load_from_folder(data)
        elif isinstance(data, np.ndarray):
            self.data = [data]
            self.filenames = []
        elif isinstance(data, list) and all(isinstance(a, np.ndarray) for a in data):
            self.data = data
            self.filenames = []
        else:
            raise TypeError("Data must be a folder path, a NumPy array, or a list of arrays.")

        try:
            self.update_descriptor()
        except ValueError as e:
            logger.error(f"Error computing descriptor: {e}")
            raise

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------
    def _load_from_folder(
        self,
        folder_path: str,
        extensions: Optional[List[str]] = None,
        recursive: bool = True
    ) -> (List[np.ndarray], List[str]):
        """
        Load all images from a folder into NumPy arrays and store filenames.

        :param folder_path: Path to folder
        :param extensions: List of allowed file extensions
        :param recursive: Include subfolders if True
        :return: Tuple (images as arrays, filenames)
        """
        if extensions is None:
            extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']

        image_paths = []
        if recursive:
            for root, _, files in os.walk(folder_path):
                for f in files:
                    if os.path.splitext(f)[1].lower() in extensions:
                        image_paths.append(os.path.join(root, f))
        else:
            for f in os.listdir(folder_path):
                if os.path.splitext(f)[1].lower() in extensions:
                    image_paths.append(os.path.join(folder_path, f))

        if not image_paths:
            raise ValueError(f"No images found in folder: {folder_path}")

        images: List[np.ndarray] = []
        filenames: List[str] = []

        for path in image_paths:
            try:
                img = Image.open(path)  # preserve original channels
                img_array = np.array(img)
                images.append(img_array)
                filenames.append(os.path.basename(path))
            except Exception as e:
                logger.warning(f"Skipping image {path}: {e}")

        if not images:
            raise ValueError(f"No valid images could be loaded from folder: {folder_path}")

        return images, filenames

    # -------------------------------------------------------------------------
    # Descriptor
    # -------------------------------------------------------------------------
    def update_descriptor(self, hwc_permutation: Optional[List[int]] = None) -> dict:
        """
        Update the dataset descriptor using ImageDatasetDescriptor.

        :param hwc_permutation: Optional permutation of dimensions expected by the model
        :return: Dictionary containing dataset description
        """
        logger.info("Creating descriptor for image dataset")
        descriptor = ImageDatasetDescriptor(self.data).describe(hwc_permutation=hwc_permutation)
        self.set_descriptor(descriptor)
        return descriptor

    # -------------------------------------------------------------------------
    # Retrieve specific instance
    # -------------------------------------------------------------------------
    def get_instance(self, key: Union[int, str]) -> np.ndarray:
        """
        Retrieve an image from the dataset by index or filename.

        :param key: Index (int) or filename (str)
        :return: NumPy array of the image
        :raises IndexError: If index is out of range
        :raises ValueError: If filename is not found
        """
        if isinstance(key, int):
            if key < 0 or key >= len(self.data):
                raise IndexError(f"Index {key} is out of range.")
            return self.data[key]

        elif isinstance(key, str):
            if not self.filenames:
                raise ValueError("No filenames available for lookup.")
            try:
                idx = self.filenames.index(key)
                return self.data[idx]
            except ValueError:
                raise ValueError(f"Filename {key} not found in dataset.")
        else:
            raise TypeError("Key must be an integer index or filename string.")
