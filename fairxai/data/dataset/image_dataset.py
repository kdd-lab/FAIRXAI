import os
from typing import List, Union, Optional, Tuple, Dict

import numpy as np
from PIL import Image

from fairxai.data.dataset import Dataset
from fairxai.data.descriptor.image_descriptor import ImageDatasetDescriptor
from fairxai.logger import logger


class ImageDataset(Dataset):
    """
    Represents an image dataset that can be loaded either from a folder
    containing image files or directly from in-memory NumPy arrays.

    The dataset supports two serialization modes:

    - **Folder-based dataset**:
      Only the folder path is serialized. Images are reloaded at project load time.

    - **Memory-based dataset**:
      Raw NumPy arrays are saved to a compressed ``.npz`` file inside the project folder.
      This allows reconstruction of datasets not tied to an external file system.

    Parameters
    ----------
    data : str | np.ndarray | list[np.ndarray]
        Either a folder path or image arrays.
    class_name : str | None, optional
        Optional class label for the dataset.

    Raises
    ------
    TypeError
        If ``data`` is not a supported type.
    ValueError
        If no valid images can be loaded.
    """

    def __init__(
            self,
            data: Union[str, np.ndarray, List[np.ndarray]],
            class_name: Optional[str] = None
    ) -> None:

        super().__init__(data=None, class_name=class_name)

        # Tracks how the dataset was created ("folder" or "memory").
        self.source_type: str

        # Path of the folder, only for folder-mode datasets.
        self.folder_path: Optional[str] = None

        # Filenames corresponding to loaded images (valid only for folder datasets).
        self.filenames: List[str] = []

        # ---------------------------------------------------------
        # FOLDER MODE: load images from a directory
        # ---------------------------------------------------------
        if isinstance(data, str):
            self.source_type = "folder"
            self.folder_path = data

            logger.info(f"Loading images from folder: {data}")
            self.data, self.filenames = self._load_from_folder(data)

        # ---------------------------------------------------------
        # MEMORY MODE: single NumPy array
        # ---------------------------------------------------------
        elif isinstance(data, np.ndarray):
            self.source_type = "memory"
            self.data = [data]
            self.filenames = []

        # ---------------------------------------------------------
        # MEMORY MODE: list of NumPy arrays
        # ---------------------------------------------------------
        elif isinstance(data, list) and all(isinstance(a, np.ndarray) for a in data):
            self.source_type = "memory"
            self.data = data
            self.filenames = []

        else:
            raise TypeError(
                "Data must be a folder path, a NumPy array, or a list of NumPy arrays."
            )

        # Build dataset descriptor
        try:
            self.update_descriptor()
        except ValueError as exc:
            logger.error(f"Error computing descriptor: {exc}")
            raise

    # ======================================================================
    # SERIALIZATION
    # ======================================================================

    def to_dict(self) -> Dict:
        """
        Serialize dataset metadata into a dictionary.

        Notes
        -----
        - Folder datasets store only ``folder_path``.
        - Memory datasets do not store raw image arrays here; arrays are saved
          separately via `save_memory_data`.

        Returns
        -------
        dict
            Metadata describing how to reconstruct the dataset.
        """
        return {
            "type": "image",
            "source_type": self.source_type,
            "folder_path": self.folder_path,
            "class_name": self.class_name,
        }

    @classmethod
    def from_dict(cls, meta: Dict, project_path: str) -> "ImageDataset":
        """
        Reconstruct an `ImageDataset` instance from serialized metadata.

        Parameters
        ----------
        meta : dict
            Serialized dataset information.
        project_path : str
            Filesystem path to the root of the project.

        Returns
        -------
        ImageDataset

        Raises
        ------
        FileNotFoundError
            If memory-based dataset arrays are missing.
        ValueError
            For unknown dataset source types.
        """
        source = meta["source_type"]

        # ---------------------------------------------------------
        # FOLDER MODE RECONSTRUCTION
        # ---------------------------------------------------------
        if source == "folder":
            return cls(
                data=meta["folder_path"],
                class_name=meta.get("class_name")
            )

        # ---------------------------------------------------------
        # MEMORY MODE RECONSTRUCTION (.npz)
        # ---------------------------------------------------------
        elif source == "memory":
            npz_path = os.path.join(project_path, "dataset", "images.npz")
            if not os.path.exists(npz_path):
                raise FileNotFoundError(f"Missing memory dataset file: {npz_path}")

            npz = np.load(npz_path)
            arrays = [npz[k] for k in npz]

            return cls(
                data=arrays,
                class_name=meta.get("class_name")
            )

        raise ValueError(f"Unknown dataset source type: {source}")

    # ======================================================================
    # SAVE MEMORY-BASED DATASETS
    # ======================================================================

    def save_memory_data(self, dest_folder: str) -> None:
        """
        Save memory-based image arrays into a compressed ``.npz`` file.

        Parameters
        ----------
        dest_folder : str
            Directory where the ``images.npz`` will be written.

        Notes
        -----
        If the dataset originates from a folder, this method does nothing.
        """
        if self.source_type != "memory":
            return

        os.makedirs(dest_folder, exist_ok=True)
        npz_path = os.path.join(dest_folder, "images.npz")

        # Save each image array under an integer key (0, 1, ...).
        np.savez_compressed(npz_path, *self.data)

        logger.info(f"Saved memory image arrays to {npz_path}")

    # ======================================================================
    # IMAGE LOADING UTILITIES
    # ======================================================================

    def _load_from_folder(
            self,
            folder_path: str,
            extensions: Optional[List[str]] = None,
            recursive: bool = True
    ) -> Tuple[List[np.ndarray], List[str]]:
        """
        Load all images inside a folder.

        Parameters
        ----------
        folder_path : str
            Path to the image folder.
        extensions : list[str], optional
            Allowed file extensions. Default is common image formats.
        recursive : bool
            If ``True``, walk directories recursively.

        Returns
        -------
        (list[np.ndarray], list[str])
            Loaded images and their corresponding filenames.

        Raises
        ------
        ValueError
            If no valid images can be loaded.
        """
        if extensions is None:
            extensions = [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]

        image_paths: List[str] = []

        # Collect image paths
        if recursive:
            for root, _, files in os.walk(folder_path):
                for filename in files:
                    if os.path.splitext(filename)[1].lower() in extensions:
                        image_paths.append(os.path.join(root, filename))
        else:
            for filename in os.listdir(folder_path):
                if os.path.splitext(filename)[1].lower() in extensions:
                    image_paths.append(os.path.join(folder_path, filename))

        if not image_paths:
            raise ValueError(f"No images found in folder: {folder_path}")

        images = []
        filenames = []

        # Load each image with fallback handling
        for path in image_paths:
            try:
                img = Image.open(path)
                images.append(np.array(img))
                filenames.append(os.path.basename(path))
            except Exception as exc:
                logger.warning(f"Skipping image {path}: {exc}")

        if not images:
            raise ValueError(
                f"No valid images could be loaded from folder: {folder_path}"
            )

        return images, filenames

    # ======================================================================
    # DESCRIPTOR + ACCESSORS
    # ======================================================================

    def update_descriptor(
            self,
            hwc_permutation: Optional[List[int]] = None
    ) -> Dict:
        """
        Compute and attach the dataset descriptor.

        Parameters
        ----------
        hwc_permutation : list[int] | None
            Optional permutation of axes (H, W, C).

        Returns
        -------
        dict
            The computed descriptor.
        """
        logger.info("Creating descriptor for image dataset")
        descriptor = ImageDatasetDescriptor(self.data).describe(
            hwc_permutation=hwc_permutation
        )
        self.set_descriptor(descriptor)
        return descriptor

    def get_instance(self, key: Union[int, str]) -> np.ndarray:
        """
        Retrieve a single image instance either by index or filename.

        Parameters
        ----------
        key : int | str
            Integer index or filename.

        Returns
        -------
        np.ndarray
            The requested image.

        Raises
        ------
        IndexError
            If index is out of range.
        ValueError
            If filename lookup fails or filenames are unavailable.
        TypeError
            If key is neither int nor str.
        """
        if isinstance(key, int):
            if key < 0 or key >= len(self.data):
                raise IndexError(f"Index {key} is out of range.")
            return self.data[key]

        if isinstance(key, str):
            if not self.filenames:
                raise ValueError("No filenames available for lookup.")

            try:
                idx = self.filenames.index(key)
                return self.data[idx]
            except ValueError:
                raise ValueError(f"Filename '{key}' not found in dataset.")

        raise TypeError("Key must be an integer index or filename string.")
