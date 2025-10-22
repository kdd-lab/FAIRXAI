import os

import numpy as np
from PIL import Image

from .base_descriptor import BaseDatasetDescriptor


class ImageDatasetDescriptor(BaseDatasetDescriptor):
    def describe(self) -> dict:
        """
        Analyzes and describes an image dataset.

        Returns a dictionary containing detailed information about the dataset,
        including the number of images, input format, resolution, and number of channels.

        Returns:
            dict: A dictionary with dataset description

        Raises:
            ValueError: If the dataset is empty
            TypeError: If the image format is not supported
        """
        # Count the total number of images in the dataset
        n_images = len(self.data)
        if n_images == 0:
            raise ValueError("No given images.")

        # Take the first image as a sample for analysis
        sample = self.data[0]

        # Initialize the description dictionary with basic information
        desc = {"type": "image", "n_samples": n_images}

        # Case 1: Image provided as file path (string)
        if isinstance(sample, str):
            # Open the image using PIL/Pillow
            img = Image.open(sample)
            desc.update({
                "input_format": "path",
                # Reverse img.size to get (Height, Width) instead of (Width, Height)
                "resolution": img.size[::-1],
                # Get the number of color channels (e.g., 3 for RGB, 1 for grayscale)
                "channels": len(img.getbands()),
                # Extract only the filename without the full path
                "sample_image": os.path.basename(sample)
            })

        # Case 2: Image provided as NumPy array
        elif isinstance(sample, np.ndarray):
            shape = sample.shape
            # FIXME: in una batch ho sempre in cima la batch, l'ordine non è fisso (dipende da come è stato codificato il dataset)
            desc.update({
                "input_format": "numpy",
                # First two dimensions are height and width
                "resolution": shape[:2],
                # Third dimension is channels if present, otherwise 1 (grayscale)
                "channels": shape[2] if len(shape) == 3 else 1
            })

        # Case 3: Unsupported format
        else:
            raise TypeError("Unsupported image format (use path or numpy array)")

        # Return the complete description dictionary
        return desc
