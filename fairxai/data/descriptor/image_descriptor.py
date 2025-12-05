import os
from typing import List, Union, Optional

import numpy as np
from PIL import Image

from fairxai.data.descriptor.base_descriptor import BaseDatasetDescriptor


class ImageDatasetDescriptor(BaseDatasetDescriptor):
    """
    Descriptor for image datasets.

    Analyzes a dataset composed of NumPy arrays or image file paths,
    providing metadata including number of samples, resolution, number of channels,
    and optional model input shape via hwc_permutation.
    """

    def __init__(self, data: List[Union[str, np.ndarray]]):
        """
        Initialize the descriptor with dataset data.

        :param data: List of image file paths or NumPy arrays
        :type data: list[Union[str, np.ndarray]]
        """
        super().__init__(data)

    def describe(self, hwc_permutation: Optional[List[int]] = None) -> dict:
        """
        Analyze and describe the dataset.

        :param hwc_permutation: Optional permutation of dimensions expected by the model (e.g., [1,2,0])
        :type hwc_permutation: list[int], optional
        :return: Dictionary containing dataset description
        :rtype: dict
        :raises ValueError: If dataset is empty or permutation is invalid
        :raises TypeError: If dataset contains unsupported types
        """
        n_images = len(self.data)
        if n_images == 0:
            raise ValueError("No images available to describe.")

        sample = self.data[0]  # Take first image or path for metadata
        desc = {"type": "image", "n_samples": n_images}

        if isinstance(sample, str):
            # Case: image path
            try:
                img = Image.open(sample)  # preserve original channels
            except Exception as e:
                raise ValueError(f"Cannot open image file {sample}: {e}")

            resolution = img.size[::-1]  # (H, W)
            channels = len(img.getbands())
            desc.update({
                "input_format": "path",
                "resolution": resolution,
                "channels": channels,
                "sample_image": os.path.basename(sample),
                "original_shape": resolution + (channels,)
            })

        elif isinstance(sample, np.ndarray):
            # Case: NumPy array
            shape = sample.shape
            if len(shape) == 2:
                h, w = shape
                c = 1
            elif len(shape) == 3:
                h, w, c = shape
            else:
                raise ValueError(f"Unsupported array shape: {shape}")

            desc.update({
                "input_format": "numpy",
                "resolution": (h, w),
                "channels": c,
                "sample_image": sample,
                "original_shape": shape
            })

        else:
            raise TypeError("Unsupported image format (use path or NumPy array)")

        # Include optional hwc_permutation and compute expected model shape
        if hwc_permutation is not None:
            desc["hwc_permutation"] = hwc_permutation
            try:
                desc["model_expected_shape"] = tuple(desc["original_shape"][i] for i in hwc_permutation)
            except IndexError:
                raise ValueError(f"Invalid hwc_permutation {hwc_permutation} for shape {desc['original_shape']}")

        return desc
