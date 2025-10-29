import numpy as np
import pytest
from fairxai.data.descriptor.image_descriptor import ImageDatasetDescriptor


def test_describe_with_empty_data():
    # Given
    data = []
    descriptor = ImageDatasetDescriptor(data)
    # When / Then
    with pytest.raises(ValueError, match="No given images."):
        descriptor.describe()


def test_describe_with_image_path():
    # Given
    image_path = "tests/data/sample_image.png"
    data = [image_path]
    descriptor = ImageDatasetDescriptor(data)
    # When
    description = descriptor.describe()
    # Then
    assert description["type"] == "image"
    assert description["n_samples"] == 1
    assert description["input_format"] == "path"
    assert description["sample_image"] == "sample_image.png"


def test_describe_with_numpy_array_grayscale():
    # Given
    data = [np.random.rand(256, 256)]
    descriptor = ImageDatasetDescriptor(data)
    # When
    description = descriptor.describe()
    # Then
    assert description["type"] == "image"
    assert description["n_samples"] == 1
    assert description["input_format"] == "numpy"
    assert description["resolution"] == (256, 256)
    assert description["channels"] == 1


def test_describe_with_numpy_array_rgb():
    # Given
    data = [np.random.rand(128, 128, 3)]
    descriptor = ImageDatasetDescriptor(data)
    # When
    description = descriptor.describe()
    # Then
    assert description["type"] == "image"
    assert description["n_samples"] == 1
    assert description["input_format"] == "numpy"
    assert description["resolution"] == (128, 128)
    assert description["channels"] == 3


def test_describe_with_unsupported_format():
    # Given
    data = [42]
    descriptor = ImageDatasetDescriptor(data)
    # When / Then
    with pytest.raises(TypeError, match="Unsupported image format \\(use path or numpy array\\)"):
        descriptor.describe()
