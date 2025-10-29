import pytest
from unittest.mock import patch, MagicMock

from fairxai.data.dataset.image_dataset import ImageDataset
from fairxai.data.descriptor.image_descriptor import ImageDatasetDescriptor
from fairxai.logger import logger


class TestImageDataset:

    def test_update_descriptor_given_valid_data_when_update_descriptor_called_then_descriptor_is_updated(self):
        # GIVEN
        data = ["image1.png", "image2.png"]
        dataset = ImageDataset(data=data, class_name="cats")
        mock_descriptor_output = {"num_images": 2, "avg_size": (128, 128)}

        # Patching ImageDatasetDescriptor.describe()
        with patch.object(ImageDatasetDescriptor, 'describe', return_value=mock_descriptor_output) as mock_describe, \
                patch.object(logger, 'info') as mock_logger:
            # WHEN
            result = dataset.update_descriptor()

            # THEN
            mock_logger.assert_called_once_with("Descriptor creation for image dataset")
            mock_describe.assert_called_once()
            assert dataset.descriptor == mock_descriptor_output
            assert result == mock_descriptor_output

    def test_init_given_data_and_class_name_when_initialized_then_attributes_are_set_correctly(self):
        # GIVEN
        data = ["img1.png"]
        class_name = "dogs"

        # WHEN
        dataset = ImageDataset(data=data, class_name=class_name)

        # THEN
        assert dataset.data == data
        assert dataset.class_name == class_name
        assert dataset.descriptor is None
