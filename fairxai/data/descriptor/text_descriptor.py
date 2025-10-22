from .base_descriptor import BaseDatasetDescriptor


class TextDatasetDescriptor(BaseDatasetDescriptor):
    """
    Descriptor for text datasets that analyzes and describes textual data.

    This class extends BaseDatasetDescriptor to provide specific functionality
    for text-based datasets, supporting both raw text strings and dictionary formats.
    """

    def describe(self) -> dict:
        """
        Analyzes the text dataset and returns a dictionary with descriptive information.

        Returns:
            dict: A dictionary containing:
                - type: Always "text"
                - n_documents: Total number of documents
                - input_format: Either "dict" or "raw_text"
                - Additional format-specific metadata

        Raises:
            ValueError: If the dataset is empty
            TypeError: If the data format is not supported (not string or dict)
        """
        # Retrieve data and count documents
        data = self.data
        n_docs = len(data)

        # Check that the dataset is not empty
        if n_docs == 0:
            raise ValueError("Empty dataset")

        # Analyze the first element to determine the format
        sample = data[0]
        desc = {"type": "text", "n_documents": n_docs}

        # Handle dictionary format
        if isinstance(sample, dict):
            keys = list(sample.keys())
            has_timestamp = "timestamp" in keys
            desc.update({
                "input_format": "dict",
                "structure": keys,  # List of keys in the dictionary
                "has_timestamp": has_timestamp  # Whether timestamp field exists
            })

        # Handle raw text format
        elif isinstance(sample, str):
            # Calculate the average length in words across all documents
            avg_len = sum(len(t.split()) for t in data) / n_docs
            desc.update({
                "input_format": "raw_text",
                "avg_length_words": avg_len
            })

        # Reject unsupported formats
        else:
            raise TypeError("Unsupported text format (use string or dict)")

        return desc
