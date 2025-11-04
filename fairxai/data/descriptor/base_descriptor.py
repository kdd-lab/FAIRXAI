from abc import ABC, abstractmethod


class BaseDatasetDescriptor(ABC):
    """
    Represents an abstract base class for dataset descriptors.

    This class serves as a blueprint for dataset descriptor implementations,
    allowing for uniform representation of datasets. It enforces the
    implementation of a method to describe the dataset in a structured manner.
    """

    def __init__(self, data):
        """
        Initializes an object with the given data.

        The constructor method sets up the initial state of the object by assigning
        the provided data to the instance variable.

        Args:
            data: The data to be associated with this instance.
        """
        self.data = data

    @abstractmethod
    def describe(self) -> dict:
        """
        An abstract representation of a describable resource or entity. Classes inheriting
        from this should implement the `describe` method to provide a detailed representation
        of the resource as a dictionary.

        The describe method is intended to serve as a blueprint for outputting structured
        data about the implementing resource.

        Methods:
            - describe: Abstract method that must be implemented by subclasses to produce
              a description of the resource.

        """
        pass
