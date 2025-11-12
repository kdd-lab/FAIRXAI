from abc import ABC, abstractmethod

from fairxai.explain.explaination.generic_explanation import GenericExplanation
from fairxai.logger import logger


class GenericExplainerAdapter(ABC):
    """
    Abstract base class for implementing generic explainer adapters.

    This class provides a standardized structure for creating explanation strategies
    for machine learning models. It defines abstract methods for generating instance-specific
    (local) explanations and global explanations of a model. It also includes methods
    for determining compatibility with specific datasets and models and building a
    generic explanation structure. The class serves as a framework for subclasses
    to implement their own logic for interpretable machine learning purposes.
    """

    # TODO: GradCam, LORE, SHAP (NTH: PivotTree, Abele)

    # Constants for explainer names and compatibility
    explainer_name = "generic"
    supported_datasets = []
    supported_models = []

    # Constants for explanation types (aligned with GenericExplanation)
    LOCAL_EXPLANATION = "local"
    GLOBAL_EXPLANATION = "global"

    # Constant for universal compatibility
    WILDCARD = "*"

    def __init__(self, model, dataset):
        """
        Initializes the instance with the given machine learning model and dataset.

        Attributes:
        model (Any): The machine learning model used for explanation.
        dataset (Any): The dataset associated with the explanations.

        Args:
        model: The machine learning model to be explained.
        dataset: The dataset on which the model operates.
        """
        self.model = model
        self.dataset = dataset
        logger.debug(
            f"{self.explainer_name} initialized with {dataset.__class__.__name__}"
        )

    # ---------------------------
    # Abstract methods to implement
    # ---------------------------

    @abstractmethod
    def explain_instance(self, instance):
        """
        Represents an abstract method to explain a specific instance of data.

        Methods:
            explain_instance: Abstract method to be implemented by subclasses, used
            to explain or provide details about a specific instance.

        Args:
            instance: The instance of data that needs to be explained. Its specifics
            depend on the implementing subclass.
        """
        pass

    @abstractmethod
    def explain_global(self):
        """
        An abstract method to provide global interpretation or explanation for a model's predictions.
        This method is part of an interpretability framework, ensuring that all implementing
        classes define their own logic for global explanation.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        pass

    # ---------------------------
    # Compatibility methods
    # ---------------------------

    @classmethod
    def is_compatible(cls, dataset_type: str, model_type: str) -> bool:
        """
        Checks compatibility between the dataset type and model type.

        Detailed evaluation to determine whether the provided dataset type and model type
        are supported and compatible with the current implementation of the class. This is
        evaluated based on internal compatibility checks for both dataset and model types.

        Args:
            dataset_type: The type of the dataset to be analyzed for compatibility.
            model_type: The type of the model to be analyzed for compatibility.

        Returns:
            A boolean value indicating whether the provided dataset type and model type
            are compatible.
        """
        is_dataset_compatible = cls._is_dataset_compatible(dataset_type)
        is_model_compatible = cls._is_model_compatible(model_type)
        return is_dataset_compatible and is_model_compatible

    @classmethod
    def _is_dataset_compatible(cls, dataset_type: str) -> bool:
        """
        Determines whether a given dataset type is compatible with the class.

        This method checks if the provided dataset type is supported by the
        class based on its `supported_datasets` attribute or if the `WILDCARD`
        placeholder is present in the supported datasets.

        Parameters:
        dataset_type: str
            The type of the dataset to check compatibility for.

        Returns:
        bool
            True if the dataset type is compatible, False otherwise.
        """
        return dataset_type in cls.supported_datasets or cls.WILDCARD in cls.supported_datasets

    @classmethod
    def _is_model_compatible(cls, model_type: str) -> bool:
        """
        Checks if the given model type is compatible with the class.

        This method verifies whether the provided model type is supported
        by the class, either explicitly or through a wildcard definition.

        Args:
            model_type: A string representing the type of the model to be checked.

        Returns:
            bool: True if the model type is compatible, otherwise False.
        """
        return model_type in cls.supported_models or cls.WILDCARD in cls.supported_models
