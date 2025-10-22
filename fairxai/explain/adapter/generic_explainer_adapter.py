from abc import ABC, abstractmethod

from fairxai.explain.explaination.generic_explanation import GenericExplanation
from fairxai.logger import logger


class GenericExplainerAdapter(ABC):
    """
    Generic interface for all explainer adapters.

    Each adapter must adapt an external explainer (e.g., LORE, LIME, SHAP)
    to a common interface, providing standardized methods for generating
    local and global explanations.

    Attributes
    ----------
    explainer_name : str
        Identifier name of the explainer.
    supported_datasets : list
        List of supported dataset types. Use "*" to support all types.
    supported_models : list
        List of supported model types. Use "*" to support all types.
    model : object
        Machine learning model to be explained.
    dataset : object
        Dataset used for explanations.
    """

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
        Initialize the adapter with a model and a dataset.

        Parameters
        ----------
        model : object
            Machine learning model to be explained.
        dataset : object
            Dataset used to generate explanations.
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
        Generate an explanation for a single instance.

        Parameters
        ----------
        instance : object
            The instance to be explained.

        Returns
        -------
        GenericExplanation
            Object containing the instance explanation.
        """
        pass

    @abstractmethod
    def explain_global(self):
        """
        Generate a global explanation of the model.

        Returns
        -------
        GenericExplanation
            Object containing the global model explanation.
        """
        pass

    # ---------------------------
    # Compatibility methods
    # ---------------------------

    @classmethod
    def is_compatible(cls, dataset_type: str, model_type: str) -> bool:
        """
        Verify compatibility between dataset type and model type.

        Parameters
        ----------
        dataset_type : str
            Dataset type to verify.
        model_type : str
            Model type to verify.

        Returns
        -------
        bool
            True if both dataset and model are compatible, False otherwise.
        """
        is_dataset_compatible = cls._is_dataset_compatible(dataset_type)
        is_model_compatible = cls._is_model_compatible(model_type)
        return is_dataset_compatible and is_model_compatible

    @classmethod
    def _is_dataset_compatible(cls, dataset_type: str) -> bool:
        """
        Check if the dataset type is supported.

        Parameters
        ----------
        dataset_type : str
            Dataset type to verify.

        Returns
        -------
        bool
            True if the dataset is supported, False otherwise.
        """
        return dataset_type in cls.supported_datasets or cls.WILDCARD in cls.supported_datasets

    @classmethod
    def _is_model_compatible(cls, model_type: str) -> bool:
        """
        Check if the model type is supported.

        Parameters
        ----------
        model_type : str
            Model type to verify.

        Returns
        -------
        bool
            True if the model is supported, False otherwise.
        """
        return model_type in cls.supported_models or cls.WILDCARD in cls.supported_models

    # ---------------------------
    # Utility methods
    # ---------------------------

    def build_generic_explanation(
            self, data: dict, explanation_type: str = LOCAL_EXPLANATION
    ) -> GenericExplanation:
        """
        Build a GenericExplanation object from raw explainer data.

        Parameters
        ----------
        data : dict
            Dictionary containing the explanation data.
        explanation_type : str, optional
            Type of explanation: "local" or "global" (default: "local").

        Returns
        -------
        GenericExplanation
            Object containing the formatted explanation.
        """
        return GenericExplanation(
            explainer_name=self.explainer_name,
            explanation_type=explanation_type,
            data=data
        )