import uuid
from datetime import datetime
from typing import List, Type, Dict, Any

from fairxai.bbox import AbstractBBox
from fairxai.data.dataset import Dataset
from fairxai.explain.adapter.generic_explainer_adapter import GenericExplainerAdapter
from fairxai.logger import logger


class Project:
    """
    Container class for Dataset, BlackBox, and Explainable AI pipeline.
    Each Project is identified by a unique ID and tracks generated explanations.

    Responsibilities:
    - Manage dataset, model, and explainers
    - Ensure compatibility between explainer and dataset/model
    - Track explanations (linked to dataset, model, and instance)
    """

    def __init__(
            self,
            dataset: Dataset,
            blackbox: AbstractBBox,
            explainers: List[Type[GenericExplainerAdapter]]
    ):
        self.id = str(uuid.uuid4())
        self.created_at = datetime.now()
        self.dataset = dataset
        self.blackbox = blackbox
        self.explainers = explainers
        self.explanations: List[Dict[str, Any]] = []

        logger.info(f"Project {self.id} created with {len(explainers)} explainers.")

    # -----------------------------
    # Private helper methods
    # -----------------------------

    def _get_dataset_type(self) -> str:
        """Returns the dataset type."""
        return getattr(self.dataset, "type", self.dataset.__class__.__name__.lower())

    def _get_model_type(self) -> str:
        """Returns the model type."""
        return getattr(self.blackbox, "type", "unknown")

    def _serialize_instance(self, instance) -> Any:
        """Serializes an instance for recording."""
        return instance.to_dict() if hasattr(instance, "to_dict") else instance

    def _create_explanation_record(
            self,
            explainer_cls: Type[GenericExplainerAdapter],
            instance,
            explanation
    ) -> Dict[str, Any]:
        """Creates an explanation record for history tracking."""
        return {
            "timestamp": datetime.now(),
            "explainer": explainer_cls.__name__,
            "instance": self._serialize_instance(instance),
            "result": explanation.to_dict(),
        }

    # -----------------------------
    # Compatibility and management
    # -----------------------------

    def get_compatible_explainers(self) -> List[Type[GenericExplainerAdapter]]:
        """
        Returns the list of explainers compatible with the dataset and model.
        """
        dataset_type = self._get_dataset_type()
        model_type = self._get_model_type()

        compatible_explainers = [
            expl for expl in self.explainers
            if expl.is_compatible(dataset_type, model_type)
        ]

        logger.debug(
            f"Found {len(compatible_explainers)} compatible explainers "
            f"for {dataset_type}/{model_type}"
        )
        return compatible_explainers

    def ensure_compatibility(self, explainer_cls: Type[GenericExplainerAdapter]):
        """
        Raises an exception if the explainer is not compatible.
        """
        dataset_type = self._get_dataset_type()
        model_type = self._get_model_type()

        if not explainer_cls.is_compatible(dataset_type, model_type):
            explainer_name = explainer_cls.__name__
            error_message = (
                f"Explainer {explainer_name} is not compatible with "
                f"dataset '{dataset_type}' and model '{model_type}'."
            )
            logger.error(error_message)
            raise ValueError(error_message)

    # -----------------------------
    # Explanation execution
    # -----------------------------

    def run_explanation(self, explainer_cls: Type[GenericExplainerAdapter], instance) -> Any:
        """
        Executes a compatible explainer on the instance and records the result.
        """
        self.ensure_compatibility(explainer_cls)

        explainer = explainer_cls(self.blackbox, self.dataset)
        explanation = explainer.explain_instance(instance)

        explanation_record = self._create_explanation_record(explainer_cls, instance, explanation)
        self.explanations.append(explanation_record)

        logger.info(f"Explanation from {explainer_cls.__name__} added to project {self.id}.")
        return explanation

    # -----------------------------
    # Project management
    # -----------------------------

    def get_explanations(self):
        """Returns the list of generated explanations."""
        return self.explanations

    def to_dict(self):
        """Serializes the project for saving or download."""
        return {
            "id": self.id,
            "created_at": str(self.created_at),
            "dataset": self.dataset.__class__.__name__,
            "model": self.blackbox.__class__.__name__,
            "explainers": [e.__name__ for e in self.explainers],
            "num_explanations": len(self.explanations),
        }

    def __repr__(self):
        dataset_name = self.dataset.__class__.__name__
        num_explainers = len(self.explainers)
        return f"<Project id={self.id}, dataset={dataset_name}, explainers={num_explainers}>"
