import uuid
from datetime import datetime
from typing import List, Type, Dict, Any, Optional

from fairxai.bbox.bbox_factory import ModelFactory
from fairxai.data.dataset.dataset_factory import DatasetFactory
from fairxai.explain.adapter.generic_explainer_adapter import GenericExplainerAdapter
from fairxai.explain.explainer_manager.explainer_manager import ExplainerManager
from fairxai.logger import logger


class Project:
    """
    Manages explainability projects by integrating datasets, predictive models, and explainers.

    This class serves as a central management point for creating explainability projects. It facilitates the
    instantiation of datasets, models, and explainers, along with tracking the generated explanation results.
    Using this class, users can interact with various explainers and manage their outputs efficiently.

    Attributes:
        id: Unique identifier string for the project instance.
        created_at: Timestamp indicating when the instance was created.
        dataset_type: Defines the type of dataset used, such as tabular, text, or image.
        model_type: Specifies the predictive model utilized in the project.
        dataset_instance: Represents the dataset instance created based on provided data and type.
        blackbox: Machine learning model instance created with the specified parameters.
        explainer_manager: Manages the list and compatibility of explainers for the model and dataset type.
        explainers: A list of compatible explainers available for the dataset type and the predictive model.
        explanations: Stores generated explanation results as a list of dictionaries.

    Methods:
        get_explanations: Returns the list of stored explanation results.
        to_dict: Serializes the project instance for saving or exporting.
    """

    def __init__(
            self,
            data: Any,
            dataset_type: str,
            model_name: str,
            target_variable: Optional[str] = None,
            categorical_columns: Optional[List[str]] = None,
            ordinal_columns: Optional[List[str]] = None,
            model_params: Optional[Dict[str, Any]] = None
    ):
        """
        Initializes an instance of the class responsible for managing an explainability project. This includes
        setting up the dataset, the model, and associated explainers.

        Attributes:
            id: A unique identifier string for the project instance.
            created_at: The timestamp when the instance was created.
            dataset_type: The type of the dataset being used, such as tabular, text, or image.
            model_type: The name of the predictive model being utilized.
            dataset_instance: An instance of the dataset, created using the DatasetFactory.
            blackbox: A machine learning model object created through the ModelFactory.
            explainer_manager: An instance of ExplainerManager, used for listing and managing explainers.
            explainers: A list of explainers compatible with the provided dataset type and model.
            explanations: A list to store generated explanation results, where each explanation is represented
            as a dictionary.

        Parameters:
            data: The data to be used for creating the dataset instance. Can be of any type supported by the
            DatasetFactory.
            dataset_type: The type of the dataset, determining how it is processed.
            model_name: The name of the model to be created.
            target_variable: An optional string indicating the target variable for prediction.
            categorical_columns: An optional list of strings specifying the names of categorical columns in the
            dataset.
            ordinal_columns: An optional list of strings specifying the names of ordinal columns in the dataset.
            model_params: An optional dictionary containing parameters for initializing the model.
        """
        self.id = str(uuid.uuid4())
        self.created_at = datetime.now()
        self.dataset_type = dataset_type
        self.model_type = model_name
        self.dataset_instance = DatasetFactory.create(data, dataset_type, class_name=target_variable,
                                                      categorical_columns=categorical_columns,
                                                      ordinal_columns=ordinal_columns)
        self.blackbox = ModelFactory.create(model_name, model_params)
        self.explainer_manager = ExplainerManager(dataset_type, model_name)
        self.explainers = self.explainer_manager.list_available_compatible_explainers()
        self.explanations: List[Dict[str, Any]] = []

        logger.info(f"Project {self.id} created with {len(self.explainers)} explainers.")

    # -----------------------------
    # Private helper methods
    # -----------------------------

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
    # Project management
    # -----------------------------
    def get_compatible_explainers(self):
        """Returns a list of compatible explainers for the dataset and model."""
        return self.explainers

    def use_explainer(self, explainer_name: str):
        if not explainer_name in self.explainers:
            raise ValueError("Uncompatible explainer for this dataset and model.")

        explainer_instance = self.explainer_manager.create_explainer(explainer_name, self.blackbox,
                                                                     self.dataset_instance)

        # TODO: esegui spiegazione

    def get_explanations(self):
        """Returns the list of generated explanations."""
        pass

    def to_dict(self):
        """Serializes the project for saving or download."""
        return {
            "id": self.id,
            "created_at": str(self.created_at),
            "dataset": self.dataset_instance.__class__.__name__,
            "model": self.blackbox.__class__.__name__,
            "explainers": [e.__name__ for e in self.explainers],
            "num_explanations": len(self.explanations),
        }

    # TODO: import method for serialized project

    def __repr__(self):
        dataset_name = self.dataset_instance.__class__.__name__
        num_explainers = len(self.explainers)
        return f"<Project id={self.id}, dataset={dataset_name}, explainers={num_explainers}>"
