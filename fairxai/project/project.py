import json
import os
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Type

import yaml

from fairxai.bbox.bbox_factory import ModelFactory
from fairxai.data.dataset.dataset_factory import DatasetFactory
from fairxai.explain.adapter.generic_explainer_adapter import GenericExplainerAdapter
from fairxai.explain.explaination.generic_explanation import GenericExplanation
from fairxai.explain.explainer_manager.explainer_manager import ExplainerManager
from fairxai.logger import logger


class Project:
    """
    Represents an explainability project binding a dataset, a trained model,
    and a set of compatible explainers. Supports saving/loading pipelines,
    explanations, and project metadata in a dedicated workspace.

    This version uses the "framework-first" ModelFactory approach, where
    the user specifies the ML framework (sklearn, torch, etc.) and optionally
    a model file path. The correct BBox wrapper is automatically instantiated
    and loaded.
    """

    def __init__(
        self,
        project_name: str,
        data: Any,
        dataset_type: str,
        framework: str,
        model_path: Optional[str] = None,
        model_params: Optional[Dict[str, Any]] = None,
        workspace_path: str = "./workspace",
        target_variable: Optional[str] = None,
        categorical_columns: Optional[List[str]] = None,
        ordinal_columns: Optional[List[str]] = None,
        device: str = "cpu"
    ):
        """
        Initialize a new project with dataset, model, and workspace folders.

        Args:
            data: Raw dataset to create the dataset instance.
            dataset_type: Dataset type ('tabular', 'image', 'text', etc.).
            framework: ML framework of the model ('sklearn', 'torch', etc.).
            model_path: Optional path to pre-trained model file (pickle/pth).
            model_params: Optional parameters for model instantiation.
            workspace_path: Base folder for project workspace.
            target_variable: Name of target variable in dataset (optional).
            categorical_columns: List of categorical columns (optional).
            ordinal_columns: List of ordinal columns (optional).
            device: Device for Torch models ('cpu' or 'cuda').
        """
        self.id = str(uuid.uuid4())
        self.created_at = datetime.now()
        self.dataset_type = dataset_type
        self.framework = framework
        self.project_name = project_name

        # Create workspace structure
        self.workspace_path = os.path.join(workspace_path, self.id)
        os.makedirs(self.workspace_path, exist_ok=True)
        os.makedirs(os.path.join(self.workspace_path, "results"), exist_ok=True)
        os.makedirs(os.path.join(self.workspace_path, "pipelines"), exist_ok=True)
        os.makedirs(os.path.join(self.workspace_path, "logs"), exist_ok=True)

        # Create dataset instance
        self.dataset_instance = DatasetFactory.create(
            data,
            dataset_type,
            class_name=target_variable,
            categorical_columns=categorical_columns,
            ordinal_columns=ordinal_columns,
        )

        # Create BBox wrapper using ModelFactory (handles file loading)
        self.blackbox = ModelFactory.create(
            framework=framework,
            model_path=model_path,
            model_params=model_params,
            device=device
        )

        # Store model type for logging / metadata
        self.model_type = type(self.blackbox.model).__name__

        logger.info(f"Project {self.id} initialized with dataset {self.dataset_type} and model {self.model_type}.")

        # Discover compatible explainers for this dataset/model pair
        self.explainer_manager = ExplainerManager(self.dataset_type, self.model_type)
        self.explainers: List[Type[GenericExplainerAdapter]] = (
            self.explainer_manager.list_available_compatible_explainers()
        )

        # Container for explanation records
        self.explanations: List[Dict[str, Any]] = []

        # Persist metadata immediately
        self._save_metadata()
        logger.info(f"Project {self.id} initialized with {len(self.explainers)} compatible explainers.")

    # -----------------------------
    # Pipeline execution
    # -----------------------------
    def run_explanation_pipeline(self, pipeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Execute a user-defined explainability pipeline.

        Each step should include:
            - "explainer": str (explainer class name)
            - "mode": "local" | "global" (optional; inferred if missing)
            - "params": dict (optional; 'local' requires "instance_index")

        Returns:
            List of explanation records (dicts) for each pipeline step.
        """
        results: List[Dict[str, Any]] = []
        explainer_map = {cls.__name__.lower(): cls for cls in self.explainers}

        for step in pipeline:
            explainer_name = step.get("explainer")
            if not explainer_name:
                raise ValueError("Pipeline step missing 'explainer' field")

            mode = step.get("mode", "").lower()
            params = step.get("params", {})
            if not mode:
                mode = "local" if "instance_index" in params else "global"
            if mode not in ("local", "global"):
                raise ValueError(f"Invalid mode '{mode}' for explainer '{explainer_name}'")

            explainer_key = explainer_name.lower()
            if explainer_key not in explainer_map:
                raise ValueError(
                    f"Explainer '{explainer_name}' not compatible for dataset '{self.dataset_type}' "
                    f"and framework '{self.framework}'. Available: {list(explainer_map.keys())}"
                )
            explainer_cls = explainer_map[explainer_key]
            explainer: GenericExplainerAdapter = explainer_cls(model=self.blackbox, dataset=self.dataset_instance)

            if mode == "local":
                if "instance_index" not in params:
                    raise ValueError("Local explanation requires 'instance_index'")
                instance_index = params["instance_index"]
                try:
                    instance = self.dataset_instance[instance_index]
                except Exception as e:
                    raise ValueError(f"Invalid instance_index {instance_index}: {e}")
                # An explainer could return multiple types of explanations for a single instance
                explanations_list: list[GenericExplanation] = explainer.explain_instance(instance)
            else:  # global explanation
                instance_index = None
                instance = None
                explanations_list: list[GenericExplanation] = explainer.explain_global()

            # Build and save record
            record = self._create_explanation_record(explainer_cls, mode, instance_index, instance, explanations_list)
            self.explanations.append(record)
            self._save_result(record)
            results.append(record)

        self._save_metadata()
        return results

    def run_pipeline_from_yaml(self, yaml_path: str) -> List[Dict[str, Any]]:
        """
        Load pipeline definition from YAML and execute it.

        Args:
            yaml_path: Path to YAML file containing pipeline specification.

        Returns:
            List of explanation records from executed pipeline.
        """
        yaml_path = os.path.expanduser(yaml_path)
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Pipeline YAML not found: {yaml_path}")

        with open(yaml_path, "r") as f:
            content = yaml.safe_load(f) or {}

        pipeline = content.get("pipeline")
        if pipeline is None:
            raise KeyError("YAML must contain top-level 'pipeline' key")

        # Save YAML copy in project workspace
        saved_pipeline_path = os.path.join(self.workspace_path, "pipelines", os.path.basename(yaml_path))
        with open(saved_pipeline_path, "w") as out:
            json.dump(content, out, indent=2, default=str)

        return self.run_explanation_pipeline(pipeline)

    # -----------------------------
    # Serialization / persistence
    # -----------------------------
    def _create_explanation_record(
        self,
        explainer_cls: Type[GenericExplainerAdapter],
        mode: str,
        instance_index: Optional[int],
        instance,
        explanations_list: List[GenericExplanation],
    ) -> Dict[str, Any]:
        """
        Build a dictionary record of a single explanation for storage/logging.
        """
        return {
            "timestamp": datetime.now().isoformat(),
            "explainer": explainer_cls.__name__,
            "mode": mode,
            "instance_index": instance_index,
            "instance": self._serialize_instance(instance),
            "result": [explanation.visualize() for explanation in explanations_list]
        }

    @staticmethod
    def _serialize_instance(instance) -> Any:
        """
        Serialize an instance for storage. Uses `to_dict()` if available.
        """
        if instance is None:
            return None
        if hasattr(instance, "to_dict"):
            return instance.to_dict()
        return str(instance)

    def _save_result(self, record: Dict[str, Any]) -> None:
        """
        Save an explanation record as JSON in the results folder.
        """
        results_dir = os.path.join(self.workspace_path, "results")
        os.makedirs(results_dir, exist_ok=True)
        ts = record["timestamp"].replace(":", "_")
        filename = f"{ts}_{record['explainer']}_{record['mode']}.json"
        path = os.path.join(results_dir, filename)
        with open(path, "w") as f:
            json.dump(record, f, indent=2, default=str)
        logger.debug(f"Saved explanation result to {path}")

    def _save_metadata(self) -> None:
        """
        Persist project metadata to JSON in the workspace.
        """
        metadata = self.to_dict()
        path = os.path.join(self.workspace_path, "project.json")
        with open(path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

    def to_dict(self) -> Dict[str, Any]:
        """
        Return a serializable dictionary of project metadata.
        """
        return {
            "project_name": self.project_name,
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "dataset_type": self.dataset_type,
            "framework": self.framework,
            "model_type": self.model_type,
            "workspace_path": self.workspace_path,
            "num_explainers": len(self.explainers),
            "num_explanations": len(self.explanations),
        }

    @classmethod
    def load_from_dict(
        cls,
        data: Dict[str, Any],
        framework: str,
        model_path: Optional[str] = None,
        model_params: Optional[Dict[str, Any]] = None,
        device: str = "cpu"
    ) -> "Project":
        """
        Restore a project from metadata dictionary.

        Args:
            data: Project metadata dictionary.
            framework: ML framework of the model ('sklearn', 'torch', etc.).
            model_path: Optional path to model file (pickle/pth).
            model_params: Optional parameters for model instantiation.
            device: Torch device if applicable.

        Returns:
            Project instance with dataset and blackbox reconstructed.
        """
        project = cls.__new__(cls)
        project.id = data["id"]
        project.project_name = data["project_name"]
        project.created_at = datetime.fromisoformat(data["created_at"])
        project.dataset_type = data["dataset_type"]
        project.framework = framework

        # Rebuild workspace path
        project.workspace_path = data["workspace_path"]

        # Reconstruct dataset and model
        project.dataset_instance = DatasetFactory.create(data=None, dataset_type=project.dataset_type)
        project.blackbox = ModelFactory.create(
            framework=framework,
            model_path=model_path,
            model_params=model_params,
            device=device
        )
        project.model_type = type(project.blackbox.model).__name__

        # Re-discover compatible explainers
        project.explainer_manager = ExplainerManager(project.dataset_type, project.framework)
        project.explainers = project.explainer_manager.list_available_compatible_explainers()
        project.explanations = []

        return project

    def __repr__(self):
        return f"Project id={self.id}, name={self.project_name}, dataset={self.dataset_type}, framework={self.framework}, model={self.model_type}>"
