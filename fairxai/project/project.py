import json
import os
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Type

import yaml

from fairxai.bbox.bbox_factory import ModelFactory
from fairxai.data.dataset.dataset_factory import DatasetFactory
from fairxai.explain.adapter.generic_explainer_adapter import GenericExplainerAdapter
from fairxai.explain.explainer_manager.explainer_manager import ExplainerManager
from fairxai.logger import logger


class Project:
    """
    Represents an explainability project binding a dataset, a trained model,
    and a set of compatible explainers. Supports saving/loading pipelines,
    explanations, and project metadata in a dedicated workspace.
    """

    def __init__(
            self,
            data: Any,
            dataset_type: str,
            model_type: str,
            workspace_path: str,
            target_variable: Optional[str] = None,
            categorical_columns: Optional[List[str]] = None,
            ordinal_columns: Optional[List[str]] = None,
            model_params: Optional[Dict[str, Any]] = None,
            model_path: Optional[str] = None,
    ):
        """
        Initialize a new project with dataset, model, and workspace folders.

        Args:
            data: Raw dataset to create the dataset instance.
            dataset_type: Dataset type ('tabular', 'image', 'text', etc.).
            model_name: Model type name to instantiate via ModelFactory.
            workspace_path: Base folder for project workspace.
            target_variable, categorical_columns, ordinal_columns: optional dataset config.
            model_params: Optional parameters for model instantiation.
            model_path: Optional path to pre-trained model weights (pickle/pth).
        """
        self.id = str(uuid.uuid4())
        self.created_at = datetime.now()
        self.dataset_type = dataset_type
        self.model_type = model_type
        self.workspace_path = os.path.join(workspace_path, self.id)

        # Create the workspace structure
        os.makedirs(self.workspace_path, exist_ok=True)
        os.makedirs(os.path.join(self.workspace_path, "results"), exist_ok=True)
        os.makedirs(os.path.join(self.workspace_path, "pipelines"), exist_ok=True)
        os.makedirs(os.path.join(self.workspace_path, "logs"), exist_ok=True)

        # Build dataset and model instances
        self.dataset_instance = DatasetFactory.create(
            data,
            dataset_type,
            class_name=target_variable,
            categorical_columns=categorical_columns,
            ordinal_columns=ordinal_columns,
        )
        self.blackbox = ModelFactory.create(model_type, model_params, model_path=model_path)

        # Discover compatible explainers for this dataset/model pair
        self.explainer_manager = ExplainerManager(dataset_type, model_type)
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
                    f"and model '{self.model_type}'. Available: {list(explainer_map.keys())}"
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
                explanation = explainer.explain_instance(instance)
            else:  # global
                instance_index = None
                instance = None
                explanation = explainer.explain_global()

            # Build and save record
            record = self._create_explanation_record(
                explainer_cls, mode, instance_index, instance, explanation
            )
            self.explanations.append(record)
            self._save_result(record)
            results.append(record)

        self._save_metadata()
        return results

    def run_pipeline_from_yaml(self, yaml_path: str) -> List[Dict[str, Any]]:
        """
        Load pipeline definition from YAML and execute it.

        YAML expected schema:
          pipeline:
            - explainer: "ShapExplainerAdapter"
              mode: "local"
              params:
                instance_index: 3
            - explainer: "LoreExplainerAdapter"
              mode: local
              params:
                strategy: "genetic"
            - explainer: "SomeGlobalExplainer"
              mode: "global"

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
    # Helpers: record building & persistence
    # -----------------------------
    def _create_explanation_record(
            self,
            explainer_cls: Type[GenericExplainerAdapter],
            mode: str,
            instance_index: Optional[int],
            instance,
            explanation,
    ) -> Dict[str, Any]:
        return {
            "timestamp": datetime.now().isoformat(),
            "explainer": explainer_cls.__name__,
            "mode": mode,
            "instance_index": instance_index,
            "instance": self._serialize_instance(instance),
            "result": getattr(explanation, "to_dict", lambda: explanation)(),
        }

    @staticmethod
    def _serialize_instance(instance) -> Any:
        if instance is None:
            return None
        if hasattr(instance, "to_dict"):
            return instance.to_dict()
        return str(instance)

    def _save_result(self, record: Dict[str, Any]) -> None:
        results_dir = os.path.join(self.workspace_path, "results")
        os.makedirs(results_dir, exist_ok=True)
        ts = record["timestamp"].replace(":", "_")
        filename = f"{ts}_{record['explainer']}_{record['mode']}.json"
        path = os.path.join(results_dir, filename)
        with open(path, "w") as f:
            json.dump(record, f, indent=2, default=str)
        logger.debug(f"Saved explanation result to {path}")

    def _save_metadata(self) -> None:
        metadata = self.to_dict()
        path = os.path.join(self.workspace_path, "project.json")
        with open(path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

    # -----------------------------
    # Serialization / utilities
    # -----------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "dataset_type": self.dataset_type,
            "model_type": self.model_type,
            "workspace_path": self.workspace_path,
            "num_explainers": len(self.explainers),
            "num_explanations": len(self.explanations),
        }

    @classmethod
    def load_from_dict(cls, data: Dict[str, Any]) -> "Project":
        """
        Restore a project from metadata dict.
        """
        project = cls.__new__(cls)
        project.id = data["id"]
        project.created_at = datetime.fromisoformat(data["created_at"])
        project.dataset_type = data["dataset_type"]
        project.model_type = data["model_type"]
        project.workspace_path = data["workspace_path"]

        project.dataset_instance = DatasetFactory.create(data=None, dataset_type=project.dataset_type)
        project.blackbox = ModelFactory.create(project.model_type, model_params=None)
        project.explainer_manager = ExplainerManager(project.dataset_type, project.model_type)
        project.explainers = project.explainer_manager.list_available_compatible_explainers()
        project.explanations = []

        return project

    def __repr__(self):
        return f"<Project id={self.id}, dataset={self.dataset_type}, model={self.model_type}>"
