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
    Represents a single explainability project.

    The Project stores dataset and model instances, discovers compatible explainers,
    and can execute user-defined pipelines of explainers (programmatic lists or YAML).
    """

    def __init__(
            self,
            data: Any,
            dataset_type: str,
            model_name: str,
            workspace_path: str,
            target_variable: Optional[str] = None,
            categorical_columns: Optional[List[str]] = None,
            ordinal_columns: Optional[List[str]] = None,
            model_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a new project and set up workspace folders.

        Args:
            data: Raw input data used to create the dataset instance.
            dataset_type: Type string used by DatasetFactory ('tabular', 'image', ...).
            model_name: Model name to instantiate via ModelFactory.
            workspace_path: Base folder where project folder will be created.
            target_variable, categorical_columns, ordinal_columns, model_params: optional config.
        """
        self.id = str(uuid.uuid4())
        self.created_at = datetime.now()
        self.dataset_type = dataset_type
        self.model_type = model_name
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
        self.blackbox = ModelFactory.create(model_name, model_params)

        # Discover compatible explainers for this dataset/model pair
        self.explainer_manager = ExplainerManager(dataset_type, model_name)
        # list_available_compatible_explainers returns classes
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

        Each pipeline step is a dict with:
            - "explainer": str  (explainer name, class __name__ is recommended)
            - "mode": "local" | "global"  (explicit; local requires instance_index)
            - "params": dict (optional additional parameters; for 'local' we expect "instance_index")

        Example:
            pipeline = [
                {"explainer": "ShapExplainerAdapter", "mode": "local", "params": {"instance_index": 12}},
                {"explainer": "GradCamAdapter", "mode": "global"}
            ]

        Returns:
            list of explanation records (and each is persisted to disk)
        """
        results: List[Dict[str, Any]] = []

        # normalize mapping of available explainers by lowercase name -> class
        explainer_map = {cls.__name__.lower(): cls for cls in self.explainers}

        for step in pipeline:
            # Basic validation / extraction
            explainer_name = step.get("explainer")
            if not explainer_name:
                raise ValueError("Pipeline step missing required field 'explainer'")

            mode = step.get("mode")  # must be 'local' or 'global'
            if mode is None:
                # infer default: if params has instance_index assume local, else global
                params = step.get("params", {})
                mode = "local" if "instance_index" in params else "global"
            mode = mode.lower()
            if mode not in ("local", "global"):
                raise ValueError(f"Invalid mode '{mode}' for explainer '{explainer_name}'. Use 'local' or 'global'.")

            params = step.get("params", {})

            # find explainer class (case-insensitive)
            explainer_key = explainer_name.lower()
            if explainer_key not in explainer_map:
                raise ValueError(
                    f"Explainer '{explainer_name}' is not compatible/available for dataset '{self.dataset_type}' "
                    f"and model '{self.model_type}'. Available: {list(explainer_map.keys())}"
                )
            explainer_cls = explainer_map[explainer_key]

            # instantiate the explainer directly (avoid name-case problems with manager)
            explainer: GenericExplainerAdapter = explainer_cls(model=self.blackbox, dataset=self.dataset_instance)

            # check explainer supports requested mode
            if mode == "local":
                if not hasattr(explainer, "explain_instance"):
                    raise ValueError(f"Explainer '{explainer_cls.__name__}' does not support local explanations.")
                # require instance_index parameter for local explanations
                if "instance_index" not in params:
                    raise ValueError("Local explanation requires 'instance_index' in params.")
                instance_index = params["instance_index"]
                try:
                    instance = self.dataset_instance[instance_index]
                except Exception as e:
                    raise ValueError(f"Invalid instance_index {instance_index}: {e}")
                # run local explanation
                explanation = explainer.explain_instance(instance)
                instance_index_for_record = instance_index

            else:  # mode == "global"
                if not hasattr(explainer, "explain_global"):
                    raise ValueError(f"Explainer '{explainer_cls.__name__}' does not support global explanations.")
                # run global explanation
                explanation = explainer.explain_global()
                instance = None
                instance_index_for_record = None

            # build record including mode and instance_index
            record = self._create_explanation_record(
                explainer_cls, mode, instance_index_for_record, instance, explanation
            )

            # append to in-memory history and persist to disk
            self.explanations.append(record)
            self._save_result(record)
            results.append(record)

        logger.info(f"Executed pipeline with {len(results)} explanations.")
        # update metadata to reflect new explanations count
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
            raise KeyError("YAML must contain top-level 'pipeline' key.")

        logger.info(f"Running pipeline loaded from YAML: {yaml_path}")
        # persist a copy of the YAML under project's pipelines folder for provenance
        basename = os.path.basename(yaml_path)
        saved_pipeline_path = os.path.join(self.workspace_path, "pipelines", basename)
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
        """
        Build a structured dictionary that captures:
          - timestamp
          - explainer class name
          - mode ('local'|'global')
          - instance_index (if local)
          - serialized instance (if local)
          - explanation payload (expected to have to_dict())
        """
        return {
            "timestamp": datetime.now().isoformat(),
            "explainer": explainer_cls.__name__,
            "mode": mode,
            "instance_index": instance_index,
            "instance": self._serialize_instance(instance),
            "result": getattr(explanation, "to_dict", lambda: explanation)(),
        }

    @classmethod
    def _serialize_instance(cls, instance) -> Any:
        """Serialize a dataset instance for recording; return None if instance is None."""
        if instance is None:
            return None
        if hasattr(instance, "to_dict"):
            return instance.to_dict()
        # Fallback: convert to string
        return str(instance)

    def _save_result(self, record: Dict[str, Any]) -> None:
        """
        Save a single explanation result as JSON into workspace/results.
        Filename encodes timestamp, explainer, and mode.
        """
        results_dir = os.path.join(self.workspace_path, "results")
        os.makedirs(results_dir, exist_ok=True)

        # sanitize timestamp for filename
        ts = record["timestamp"].replace(":", "_")
        expl = record["explainer"]
        mode = record["mode"]
        filename = f"{ts}_{expl}_{mode}.json"
        filepath = os.path.join(results_dir, filename)

        with open(filepath, "w") as f:
            json.dump(record, f, indent=2, default=str)

        logger.debug(f"Saved explanation result to {filepath}")

    def _save_metadata(self) -> None:
        """
        Persist project metadata (summary) to project.json in workspace root.
        """
        metadata = self.to_dict()
        path = os.path.join(self.workspace_path, "project.json")
        with open(path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

    # -----------------------------
    # Serialization / small utilities
    # -----------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Return a serializable summary of the project."""
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
        Restore a project from a metadata dict.

        Note: this reconstructs dataset/model via factories. Raw dataset contents
        are not automatically reloaded here (caller may re-inject data if needed).
        """
        # Recreate the project shell using __new__ then populate fields
        project = cls.__new__(cls)

        project.id = data["id"]
        project.created_at = datetime.fromisoformat(data["created_at"])
        project.dataset_type = data["dataset_type"]
        project.model_type = data["model_type"]
        project.workspace_path = data["workspace_path"]

        # Recreate minimal workspace structure in memory (do not overwrite disk)
        project.dataset_instance = DatasetFactory.create(data=None, dataset_type=project.dataset_type)
        project.blackbox = ModelFactory.create(project.model_type, model_params=None)
        project.explainer_manager = ExplainerManager(project.dataset_type, project.model_type)
        project.explainers = project.explainer_manager.list_available_compatible_explainers()
        project.explanations = []  # explanation files remain on disk; in-memory history starts empty

        return project

    def __repr__(self):
        return f"<Project id={self.id}, dataset={self.dataset_type}, model={self.model_type}>"
