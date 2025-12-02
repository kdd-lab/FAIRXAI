import json
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Type

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
    a model file path. The correct black-box wrapper is automatically instantiated
    and loaded.

    Attributes:
        id (str): UUID of the project.
        created_at (datetime): Timestamp of creation.
        dataset_instance (Dataset): The dataset instance (tabular, image, etc.).
        blackbox (Any): Wrapped ML model.
        explainers (List[Type[GenericExplainerAdapter]]): Compatible explainer classes.
        explanations (List[Dict[str, Any]]): Stored explanation records.
    """

    def __init__(
            self,
            project_name: str,
            data: Any = None,
            dataset_metadata: Optional[dict] = None,
            dataset_type: Optional[str] = None,
            framework: Optional[str] = None,
            model_path: Optional[str] = None,
            model_params: Optional[Dict[str, Any]] = None,
            workspace_path: str = "./workspace",
            target_variable: Optional[str] = None,
            categorical_columns: Optional[List[str]] = None,
            ordinal_columns: Optional[List[str]] = None,
            device: str = "cpu",
            is_reload: bool = False,
            id: str = None
    ):
        """
        Initialize a new project or reload an existing one.

        Either `data` (for new project) or `dataset_metadata` (for reload) must be provided.

        :param project_name: Name of the project
        :param data: Raw dataset to create the dataset instance (new project)
        :param dataset_metadata: Metadata to reload dataset (existing project)
        :param dataset_type: Dataset type ('tabular', 'image', 'text', 'timeseries')
        :param framework: ML framework ('sklearn', 'torch', etc.)
        :param model_path: Optional path to pre-trained model
        :param model_params: Optional parameters for model instantiation
        :param workspace_path: Base folder for project workspace
        :param target_variable: Name of target variable in dataset (optional)
        :param categorical_columns: Categorical columns (tabular only)
        :param ordinal_columns: Ordinal columns (tabular only)
        :param device: Device for Torch models ('cpu' or 'cuda')
        :param is_reload: If True, indicates that this project is being reloaded and folders should not be recreated
        """
        self.id: str = str(uuid.uuid4()) if is_reload is False else id
        self.created_at: datetime = datetime.now()
        self.project_name: str = project_name
        self.dataset_type: Optional[str] = dataset_type
        self.framework: Optional[str] = framework

        # Decide workspace path: reuse if reloading, else create new UUID folder
        if is_reload and dataset_metadata is not None:
            # Use workspace from metadata if reloading
            self.workspace_path: str = workspace_path
            os.makedirs(self.workspace_path, exist_ok=True)  # ensure it exists
        else:
            # New project: create unique workspace folder
            self.workspace_path: str = os.path.join(workspace_path, self.id)
            os.makedirs(self.workspace_path, exist_ok=True)
            os.makedirs(os.path.join(self.workspace_path, "results"), exist_ok=True)
            os.makedirs(os.path.join(self.workspace_path, "pipelines"), exist_ok=True)
            os.makedirs(os.path.join(self.workspace_path, "logs"), exist_ok=True)


        # Store model & dataset parameters
        self.model_path: Optional[str] = model_path
        self.model_params: Optional[Dict[str, Any]] = model_params
        self.device: str = device
        self.target_variable: Optional[str] = target_variable
        self.categorical_columns: Optional[List[str]] = categorical_columns
        self.ordinal_columns: Optional[List[str]] = ordinal_columns
        
        # -------------------------
        # Dataset initialization
        # -------------------------
        if data is not None:
            # New dataset instance from raw data
            self.dataset_instance = DatasetFactory.create(
                data,
                dataset_type,
                class_name=target_variable,
                categorical_columns=categorical_columns,
                ordinal_columns=ordinal_columns
            )
        elif dataset_metadata is not None:
            # Reload dataset from saved metadata using dataset-specific from_dict
            dataset_class = DatasetFactory.get_class(dataset_type)
            self.dataset_instance = dataset_class.from_dict(
                dataset_metadata,
                project_path=self.workspace_path
            )
        else:
            raise ValueError("Either `data` or `dataset_metadata` must be provided")

        # -------------------------
        # Model / blackbox wrapper
        # -------------------------
        self.blackbox = ModelFactory.create(
            framework=framework,
            model_path=model_path,
            model_params=model_params,
            device=device
        )
        self.model_type: str = type(self.blackbox.model).__name__

        # -------------------------
        # Explainer discovery
        # -------------------------
        self.explainer_manager = ExplainerManager(self.dataset_type, self.model_type)
        self.explainers: List[Type[GenericExplainerAdapter]] = (
            self.explainer_manager.list_available_compatible_explainers()
        )

        # -------------------------
        # Explanation storage
        # -------------------------
        self.explanations: List[Dict[str, Any]] = []

        logger.info(f"Project {self.id} initialized with dataset {self.dataset_type} and model {self.model_type}.")
        logger.info(f"Found {len(self.explainers)} compatible explainers.")

        # Persist metadata immediately
        self._save_metadata()

    # -------------------------------------------------------------------------
    # Pipeline execution
    # -------------------------------------------------------------------------
    def run_explanation_pipeline(self, pipeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Execute a user-defined explainability pipeline.

        Each step should include:
            - 'explainer': str (explainer class name)
            - 'mode': 'local' | 'global' (optional; inferred if missing)
            - 'params': dict (optional; 'local' requires 'instance_index')

        :param pipeline: List of pipeline steps
        :return: List of explanation records
        """
        results: List[Dict[str, Any]] = []
        explainer_map = {cls.explainer_name: cls for cls in self.explainers}

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

            if explainer_name not in explainer_map:
                raise ValueError(
                    f"Explainer '{explainer_name}' not compatible for dataset '{self.dataset_type}' "
                    f"and framework '{self.framework}'. Available: {list(explainer_map.keys())}"
                )
            explainer_cls = explainer_map[explainer_name]
            explainer: GenericExplainerAdapter = explainer_cls(model=self.blackbox, dataset=self.dataset_instance)

            if mode == "local":
                params = step.get("params", {})

                # -----------------------------
                # 1. Fetch instance correctly
                # -----------------------------
                instance = None

                if "instance_filename" in params:
                    key = params["instance_filename"]
                    if hasattr(self.dataset_instance, "get_instance"):
                        instance = self.dataset_instance.get_instance(key)
                        instance_index = None
                    else:
                        raise TypeError(
                            f"Dataset {type(self.dataset_instance).__name__} does not support filename-based access."
                        )

                elif "instance_index" in params:
                    instance_index = params["instance_index"]
                    if hasattr(self.dataset_instance, "get_instance"):
                        instance = self.dataset_instance.get_instance(instance_index )
                    else:
                        instance = self.dataset_instance[instance_index ]

                else:
                    raise ValueError(
                        "Local explanation requires either 'instance_index' or 'instance_filename' in params."
                    )

                # ---------------------------------------
                # 2. Pass params to the explainer
                #    (this includes hwc_permutation)
                # ---------------------------------------
                explanations_list = explainer.explain_instance(
                    instance,
                    params=params  # This forwards hwc_permutation, instance_filename and so on
                )

            else:
                instance_index = None
                instance = None
                explanations_list: list[GenericExplanation] = explainer.explain_global()

            # Build and save record
            record = self._create_explanation_record(explainer_cls, mode, instance_index, instance, explanations_list)
            self.explanations.append(record)
            self._save_result(record)
            results.append(record)

        # Update project metadata after pipeline execution
        self._save_metadata()
        return results

    # -------------------------------------------------------------------------
    # Pipeline from YAML
    # -------------------------------------------------------------------------
    def run_pipeline_from_yaml(self, yaml_path: str) -> List[Dict[str, Any]]:
        """
        Load pipeline definition from YAML file and execute it.

        :param yaml_path: Path to YAML file containing pipeline specification.
        :return: List of explanation records
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

    # -------------------------------------------------------------------------
    # Serialization / persistence
    # -------------------------------------------------------------------------
    def _create_explanation_record(
        self,
        explainer_cls: Type[GenericExplainerAdapter],
        mode: str,
        instance_index: Optional[int],
        instance,
        explanations_list: List[GenericExplanation],
    ) -> Dict[str, Any]:
        """Build a dictionary record of a single explanation for storage/logging."""
        return {
            "timestamp": datetime.now().isoformat(),
            "explainer": explainer_cls.__name__,
            "mode": mode,
            "instance_index": instance_index if instance_index is not None else -1,
            "instance": self._serialize_instance(instance),
            "result": [explanation.visualize() for explanation in explanations_list]
        }

    @staticmethod
    def _serialize_instance(instance) -> Any:
        """Serialize an instance for storage. Uses `to_dict()` if available."""
        if instance is None:
            return None
        if hasattr(instance, "to_dict"):
            return instance.to_dict()
        return str(instance)

    def _save_result(self, record: Dict[str, Any]) -> None:
        """Save an explanation record as JSON in the results folder."""
        results_dir = os.path.join(self.workspace_path, "results")
        os.makedirs(results_dir, exist_ok=True)
        ts = record["timestamp"].replace(":", "_")
        filename = f"{ts}_{record['explainer']}_{record['mode']}.json"
        path = os.path.join(results_dir, filename)
        with open(path, "w") as f:
            json.dump(record, f, indent=2, default=str)
        logger.debug(f"Saved explanation result to {path}")

    def _save_metadata(self) -> None:
        """Persist project metadata (including dataset) to JSON in the workspace."""
        metadata = self.to_dict()
        # Add dataset metadata
        metadata["dataset_metadata"] = self.dataset_instance.to_dict()
        path = os.path.join(self.workspace_path, "project.json")
        with open(path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

    def to_dict(self) -> Dict[str, Any]:
        """Return a serializable dictionary of project metadata."""
        return {
            "project_name": self.project_name,
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "dataset_type": self.dataset_type,
            "framework": self.framework,
            "model_type": self.model_type,
            "model_path": self.model_path,
            "model_params": self.model_params,
            "device": self.device,
            "target_variable": self.target_variable,
            "categorical_columns": self.categorical_columns,
            "ordinal_columns": self.ordinal_columns,
            "workspace_path": self.workspace_path,
            "num_explainers": len(self.explainers),
            "num_explanations": len(self.explanations),
        }

    @classmethod
    def load_from_dict(cls, project_metadata: Dict[str, Any]) -> "Project":
        """
        Restore a project from metadata dictionary.

        :param project_metadata: Project metadata dictionary.
        :return: Project instance with dataset and blackbox reconstructed.
        """
        project = cls(
            project_name=project_metadata["project_name"],
            data=None,
            dataset_metadata=project_metadata.get("dataset_metadata"),
            dataset_type=project_metadata["dataset_type"],
            framework=project_metadata["framework"],
            workspace_path=project_metadata["workspace_path"],
            model_path=project_metadata["model_path"],
            model_params=project_metadata["model_params"],
            target_variable=project_metadata["target_variable"],
            categorical_columns=project_metadata["categorical_columns"],
            ordinal_columns=project_metadata["ordinal_columns"],
            device=project_metadata["device"],
            is_reload=True,
            id=project_metadata["id"]
        )
        project.created_at = datetime.fromisoformat(project_metadata["created_at"])
        return project

    def __repr__(self):
        return (
            f"<Project id={self.id}, name={self.project_name}, "
            f"dataset={self.dataset_type}, framework={self.framework}, model={self.model_type}>"
        )
