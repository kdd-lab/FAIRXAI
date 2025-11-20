import os
import json
import yaml
import pytest
from unittest.mock import MagicMock, patch
from fairxai.project.project import Project


@pytest.fixture
def temp_workspace(tmp_path):
    """Create a temporary workspace directory."""
    return tmp_path / "workspace"


@pytest.fixture
def mock_factories():
    """Patch DatasetFactory, ModelFactory, and ExplainerManager."""
    with patch("fairxai.project.project.DatasetFactory.create") as mock_dataset_factory, \
         patch("fairxai.project.project.ModelFactory.create") as mock_model_factory, \
         patch("fairxai.project.project.ExplainerManager") as mock_manager_class:

        dataset_instance = MagicMock()
        model_instance = MagicMock()

        mock_dataset_factory.return_value = dataset_instance
        mock_model_factory.return_value = model_instance

        explainer_cls = MagicMock()
        explainer_cls.__name__ = "MockExplainer"
        explainer_instance = MagicMock()
        explainer_instance.explain_global.return_value = {"type": "global"}
        explainer_instance.explain_instance.return_value = {"type": "local"}
        explainer_cls.return_value = explainer_instance

        mock_manager = MagicMock()
        mock_manager.list_available_compatible_explainers.return_value = [explainer_cls]
        mock_manager_class.return_value = mock_manager

        yield {
            "dataset_factory": mock_dataset_factory,
            "model_factory": mock_model_factory,
            "manager_class": mock_manager_class,
            "explainer_cls": explainer_cls,
            "explainer_instance": explainer_instance,
        }


# ============================================================
# GIVEN–WHEN–THEN TESTS
# ============================================================

def test_given_valid_inputs_when_project_initialized_then_workspace_and_metadata_created(temp_workspace, mock_factories):
    """GIVEN valid dataset and model parameters
       WHEN a Project is instantiated
       THEN it should create its workspace structure and save metadata."""
    project = Project(
        data="dummy_data",
        dataset_type="tabular",
        model_name="mock_model",
        workspace_path=str(temp_workspace),
    )

    # THEN all workspace subfolders must exist
    for subdir in ["results", "pipelines", "logs"]:
        assert os.path.exists(os.path.join(project.workspace_path, subdir))

    # AND a metadata file must exist
    metadata_path = os.path.join(project.workspace_path, "project.json")
    assert os.path.exists(metadata_path)

    # AND the metadata must contain correct keys
    with open(metadata_path) as f:
        meta = json.load(f)
    assert meta["dataset_type"] == "tabular"
    assert "id" in meta


def test_given_valid_pipeline_when_run_explanation_pipeline_then_results_are_saved(temp_workspace, mock_factories):
    """GIVEN a valid project and compatible explainer
       WHEN run_explanation_pipeline() is executed
       THEN it should produce explanation records and save JSON results."""
    project = Project(
        data="dummy",
        dataset_type="tabular",
        model_name="mock",
        workspace_path=str(temp_workspace),
    )

    pipeline = [{"explainer": "MockExplainer", "mode": "global"}]

    results = project.run_explanation_pipeline(pipeline)

    # THEN one explanation record should be generated
    assert len(results) == 1
    record = results[0]
    assert record["explainer"] == "MockExplainer"
    assert record["mode"] == "global"

    # AND results should be saved in the 'results' folder
    results_files = list((temp_workspace / project.id / "results").glob("*.json"))
    assert len(results_files) == 1


def test_given_local_mode_when_instance_index_missing_then_raises_value_error(temp_workspace, mock_factories):
    """GIVEN a pipeline step in local mode without instance_index
       WHEN run_explanation_pipeline() is called
       THEN it should raise ValueError."""
    project = Project(
        data="dummy",
        dataset_type="tabular",
        model_name="mock",
        workspace_path=str(temp_workspace),
    )

    pipeline = [{"explainer": "MockExplainer", "mode": "local", "params": {}}]

    with pytest.raises(ValueError, match="Local explanation requires 'instance_index'"):
        project.run_explanation_pipeline(pipeline)


def test_given_yaml_file_when_run_pipeline_from_yaml_then_pipeline_is_executed(temp_workspace, mock_factories):
    """GIVEN a valid YAML pipeline file
       WHEN run_pipeline_from_yaml() is called
       THEN it should load the file, execute the pipeline, and save a JSON copy."""
    project = Project(
        data="dummy",
        dataset_type="tabular",
        model_name="mock",
        workspace_path=str(temp_workspace),
    )

    yaml_path = temp_workspace / "pipeline.yaml"
    yaml_content = {
        "pipeline": [
            {"explainer": "MockExplainer", "mode": "global"}
        ]
    }
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f)

    results = project.run_pipeline_from_yaml(str(yaml_path))

    # THEN results must contain one record
    assert len(results) == 1
    # AND the saved JSON pipeline copy must exist
    saved_copy = temp_workspace / project.id / "pipelines" / yaml_path.name
    assert saved_copy.exists()


def test_given_yaml_missing_pipeline_key_when_run_pipeline_from_yaml_then_raises_key_error(temp_workspace, mock_factories):
    """GIVEN a YAML file without 'pipeline' key
       WHEN run_pipeline_from_yaml() is called
       THEN it should raise KeyError."""
    project = Project(
        data="dummy",
        dataset_type="tabular",
        model_name="mock",
        workspace_path=str(temp_workspace),
    )

    yaml_path = temp_workspace / "bad.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump({"wrong": []}, f)

    with pytest.raises(KeyError):
        project.run_pipeline_from_yaml(str(yaml_path))


def test_given_missing_yaml_file_when_run_pipeline_from_yaml_then_raises_file_not_found(temp_workspace, mock_factories):
    """GIVEN a non-existing YAML file
       WHEN run_pipeline_from_yaml() is called
       THEN it should raise FileNotFoundError."""
    project = Project(
        data="dummy",
        dataset_type="tabular",
        model_name="mock",
        workspace_path=str(temp_workspace),
    )

    missing_path = temp_workspace / "not_found.yaml"
    with pytest.raises(FileNotFoundError):
        project.run_pipeline_from_yaml(str(missing_path))


def test_given_project_dict_when_load_from_dict_then_project_is_restored(mock_factories):
    """GIVEN a metadata dictionary
       WHEN load_from_dict() is called
       THEN it should return a valid Project instance with correct attributes."""
    data = {
        "id": "123",
        "created_at": "2023-01-01T00:00:00",
        "dataset_type": "tabular",
        "model_type": "mock_model",
        "workspace_path": "/tmp/workspace",
    }

    project = Project.load_from_dict(data)

    assert project.id == "123"
    assert project.dataset_type == "tabular"
    assert project.model_type == "mock_model"
    assert isinstance(project.explainers, list)
