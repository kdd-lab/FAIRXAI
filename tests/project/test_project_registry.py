import os
import json
import pytest
from unittest.mock import MagicMock, patch
from fairxai.project.project import Project
from fairxai.project.project_registry import ProjectRegistry


@pytest.fixture
def temp_workspace(tmp_path):
    """Create a temporary workspace directory."""
    return tmp_path / "workspace"


@pytest.fixture
def mock_project():
    """Create a mock Project instance with a fake ID."""
    project = MagicMock(spec=Project)
    project.id = "test_project"
    return project


# ============================================================
# GIVEN–WHEN–THEN TESTS
# ============================================================

def test_given_new_workspace_when_create_registry_then_instance_is_created(temp_workspace):
    """GIVEN an empty workspace
       WHEN a ProjectRegistry is created
       THEN it should create the directory and an instance of ProjectRegistry."""
    registry = ProjectRegistry(str(temp_workspace))

    assert isinstance(registry, ProjectRegistry)
    assert os.path.exists(registry.workspace_base)


def test_given_same_workspace_when_create_multiple_registries_then_same_instance(temp_workspace):
    """GIVEN a workspace path
       WHEN two registries are created with the same path
       THEN they must refer to the same singleton instance."""
    reg1 = ProjectRegistry(str(temp_workspace))
    reg2 = ProjectRegistry(str(temp_workspace))

    assert reg1 is reg2


def test_given_project_when_add_then_project_is_registered(temp_workspace, mock_project):
    """GIVEN a registry and a mock project
       WHEN the project is added
       THEN it must appear in the registry."""
    registry = ProjectRegistry(str(temp_workspace))
    registry.add(mock_project)

    assert registry.get(mock_project.id) == mock_project
    assert mock_project.id in registry.list_all()


def test_given_existing_project_when_remove_then_project_is_deleted(temp_workspace, mock_project):
    """GIVEN a registry with a project
       WHEN the project is removed
       THEN it must no longer be in the registry."""
    registry = ProjectRegistry(str(temp_workspace))
    registry.add(mock_project)

    registry.remove(mock_project.id)

    assert registry.get(mock_project.id) is None
    assert mock_project.id not in registry.list_all()


def test_given_nonexistent_project_when_remove_then_warning_is_logged(temp_workspace):
    """GIVEN an empty registry
       WHEN trying to remove a non-existent project
       THEN a warning should be logged."""
    registry = ProjectRegistry(str(temp_workspace))

    with patch("fairxai.project.project_registry.logger.warning") as mock_warn:
        registry.remove("nonexistent_id")

    mock_warn.assert_called_once()


def test_given_projects_when_clear_then_registry_is_empty(temp_workspace, mock_project):
    """GIVEN a registry with one or more projects
       WHEN clear() is called
       THEN all projects should be removed."""
    registry = ProjectRegistry(str(temp_workspace))
    registry.add(mock_project)

    registry.clear()

    assert registry.list_all() == []


def test_given_project_id_when_get_project_path_then_correct_path_is_returned(temp_workspace):
    """GIVEN a registry
       WHEN get_project_path() is called with an ID
       THEN it should return the correct workspace path."""
    registry = ProjectRegistry(str(temp_workspace))
    project_id = "abc123"

    path = registry.get_project_path(project_id)

    assert path == os.path.join(str(temp_workspace), project_id)


def test_given_project_json_exists_when_load_project_then_project_is_loaded(temp_workspace):
    """GIVEN a project directory with project.json on disk
       WHEN load_project() is called
       THEN it should load and register the project."""
    project_id = "proj1"
    project_dir = temp_workspace / project_id
    project_dir.mkdir(parents=True)
    project_data = {"id": project_id, "name": "Test Project"}

    with open(project_dir / "project.json", "w") as f:
        json.dump(project_data, f)

    registry = ProjectRegistry(str(temp_workspace))

    with patch.object(Project, "load_from_dict", return_value=MagicMock(id=project_id)) as mock_load:
        project = registry.load_project(project_id)

    mock_load.assert_called_once_with(project_data)
    assert project.id == project_id
    assert registry.get(project_id) == project


def test_given_missing_project_file_when_load_project_then_raises_file_not_found(temp_workspace):
    """GIVEN a workspace with no project.json
       WHEN load_project() is called
       THEN it should raise FileNotFoundError."""
    registry = ProjectRegistry(str(temp_workspace))
    project_id = "missing_project"

    with pytest.raises(FileNotFoundError):
        registry.load_project(project_id)
