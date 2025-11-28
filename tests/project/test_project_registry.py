import json
import os
from pathlib import Path

import pytest

from fairxai.project.project import Project
from fairxai.project.project_registry import ProjectRegistry


def create_fake_project(
    base: Path,
    project_id: str,
    name: str = "test_project",
    dataset_path: str = "/data/test.csv",
    model_path: str = "/models/model.pkl",
):
    """
    Utility function to create a fake project folder with project.json.
    """
    project_dir = base / project_id
    project_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "id": project_id,
        "name": name,
        "dataset_path": dataset_path,
        "model_path": model_path,
    }

    with open(project_dir / "project.json", "w") as f:
        json.dump(metadata, f)

    return metadata


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """
    Provides a temporary workspace directory for tests.
    """
    return tmp_path / "workspace"


def test_registry_initialization_with_existing_projects(workspace: Path):
    """
    Registry should detect existing project folders on disk during __init__,
    but should not load Project objects yet (value = None).
    """
    workspace.mkdir(parents=True, exist_ok=True)
    create_fake_project(workspace, "proj1")
    create_fake_project(workspace, "proj2")

    registry = ProjectRegistry(str(workspace))

    assert set(registry._projects.keys()) == {"proj1", "proj2"}
    assert registry.get("proj1") is None
    assert registry.get("proj2") is None


def test_list_all_metadata(workspace: Path):
    """
    list_all() should return metadata dictionaries that match project.json content.
    """
    workspace.mkdir(parents=True, exist_ok=True)
    md1 = create_fake_project(workspace, "pA", name="Alpha")
    md2 = create_fake_project(workspace, "pB", name="Beta")

    registry = ProjectRegistry(str(workspace))
    results = registry.list_all()

    assert len(results) == 2
    assert md1 in results
    assert md2 in results


def test_get_metadata_for_specific_project(workspace: Path):
    """
    get_metadata(project_id) must return the correct metadata dictionary.
    """
    workspace.mkdir(parents=True, exist_ok=True)
    md = create_fake_project(workspace, "p123", name="MyProj")

    registry = ProjectRegistry(str(workspace))
    result = registry.get_metadata("p123")

    assert result == md
    assert result["name"] == "MyProj"


def test_find_by_name(workspace: Path):
    """
    find_by_name must return the metadata for the matching project name.
    """
    workspace.mkdir(parents=True, exist_ok=True)
    create_fake_project(workspace, "p1", name="ToFind")
    create_fake_project(workspace, "p2", name="Another")

    registry = ProjectRegistry(str(workspace))
    result = registry.find_by_name("ToFind")

    assert result is not None
    assert result["id"] == "p1"


def test_load_project_returns_project_instance(monkeypatch, workspace: Path):
    """
    load_project must return a Project object created via Project.load_from_dict,
    and registry._projects should store it.
    """

    # Prepare fake project on disk
    workspace.mkdir(parents=True, exist_ok=True)
    md = create_fake_project(workspace, "px", name="LoadMe")

    # Monkeypatch Project.load_from_dict to avoid requiring full Project implementation
    class DummyProject:
        def __init__(self, **kwargs):
            self.id = kwargs["id"]
            self.name = kwargs["name"]

    def fake_loader(d):
        return DummyProject(**d)

    monkeypatch.setattr(Project, "load_from_dict", staticmethod(fake_loader))

    registry = ProjectRegistry(str(workspace))
    proj = registry.load_project("px")

    assert isinstance(proj, DummyProject)
    assert proj.id == "px"
    # Check registry stored the loaded project
    assert registry.get("px") is proj


def test_add_and_remove_project(workspace: Path):
    """
    add() should store a project instance in memory,
    remove() should delete it from registry._projects.
    """
    workspace.mkdir(parents=True, exist_ok=True)

    registry = ProjectRegistry(str(workspace))

    # Prepare fake project object
    class P:
        id = "abc"

    p = P()
    registry.add(p)

    assert registry.get("abc") is p

    registry.remove("abc")
    assert registry.get("abc") is None


def test_clear_registry(workspace: Path):
    """
    clear() removes all loaded project references from memory.
    """
    workspace.mkdir(parents=True, exist_ok=True)
    create_fake_project(workspace, "A")

    registry = ProjectRegistry(str(workspace))
    assert "A" in registry._projects

    registry.clear()
    assert registry._projects == {}
