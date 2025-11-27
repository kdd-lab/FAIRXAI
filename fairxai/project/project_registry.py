import json
import os
from typing import Dict, Optional, List

from fairxai.logger import logger
from fairxai.project.project import Project


class ProjectRegistry:
    """
        Registry to store and manage multiple explainability projects.

        This registry acts as a central catalog of active or stored projects,
        allowing them to be saved, reloaded, and managed programmatically.

        Each project is uniquely identified by its UUID.
        """
    _instances = {}

    def __new__(cls, workspace_path: str):
        if workspace_path not in cls._instances:
            cls._instances[workspace_path] = super(ProjectRegistry, cls).__new__(cls)
        return cls._instances[workspace_path]

    def __init__(self, workspace_base: str):
        """
        Initialize a registry bound to a workspace base directory.
        """
        self.workspace_base = workspace_base
        os.makedirs(self.workspace_base, exist_ok=True)
        self._projects: Dict[str, Project] = {}
        for project_dir in os.listdir(workspace_base):
            self.load_project(project_dir)

    # ============================================================
    # Basic management
    # ============================================================
    def add(self, project: Project) -> None:
        """Registers a project in the registry."""
        if project.id in self._projects:
            logger.warning(f"Project {project.id} already registered, overwriting.")
        self._projects[project.id] = project
        logger.info(f"Project {project.id} added to registry.")

    def get(self, project_id: str) -> Optional[Project]:
        """Retrieves a project by its ID."""
        return self._projects.get(project_id)

    def remove(self, project_id: str) -> None:
        """Removes a project from the registry."""
        if project_id in self._projects:
            del self._projects[project_id]
            logger.info(f"Project {project_id} removed from registry.")
        else:
            logger.warning(f"Tried to remove unknown project {project_id}.")

    def list_all(self) -> List[str]:
        """Returns a list of all project IDs currently in the registry."""
        return list(self._projects.keys())

    def get_project_path(self, project_id: str) -> str:
        """Returns the workspace path of a given project."""
        return os.path.join(self.workspace_base, project_id)

    # ============================================================
    # Persistence
    # ============================================================

    def load_project(self, project_id: str) -> Optional[Project]:
        """Loads an existing project by ID from disk."""
        project_dir = os.path.join(self.workspace_base, project_id)
        metadata_path = os.path.join(project_dir, "project.json")

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"No project found at {metadata_path}")

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        project = Project.load_from_dict(metadata)
        self._projects[project.id] = project
        logger.info(f"Loaded project {project_id} from disk.")
        return project

    def clear(self) -> None:
        """Removes all registered projects."""
        self._projects.clear()
        logger.info("Registry cleared.")
