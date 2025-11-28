import json
import os
from typing import Dict, Optional, List

from fairxai.logger import logger
from fairxai.project.project import Project


class ProjectRegistry:
    """
    Registry to store and manage multiple explainability projects.

    Acts as a central catalog of active or stored projects, allowing them
    to be saved, reloaded, and managed programmatically.

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

        Only loads projects that have a valid `project.json` file.
        """
        self.workspace_base = workspace_base
        os.makedirs(self.workspace_base, exist_ok=True)
        self._projects: Dict[str, Project] = {}

        #FIXME: qui non devo caricare tutti i progetti, ma quando aggiungo elimino devo sempre aggiornare self._projects.
        # list_all in qualche modo dovrebbe restituirmi un dizionario con i metadati principali, nome progetto, uiid, path dataset e modello e cose del genere per ogni progetto
        # get dovrebbe restituirmi il dizionario di un progetto specifico
        # prevedere il retrieve anche filtrando per nome
        # load project dovrebbe caricare il progetto da disco e restituire un oggetto Project

        # entries = os.listdir(workspace_base)
        # if not entries:
        #     return  # Workspace empty, nothing to load
        #
        # for project_dir in entries:
        #     project_path = os.path.join(workspace_base, project_dir)
        #     metadata_path = os.path.join(project_path, "project.json")
        #     if os.path.isdir(project_path) and os.path.exists(metadata_path):
        #         try:
        #             self.load_project(project_dir)
        #         except Exception as e:
        #             logger.warning(f"Skipping project '{project_dir}': {e}")
        #     else:
        #         logger.debug(f"Skipping non-project entry: {project_dir}")

    # ============================================================
    # Basic management
    # ============================================================
    def add(self, project: Project) -> None:
        """Register a project in the registry."""
        if project.id in self._projects:
            logger.warning(f"Project {project.id} already registered, overwriting.")
        self._projects[project.id] = project
        logger.info(f"Project {project.id} added to registry.")

    def get(self, project_id: str) -> Optional[Project]:
        """Retrieve a project by its ID."""
        return self._projects.get(project_id)

    def remove(self, project_id: str) -> None:
        """Remove a project from the registry."""
        if project_id in self._projects:
            del self._projects[project_id]
            logger.info(f"Project {project_id} removed from registry.")
        else:
            logger.warning(f"Tried to remove unknown project {project_id}.")

    def list_all(self) -> List[str]:
        """Return a list of all project IDs currently in the registry."""
        return list(self._projects.keys())

    def get_project_path(self, project_id: str) -> str:
        """Return the workspace path of a given project."""
        return os.path.join(self.workspace_base, project_id)

    # ============================================================
    # Persistence
    # ============================================================
    def load_project(self, project_id: str) -> Project:
        """
        Load an existing project by ID from disk.

        Raises:
            FileNotFoundError: if the project.json file does not exist
        """
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
        """Remove all registered projects."""
        self._projects.clear()
        logger.info("Registry cleared.")
