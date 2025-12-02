import json
import os
from typing import Dict, Optional, List, Any

from fairxai.logger import logger
from fairxai.project.project import Project


class ProjectRegistry:
    """
    Registry that manages explainability projects inside a workspace.

    This registry does **not** eagerly load all projects into memory.
    Instead, during initialization it scans the workspace directory
    and registers all project folders that contain a valid ``project.json``.
    Each project is then loaded on-demand only when required.

    The registry maintains two types of information:

    * the *index* of existing project IDs on disk (always populated)
    * the *loaded Project objects* in memory (populated only after `load_project`)

    Parameters
    ----------
    workspace_path : str
        Base directory where all projects are stored.

    Notes
    -----
    This class ensures that each workspace path has exactly one registry
    instance (per-workspace singleton).
    """

    _instances: Dict[str, "ProjectRegistry"] = {}

    def __new__(cls, workspace_path: str) -> "ProjectRegistry":
        if workspace_path not in cls._instances:
            cls._instances[workspace_path] = super(ProjectRegistry, cls).__new__(cls)
        return cls._instances[workspace_path]

    def __init__(self, workspace_path: str) -> None:
        """
        Initialize the registry and build the *project index*.

        The index maps project IDs to either ``None`` (if not yet loaded)
        or a fully-loaded ``Project`` object.

        Existing projects are detected by locating folders that contain a
        ``project.json`` file.
        """
        self.workspace_base: str = workspace_path
        os.makedirs(self.workspace_base, exist_ok=True)

        # Maps project_id -> Project instance or None (if not loaded yet)
        self._projects: Dict[str, Optional[Project]] = {}

        # Scan the workspace and register all valid project directories
        for entry in os.listdir(self.workspace_base):
            project_dir = os.path.join(self.workspace_base, entry)
            metadata_path = os.path.join(project_dir, "project.json")

            # Only folders with project.json are considered valid projects
            if os.path.isdir(project_dir) and os.path.exists(metadata_path):
                self._projects[entry] = None
            else:
                logger.debug(f"Skipping non-project entry: {entry}")

    # ============================================================
    # Index and metadata inspection
    # ============================================================

    def list_all(self) -> List[Dict[str, Any]]:
        """
        Return metadata for all projects on disk.

        This method reads the ``project.json`` files directly without
        loading full ``Project`` objects into memory.

        Returns
        -------
        List[Dict[str, Any]]
            A list of dictionaries containing project metadata.
        """
        results: List[Dict[str, Any]] = []

        for project_id in self._projects.keys():
            metadata = self.get_metadata(project_id)
            if metadata is not None:
                results.append(metadata)

        return results

    def get_metadata(self, project_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve metadata for a specific project by its ID.

        Parameters
        ----------
        project_id : str
            The unique identifier of the project.

        Returns
        -------
        dict or None
            The parsed ``project.json`` content, or ``None`` if missing or invalid.
        """
        metadata_path = os.path.join(self.workspace_base, project_id, "project.json")

        if not os.path.exists(metadata_path):
            logger.warning(f"Metadata not found for project {project_id}")
            return None

        try:
            with open(metadata_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading metadata for {project_id}: {e}")
            return None

    def find_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve project metadata by matching the project name.

        Parameters
        ----------
        name : str
            The project name to search for.

        Returns
        -------
        dict or None
            Metadata of the first matching project, or None if no match is found.
        """
        for metadata in self.list_all():
            if metadata.get("project_name") == name:
                return metadata
        return None

    # ============================================================
    # Registry management
    # ============================================================

    def add(self, project: Project) -> None:
        """
        Add or update a loaded project in the registry.

        Parameters
        ----------
        project : Project
            The project instance to register.
        """
        # Add or override the project instance
        self._projects[project.id] = project
        logger.info(f"Project {project.id} added to registry.")

    def get(self, project_id: str) -> Optional[Project]:
        """
        Get project metadata (does not read from disk).

        Parameters
        ----------
        project_id : str
            ID of the project to retrieve.

        Returns
        -------
        Project or None
            The loaded project instance, or None if not loaded yet.
        """
        return self._projects.get(project_id)

    def remove(self, project_id: str) -> None:
        """
        Remove a project from the in-memory registry.

        This does not delete the project from disk.

        Parameters
        ----------
        project_id : str
            The project ID to remove.
        """
        if project_id in self._projects:
            del self._projects[project_id]
            logger.info(f"Project {project_id} removed from registry.")
        else:
            logger.warning(f"Tried to remove unknown project {project_id}.")

    def get_project_path(self, project_id: str) -> str:
        """
        Get the filesystem path of a project folder.

        Parameters
        ----------
        project_id : str

        Returns
        -------
        str
            Path to the project's directory.
        """
        return os.path.join(self.workspace_base, project_id)

    # ============================================================
    # Persistence and loading
    # ============================================================

    def load_project(self, project_id: str) -> Project:
        """
        Load a project from disk into memory.

        Parameters
        ----------
        project_id : str
            Unique project identifier.

        Returns
        -------
        Project
            The fully loaded project instance.

        Raises
        ------
        FileNotFoundError
            If ``project.json`` does not exist.
        """
        metadata = self.get_metadata(project_id)
        if metadata is None:
            raise FileNotFoundError(
                f"No valid project.json found for project {project_id}"
            )

        # Construct full project object using its loader
        project = Project.load_from_dict(metadata)

        # Store in registry
        self._projects[project.id] = project
        logger.info(f"Loaded project {project_id} from disk.")

        return project

    def clear(self) -> None:
        """
        Clear all loaded project instances from memory.

        This does not remove project folders from disk.
        """
        self._projects.clear()
        logger.info("Registry cleared.")