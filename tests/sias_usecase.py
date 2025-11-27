# ----------------------------------------------------
# Automatically determine project root and workspace
# ----------------------------------------------------
# directory where THIS script lives
import os

from fairxai.project.project import Project
from fairxai.project.project_registry import ProjectRegistry

_CURRENT = os.path.abspath(os.path.dirname(__file__))

# parent directory = project root
PROJECT_ROOT = os.path.dirname(_CURRENT)

# workspace inside project root
WORKSPACE_BASE = os.path.join(PROJECT_ROOT, "workspace")
os.makedirs(WORKSPACE_BASE, exist_ok=True)

print(f"Project root: {PROJECT_ROOT}")
print(f"Workspace base: {WORKSPACE_BASE}")

DATA = os.path.join(WORKSPACE_BASE, "usecase_scam_class_train_clean.csv")
DATASET_TYPE = "image"                   # "tabular" | "image" | "text"
MODEL_PARAMS = None
MODEL_PATH = os.path.join(WORKSPACE_BASE, "pytorch_model.pth")
TARGET_VARIABLE = None
PIPELINE_YAML_LOCAL = os.path.join(WORKSPACE_BASE, "sias_example_pipeline.yaml")

# ----------------------------------------------------
# Create registry (singleton per workspace path)
# ----------------------------------------------------
registry = ProjectRegistry(WORKSPACE_BASE)

# ----------------------------------------------------
# Create a new project
# ----------------------------------------------------
print("Creating Project inside workspace/...")

project = Project(
    project_name="SIAS with PyTorch",
    data=[],
    dataset_type=DATASET_TYPE,
    framework="torch",
    workspace_path=WORKSPACE_BASE,
    target_variable=TARGET_VARIABLE,
    model_params=MODEL_PARAMS,
    model_path=MODEL_PATH)

registry.add(project)

print("Project created:")
print(" - ID:", project.id)
print(" - Workspace:", project.workspace_path)