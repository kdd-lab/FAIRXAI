# ----------------------------------------------------
# Automatically determine project root and workspace
# ----------------------------------------------------
# directory where THIS script lives
import os
from pprint import pprint

from fairxai.project.project_registry import ProjectRegistry

_CURRENT = os.path.abspath(os.path.dirname(__file__))

# parent directory = project root
PROJECT_ROOT = os.path.dirname(_CURRENT)

# workspace inside project root
WORKSPACE_BASE = os.path.join(PROJECT_ROOT, "workspace")
os.makedirs(WORKSPACE_BASE, exist_ok=True)

print(f"Project root: {PROJECT_ROOT}")
print(f"Workspace base: {WORKSPACE_BASE}")

DATA = os.path.join(WORKSPACE_BASE, "train")
DATASET_TYPE = "image"  # "tabular" | "image" | "text"
MODEL_PARAMS = None
MODEL_PATH = os.path.join(WORKSPACE_BASE, "pytorch_model_full.pt")
TARGET_VARIABLE = None
PIPELINE_YAML_LOCAL = os.path.join(WORKSPACE_BASE, "sias_example_pipeline.yaml")

# ----------------------------------------------------
# Create registry (singleton per workspace path)
# ----------------------------------------------------
registry = ProjectRegistry(WORKSPACE_BASE)

# ----------------------------------------------------
# Create a new project
# ----------------------------------------------------
# print("Creating Project inside workspace/...")
#
# project = Project(
#     project_name="SIAS with PyTorch",
#     data=DATA,
#     dataset_type=DATASET_TYPE,
#     framework="torch",
#     workspace_path=WORKSPACE_BASE,
#     target_variable=TARGET_VARIABLE,
#     model_params=MODEL_PARAMS,
#     model_path=MODEL_PATH)
#
# registry.add(project)
#
# print("Project created:")
# print(" - ID:", project.id)
# print(" - Workspace:", project.workspace_path)

existing_projects = registry.list_all()
pprint(existing_projects)

project = registry.load_project("6982bdd0-b220-4504-92b1-5ff7e820229a")

# ----------------------------------------------------
# Run from YAML if available
# ----------------------------------------------------
if PIPELINE_YAML_LOCAL and os.path.exists(PIPELINE_YAML_LOCAL):
    print(f"Running pipeline from YAML: {PIPELINE_YAML_LOCAL}")
    results = project.run_pipeline_from_yaml(PIPELINE_YAML_LOCAL)
else:
    print("No YAML found")
    results = None

# ----------------------------------------------------
# Show results
# ----------------------------------------------------
print("\nPipeline results:")
pprint(results)

print("\nSaved result files:")
for fname in os.listdir(os.path.join(project.workspace_path, "results")):
    print(" -", fname)

print("\nProject metadata:")
with open(os.path.join(project.workspace_path, "project.json"), "r") as f:
    print(f.read())

print("\nDone.")
