"""
run_project_pipeline.py

End-to-end runner that:
- Automatically creates workspace under the project root folder.
- Creates a Project.
- Registers it inside ProjectRegistry.
- Runs YAML or programmatic pipeline.
"""

import os
from pprint import pprint

import pandas as pd

from fairxai.project.project import Project
from fairxai.project.project_registry import ProjectRegistry

# ----------------------------------------------------
# Automatically determine project root and workspace
# ----------------------------------------------------
# directory where THIS script lives
_CURRENT = os.path.abspath(os.path.dirname(__file__))

# parent directory = project root
PROJECT_ROOT = os.path.dirname(_CURRENT)

# workspace inside project root
WORKSPACE_BASE = os.path.join(PROJECT_ROOT, "workspace")
os.makedirs(WORKSPACE_BASE, exist_ok=True)

print(f"Project root: {PROJECT_ROOT}")
print(f"Workspace base: {WORKSPACE_BASE}")

DATA = os.path.join(WORKSPACE_BASE, "usecase_scam_class_train_clean.csv")
DATASET_TYPE = "tabular"  # "tabular" | "image" | "text"
MODEL_PARAMS = None
MODEL_PATH = os.path.join(WORKSPACE_BASE, "usecase_scam_model_class.pkl")
TARGET_VARIABLE = "target"
PIPELINE_YAML_LOCAL = os.path.join(WORKSPACE_BASE, "scam_example_pipeline.yaml")
# target_name, type, sample_name, group
CATEGORICAL_COLUMNS = ["target_name", "type", "sample_name", "group"]
# group_id,sample_id, target_id,type_id,timestamp
ORDINAL_COLUMNS = ["group_id", "sample_id", "target_id", "type_id", "timestamp"]

df_train = pd.read_csv(DATA)
numeric_cols = [c for c in df_train.columns if c.replace('.', '', 1).isdigit()]
clean_train_df = df_train[numeric_cols + [TARGET_VARIABLE]]

# ----------------------------------------------------
# Create registry (singleton per workspace path)
# ----------------------------------------------------
registry = ProjectRegistry(WORKSPACE_BASE)

# ----------------------------------------------------
# Create a new project
# ----------------------------------------------------
print("Creating Project inside workspace/...")

project = Project(
    project_name="SCAM with LORE",
    data=DATA,
    dataset_type=DATASET_TYPE,
    framework="sklearn",
    workspace_path=WORKSPACE_BASE,
    target_variable=TARGET_VARIABLE,
    model_params=MODEL_PARAMS,
    model_path=MODEL_PATH,
    categorical_columns=CATEGORICAL_COLUMNS,
    ordinal_columns=ORDINAL_COLUMNS)

registry.add(project)

print("Project created:")
print(" - ID:", project.id)
print(" - Workspace:", project.workspace_path)

# ----------------------------------------------------
# Run from YAML if available
# ----------------------------------------------------
if PIPELINE_YAML_LOCAL and os.path.exists(PIPELINE_YAML_LOCAL):
    print(f"Running pipeline from YAML: {PIPELINE_YAML_LOCAL}")
    results = project.run_pipeline_from_yaml(PIPELINE_YAML_LOCAL)
else:
    print("No YAML found -> Running example programmatic pipeline")
    pipeline = [
        {
            "explainer": "LoreExplainerAdapter",
            "mode": "local",
            "params": {"instance_index": 0, "strategy": "genetic", "num_samples": 1500}
        },
        {
            "explainer": "LoreExplainerAdapter",
            "mode": "global",
            "params": {"strategy": "genetic", "n_samples": 5, "num_samples": 1000}
        }
    ]
    results = project.run_explanation_pipeline(pipeline)

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
