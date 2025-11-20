import json
import os

import pandas as pd
import streamlit as st
from fairxai.project.project_registry import ProjectRegistry

def results_page():
    st.subheader("Visualizza risultati delle spiegazioni")

    workspace_path = os.path.abspath(os.path.normpath(os.path.join(os.getcwd(), "..", "..", "workspace")))

    registry = ProjectRegistry(workspace_path)
    project_ids = registry.list_all()

    if not project_ids:
        st.info("Nessun progetto disponibile.")
        return

    selected_id = st.selectbox("Seleziona un progetto:", project_ids)
    project = registry.load_project(selected_id)

    results_dir = os.path.join(project.workspace_path, "results")
    result_files = [f for f in os.listdir(results_dir) if f.endswith(".json")]

    if not result_files:
        st.warning("Nessuna spiegazione trovata.")
        return

    selected_result = st.selectbox("Scegli una spiegazione:", result_files)
    result_path = os.path.join(results_dir, selected_result)

    with open(result_path, "r") as f:
        record = json.load(f)

    st.write(f"**Explainer:** {record['explainer']} | **Modalità:** {record['mode']}")
    st.json(record["result"])

    if "feature_importance" in record["result"]:
        fi = record["result"]["feature_importance"]
        df = pd.DataFrame(fi.items(), columns=["feature", "importance"])
        st.bar_chart(df.set_index("feature"))
