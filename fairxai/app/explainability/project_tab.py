import os
from pathlib import Path
import streamlit as st

from fairxai.project.project_registry import ProjectRegistry

def projects_page():

    st.header("Elenco progetti")
    workspace_path = os.path.abspath(os.path.normpath(os.path.join(os.getcwd(), "..","..","workspace")))

    registry = ProjectRegistry(workspace_path)
    project_ids = registry.list_all()

    if not project_ids:
        st.info("Nessun progetto trovato nel workspace.")
        return

    selected_id = st.selectbox("Seleziona un progetto:", project_ids)
    load = st.button("Carica progetto")
    if load:
        try:
            project = registry.load_project(selected_id)
            st.success(f"Progetto {project.id} caricato.")
            st.json(project.to_dict())
        except Exception as e:
            st.error(f"Errore durante il caricamento: {e}")