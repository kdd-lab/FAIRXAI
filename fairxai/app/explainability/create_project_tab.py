import tempfile

import streamlit as st
from fairxai.project.project import Project
from fairxai.project.project_registry import ProjectRegistry
import pandas as pd
import os
def create_project_page():

    st.header("Crea un nuovo progetto")

    st.markdown("Compila i parametri per iniziare un nuovo progetto di spiegabilità.")

    workspace = st.text_input("Workspace directory",value=os.path.expanduser("~/fairxai_workspace"))

    uploaded_file = st.file_uploader("Carica dataset (CSV)", type=["csv"])
    dataset_type = st.selectbox("Tipo di dataset", ["tabular","image"])
    model_name = st.selectbox("Modello", ["xgboost", "random_forest", "logreg"])
    target = st.text_input("Target variable (opzionale)")

    if st.button("Crea progetto"):

        if uploaded_file is None:
            st.error("Devi caricare un file CSV.")
            return

        df = pd.read_csv(uploaded_file)

        # salva temporaneamente il dataset per passarne il path al Project
        temp_path = tempfile.mktemp(suffix=".csv")
        df.to_csv(temp_path, index=False)

        registry = ProjectRegistry(workspace)
        project = Project(
            data=temp_path,
            dataset_type=dataset_type,
            model_name=model_name,
            workspace_path=workspace,
            target_variable=target
        )

        registry.add(project)

        st.success(f"Progetto creato con ID: {project.id}")
        st.json(project.to_dict())