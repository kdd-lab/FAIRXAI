import tempfile

import streamlit as st
from fairxai.project.project import Project
from fairxai.project.project_registry import ProjectRegistry
import pandas as pd
import os
def create_project_page():

    st.header("Crea un nuovo progetto")

    st.markdown("Compila i parametri per iniziare un nuovo progetto di spiegabilità.")

    workspace_path = os.path.abspath(os.path.normpath(os.path.join(os.getcwd(), "..","..","workspace")))

    dataset_type = st.selectbox("Tipo di dataset", ["tabular", "image"])
    model_name = st.text_input("Nome del modello", "dummy_model")
    target = st.text_input("Variabile target (solo per tabular)", "")

    uploaded_file = st.file_uploader("Carica dataset", type=["csv"])
    params_str = st.text_area("Parametri del modello (JSON)", value="{}")

    if st.button("Crea progetto"):

        if uploaded_file is None:
            st.error("Devi caricare un file CSV.")
            return

        df = pd.read_csv(uploaded_file)

        temp_path = tempfile.mktemp(suffix=".csv")
        df.to_csv(temp_path, index=False)

        registry = ProjectRegistry(workspace_path)
        project = Project(
            data=temp_path,
            dataset_type=dataset_type,
            model_name=model_name,
            workspace_path=workspace_path,
            target_variable=target
        )

        registry.add(project)

        st.success(f"Progetto creato con ID: {project.id}")
        st.json(project.to_dict())