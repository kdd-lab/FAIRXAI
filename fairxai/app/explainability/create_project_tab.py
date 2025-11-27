import datetime
import json

import streamlit as st
from fairxai.project.project import Project
from fairxai.project.project_registry import ProjectRegistry
import pandas as pd
import os

def create_project_page():

    st.header("Crea un nuovo progetto")
    st.markdown("Compila i parametri per iniziare un nuovo progetto di spiegabilità.")
    workspace = os.path.abspath(os.path.normpath(os.path.join(os.getcwd(), "..","..","workspace")))

    project_name = st.text_area("Nome del progetto", value="Project-{0}".format(datetime.datetime.now().strftime('%Y%M%d-%H%M') ))
    dataset_type = st.selectbox("Tipo di dataset", ["tabular", "image", "text"])
    if dataset_type=="tabular":
        target_variable = st.text_input("Variabile target", "")
        categorical_columns = st.text_input("Variabili categoriche (separate da virgola)", "")
        ordinal_columns = st.text_input("Variabili ordinali (separate da virgola)", "")
    else:
        target_variable = None
        categorical_columns = None
        ordinal_columns = None

    framework = st.selectbox("Tipo di modello", ["sklearn", "torch"])
    if framework=="torch":
        device = st.selectbox("Devide", ["cpu", "gpu"])

    model_path = st.file_uploader("Carica modello")
    model_params = st.text_area("Parametri del modello (JSON)", value="{}")

    uploaded_file = st.file_uploader("Carica dataset", type=["csv"])
    creating = st.button("Crea progetto")

    if creating is True:
        try:
            model_params = json.loads(model_params)
            data = None

            if uploaded_file is not None:
                if dataset_type == "tabular":
                    data = pd.read_csv(uploaded_file)
                else:
                    data = uploaded_file.read()

            registry = ProjectRegistry(workspace)
            project = Project(
                name_project = project_name,
                data=data,
                dataset_type=dataset_type,
                framework=framework,
                model_path=model_path or None,
                model_params=model_params,
                workspace_path=workspace,
                target_variable=target_variable,
                categorical_columns=[c.strip() for c in categorical_columns.split(",") if categorical_columns!= None ],
                ordinal_columns=[c.strip() for c in ordinal_columns.split(",") if ordinal_columns!= None ],
                device=device,
            )
            registry.add(project)
            st.success(f"Progetto creato con successo! ID: {project.name_project}")
            st.json(project.to_dict())

        except Exception as e:
            st.error(f"Errore nella creazione del progetto: {e}")