import datetime
import json
from pathlib import Path

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
        device = st.selectbox("Device", ["cpu", "gpu"])
    else: device = "cpu"

    # ==============================
    # FILE BROWSER PER IL MODELLO
    # ==============================

    # cartella iniziale = home utente
    base_path = Path.home()

    # input testuale per navigare manualmente (opzionale)
    current_dir = st.text_input("Cartella corrente:", value=str(base_path))

    if not os.path.isdir(current_dir):
        st.error("Percorso non valido.")
        current_dir = str(base_path)

    try:
        files = os.listdir(current_dir)
        # mostra solo file comuni di modelli
        model_files = [f for f in files if f.endswith(('.pkl', '.pth', '.joblib', '.sav','.pt'))]
        selected_file = st.selectbox("Seleziona un file modello:", model_files)
        model_path = os.path.join(current_dir, selected_file) if selected_file else None
    except Exception as e:
        st.error(f"Errore nella navigazione: {e}")
        model_path = None

    st.caption(f"Path selezionato: `{model_path}`" if model_path else "Nessun file selezionato")

    model_params = st.text_area("Parametri del modello (JSON)", value="{}")

    if not os.path.isdir(current_dir):
        st.error("Percorso non valido.")
        current_dir = str(base_path)

    data = st.selectbox("Tipo di dato", ["singolo file", "folder"])
    if data=='folder':
        try:
            files = os.listdir(current_dir)
            dataset_dir = [d for d in files if os.path.isdir(os.path.join(current_dir,d))]
            selected_file = st.selectbox("Seleziona un file dataset:", dataset_dir)
            dataset_path = os.path.join(current_dir, selected_file) if selected_file else None
        except Exception as e:
            st.error(f"Errore nella navigazione: {e}")
            dataset_path = None

    elif data=="singolo file":
        try:
            files = os.listdir(current_dir)
            dataset_files = [f for f in files if f.endswith(('.csv'))]
            selected_file = st.selectbox("Seleziona un file dataset:", dataset_files)
            dataset_path = os.path.join(current_dir, selected_file) if selected_file else None
        except Exception as e:
            st.error(f"Errore nella navigazione: {e}")
            dataset_path = None

    creating = st.button("Crea progetto")

    # ==============================
    # CREAZIONE DEL PROGETTO
    # ==============================

    if creating is True:
        try:
            model_params = json.loads(model_params)
            data = None

            if dataset_path is not None:
                if dataset_type == "tabular" and data=="singolo file":
                    try:
                        data = pd.read_csv(dataset_path)
                        st.success(f"Dataset caricato con successo! {data.shape[0]} righe × {data.shape[1]} colonne.")
                        st.dataframe(data.head())
                    except Exception as e:
                        st.error(f"Errore nel caricamento del CSV: {e}")
                        return


            registry = ProjectRegistry(workspace)
            project = Project(
                project_name= project_name,
                data=dataset_path,
                dataset_type=dataset_type,
                framework=framework,
                model_path=model_path or None,
                model_params=model_params,
                workspace_path=workspace,
                target_variable=target_variable if target_variable!= None else None,
                categorical_columns=[c.strip() for c in categorical_columns.split(",")]  if categorical_columns!= None  else None,
                ordinal_columns=[c.strip() for c in ordinal_columns.split(",")]  if ordinal_columns!= None else None,
                device=device,
            )
            registry.add(project)
            st.success(f"Progetto creato con successo! ID: {project.project_name}")
            st.json(project.to_dict())

        except Exception as e:
            st.error(f"Errore nella creazione del progetto: {e}")