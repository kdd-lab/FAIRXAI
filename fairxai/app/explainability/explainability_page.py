from pathlib import Path
import streamlit as st

from fairxai.app.explainability.create_project_tab import create_project_page
from fairxai.app.explainability.project_tab import projects_page
from fairxai.app.explainability.run_pipeline_tab import run_pipeline_page
from fairxai.app.explainability.view_results_tab import results_page


def show():
    st.title("Spiegabilità dei modelli")
    create_tab, project_tab, run_pipeline_tab, run_result_tab = st.tabs(["Crea nuovo progetto","Progetti", "Esegui spiegazione","Risultati"])

    with project_tab:
        projects_page()

    with create_tab:
        create_project_page()

    with run_pipeline_tab:
        run_pipeline_page()

    with run_result_tab:
        results_page()