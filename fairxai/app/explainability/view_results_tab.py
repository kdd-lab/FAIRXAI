import json
from pathlib import Path

import streamlit as st

from fairxai.app.explainability.visualization import visualize_explanation


def results_page():
    st.title("Visualizza risultati")

    project = st.session_state.get("current_project", None)

    if project:
        results_dir = Path(project.workspace_path) / "results"
        if not results_dir.exists():
            st.info("Nessun risultato trovato per questo progetto.")
            return

        result_files = list(results_dir.glob("*.json"))
        if not result_files:
            st.info("Nessun risultato trovato nella cartella results.")
            return

        selected_file = st.selectbox("Seleziona risultato:", [f.name for f in result_files])
        with open(results_dir / selected_file, "r") as f:
            data = json.load(f)

        st.markdown(f"### 📄 Risultato: `{selected_file}`")
        st.write(f"**Explainer:** {data['explainer']}")
        st.write(f"**Modalità:** {data['mode']}")
        st.write(f"**Timestamp:** {data['timestamp']}")
        st.write(f"**Tipo dati:** {project.dataset_type}")

        for expl in data.get("result", []):
            visualize_explanation(expl, data_type=project.dataset_type, instance_str=data['instance'])
    else:
        st.markdown(f"Caricare un progetto tra quelli a disposizione")
