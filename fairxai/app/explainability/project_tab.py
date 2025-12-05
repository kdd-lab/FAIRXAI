import os

import streamlit as st

from fairxai.project.project_registry import ProjectRegistry


def projects_page():
    st.markdown("### Seleziona il workspace")

    # Workspace corrente (manteniamo in sessione)
    workspace = os.path.abspath(os.path.normpath(os.path.join(os.getcwd(), "..", "..", "workspace")))
    if "current_workspace" not in st.session_state:
        st.session_state["current_workspace"] = workspace

    current_workspace = st.text_input(
        "Percorso del workspace",
        value=st.session_state["current_workspace"],
        help="Inserisci o seleziona il percorso dove sono salvati i progetti FAIRXAI"
    )

    if st.button("Cambia workspace"):
        try:
            os.makedirs(current_workspace, exist_ok=True)
            st.session_state["current_workspace"] = current_workspace
            st.success(f"Workspace impostato su: `{current_workspace}`")
        except Exception as e:
            st.error(f"Errore nel creare o accedere al workspace: {e}")

    workspace_path = st.session_state["current_workspace"]
    st.markdown(f"**Workspace attuale:** `{workspace_path}`")

    # Caricamento dei progetti
    registry = ProjectRegistry(workspace_path)
    projects = registry.list_all()

    st.markdown("---")
    st.markdown("### 📜 Elenco dei progetti")

    if not projects:
        st.info("Nessun progetto trovato nel workspace selezionato.")
        return

    project_names = {p["id"]: f'{p.get("project_name", "Project")} - {p.get("dataset_type")} - {p.get("model_type")}'
                     for p in projects}
    selected_id = st.selectbox("Seleziona progetto:", list(project_names.keys()),
                               format_func=lambda x: project_names[x])

    # Salva progetto selezionato in session_state
    st.session_state.selected_project_id = selected_id
    st.session_state.registry = registry
    metadata = registry.get_metadata(selected_id)

    if metadata:
        st.markdown("### 🧾 Metadati del progetto")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**ID:** {metadata.get('id')}")
            st.write(f"**Creato il:** {metadata.get('created_at')}")
            st.write(f"**Dataset:** {metadata.get('dataset_type')}")
            st.write(f"**Modello:** {metadata.get('model_type')}")
        with col2:
            st.write(f"**Explainers compatibili:** {metadata.get('num_explainers')}")
            st.write(f"**Spiegazioni generate:** {metadata.get('num_explanations')}")
            st.write(f"**Workspace:** `{metadata.get('workspace_path')}`")

        with st.expander("📄 Visualizza JSON completo"):
            st.json(metadata)

        # Caricamento progetto
        if st.button("📦 Carica progetto in memoria"):
            try:
                project = registry.load_project(selected_id)
                st.session_state["current_project"] = project
                st.success(f"Progetto `{project.id}` caricato in memoria.")
            except Exception as e:
                st.error(f"Errore nel caricamento del progetto: {e}")
    else:
        st.error("Metadati non trovati o non validi per il progetto selezionato.")
