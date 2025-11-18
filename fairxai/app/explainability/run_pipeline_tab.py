import os

import streamlit as st

from fairxai.project.project_registry import ProjectRegistry


def run_pipeline_page():
    st.subheader("Esegui una pipeline di spiegazioni")

    workspace_path = os.path.abspath(os.path.normpath(os.path.join(os.getcwd(), "..","..","workspace")))

    registry = ProjectRegistry(workspace_path)
    project_ids = registry.list_all()

    if not project_ids:
        st.info("Nessun progetto disponibile per l’esecuzione.")
        return

    selected_id = st.selectbox("Seleziona un progetto:", project_ids)
    project = registry.load_project(selected_id)

    st.write(f"Project ID: `{project.id}` — Model: `{project.model_type}`")

    st.markdown("Carica pipeline YAML")
    uploaded_yaml = st.file_uploader("File YAML", type=["yaml", "yml"])

    if uploaded_yaml and st.button("Esegui pipeline YAML"):
        try:
            with open(os.path.join(project.workspace_path, "temp_pipeline.yaml"), "wb") as f:
                f.write(uploaded_yaml.read())

            results = project.run_pipeline_from_yaml(f.name)
            st.success(f"Pipeline eseguita con successo! {len(results)} spiegazioni generate.")
        except Exception as e:
            st.error(f"Errore durante l’esecuzione della pipeline: {e}")

    st.markdown("---")
    st.markdown("Oppure definisci una pipeline manuale")

    explainer = st.text_input("Explainer", "ShapExplainerAdapter")
    mode = st.selectbox("Modalità", ["global", "local"])
    instance_idx = st.number_input("Indice istanza (solo local)", min_value=0, value=0)

    if st.button("Esegui pipeline manuale"):
        pipeline = [
            {
                "explainer": explainer,
                "mode": mode,
                "params": {"instance_index": int(instance_idx)} if mode == "local" else {}
            }
        ]
        try:
            results = project.run_explanation_pipeline(pipeline)
            st.success(f"Pipeline eseguita ({len(results)} spiegazioni).")
            st.json(results[0])
        except Exception as e:
            st.error(f"Errore durante l’esecuzione: {e}")