import os

import streamlit as st

from fairxai.project.project_registry import ProjectRegistry


def run_pipeline_page():
    st.subheader("Esegui la spiegazione del progetto caricato")
    project = st.session_state.get("current_project", None)

    if project is None:
        st.warning("⚠️ Nessun progetto caricato. Vai prima nella tab 'Progetti' e carica un progetto.")
        return

    st.info(f"`{project.id}` ({project.model_type} su {project.dataset_type})")

    # Mostra gli explainers compatibili
    explainer_names = [cls.explainer_name for cls in project.explainers]
    if not explainer_names:
        st.error("Nessuno spiegatore compatibile trovato per questo progetto.")
        return

    selected_explainer = st.selectbox("Scegli uno spiegatore", explainer_names)
    mode = st.radio("Modalità di spiegazione", ["local", "global"], horizontal=True)

    params = {}
    if mode == "local":
        instance_index = st.number_input("Indice dell'istanza da spiegare", min_value=0, step=1)
        params["instance_index"] = int(instance_index)

    if st.button("Esegui spiegazione"):
        try:
            pipeline = [{"explainer": selected_explainer, "mode": mode, "params": params}]
            results = project.run_explanation_pipeline(pipeline)
            st.success(f"Spiegazione completata! {len(results)} risultato(i) generato/i.")
            result = results[0]

            st.markdown("### 📊 Risultato")
            st.json(result)

            # Eventuale rendering grafico se il risultato è un dizionario di feature importance
            if isinstance(result.get("result"), dict):
                result_values = result["result"]
                if all(isinstance(v, (int, float)) for v in result_values.values()):
                    st.bar_chart(result_values)

        except Exception as e:
            st.error(f"Errore nell'esecuzione dello spiegatore: {e}")