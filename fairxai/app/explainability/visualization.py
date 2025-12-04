import base64
from io import BytesIO
from collections import Counter

import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from PIL import Image
import re


def visualize_explanation(expl: dict, data_type: str = "tabular", instance_str: str = None):
    """
    Visualizza una singola spiegazione in Streamlit, differenziando
    tra dataset tabulari e immagini.

    Parameters
    ----------
    expl : dict
        Dizionario contenente la spiegazione (FeatureImportance, RuleBased, ecc.)
    data_type : str
        Tipo di dataset: 'tabular' o 'image'
    """

    expl_type = expl.get("explanation_type", "Unknown")
    explainer_name = expl.get("explainer_name", "Explainer sconosciuto")

    st.markdown(f"### 🔍 Tipo di spiegazione: `{expl_type}` — **{explainer_name}**")

    # ============================================================
    # FEATURE IMPORTANCE
    # ============================================================
    if expl_type == "FeatureImportanceExplanation":
        if data_type == "tabular":
            features = expl.get("sorted_features", [])
            importances = expl.get("sorted_importances", [])
            if features and importances:
                df = pd.DataFrame({"Feature": features, "Importance": importances})
                df = df.sort_values("Importance", ascending=True)
                df = df[df["Importance"] > 0]

                fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.25)))
                bars = ax.barh(df["Feature"], df["Importance"], color="#1f77b4")
                ax.set_xlabel("Importanza")
                ax.set_ylabel("Feature")
                ax.set_title(f"Importanza delle feature - {explainer_name}")
                ax.invert_yaxis()
                ax.grid(axis="x", linestyle="--", alpha=0.6)

                if len(df) <= 20:
                    for bar in bars:
                        width = bar.get_width()
                        ax.text(
                            width + max(importances) * 0.01,
                            bar.get_y() + bar.get_height() / 2,
                            f"{width:.3f}",
                            va="center",
                            fontsize=8,
                        )

                st.pyplot(fig)
                with st.expander("📋 Tabella completa delle feature importance"):
                    st.dataframe(df.sort_values("Importance", ascending=False).reset_index(drop=True))
            else:
                st.info("Nessun valore di importanza disponibile.")

        elif data_type == "image":
            visualization = expl.get("visualization", {})
            heatmap_b64 = visualization.get("heatmap_base64")
            original_size = visualization.get("original_size", None)

            original_img = None

            if instance_str:
                try:
                    clean = re.sub(r'\.\.\.', '', instance_str)
                    numbers = list(map(int, re.findall(r'\d+', clean)))
                    arr = np.array(numbers, dtype=np.uint8)

                    arr = arr.reshape((6, 6, 3))

                    st.image(arr, caption="🖼️ Immagine originale ricostruita")

                except Exception as e:
                    st.warning(f"⚠️ Errore nella ricostruzione dell'immagine originale: {e}")

            # 🔹 Heatmap Grad-CAM
            if heatmap_b64:
                try:
                    heatmap_data = base64.b64decode(heatmap_b64)
                    heatmap_img = Image.open(BytesIO(heatmap_data))

                    # Ridimensiona la heatmap all'eventuale dimensione originale
                    if original_size:
                        heatmap_img = heatmap_img.resize(tuple(original_size), Image.LANCZOS)
                    elif original_img:
                        heatmap_img = heatmap_img.resize(original_img.size, Image.LANCZOS)

                    # 🔹 Visualizza affiancate
                    col1, col2 = st.columns(2)
                    with col1:
                        if original_img:
                            st.image(original_img, caption="🖼️ Immagine originale", use_container_width=True)
                        else:
                            st.info("Immagine originale non disponibile.")
                    with col2:
                        st.image(heatmap_img, caption="🔥 Heatmap (GradCAM)", use_container_width=True)

                except Exception as e:
                    st.error(f"Errore nella decodifica o ridimensionamento della heatmap: {e}")
            else:
                st.info("Nessuna heatmap disponibile per questa spiegazione.")
    # ============================================================
    # RULE-BASED
    # ============================================================
    elif expl_type == "RuleBasedExplanation":
        rules = expl.get("rules", [])
        if not rules:
            st.info("Nessuna regola disponibile.")
            return

        if data_type == "tabular":
            # Frequenza delle feature
            all_features = [p.get("attr") for r in rules for p in r.get("premises", [])]
            if all_features:
                freq = Counter(all_features)
                df_freq = pd.DataFrame(freq.items(), columns=["Feature", "Occorrenze"]).sort_values("Occorrenze", ascending=True)
                fig = px.bar(
                    df_freq,
                    x="Occorrenze",
                    y="Feature",
                    orientation="h",
                    color="Occorrenze",
                    color_continuous_scale="Viridis",
                    title="Frequenza delle feature nelle regole",
                )
                fig.update_layout(height=max(400, len(df_freq) * 20), yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("📜 Regole generate")
            for i, rule in enumerate(rules):
                with st.expander(f"🔹 Regola #{i + 1}"):
                    premises = rule.get("premises", [])
                    consequence = rule.get("consequence", {})
                    if premises:
                        st.dataframe(pd.DataFrame(premises))
                    if consequence:
                        st.markdown(f"**→ Conseguenza:** `{consequence.get('attr')} {consequence.get('op')} {consequence.get('val')}`")

        elif data_type == "image":
            st.info("Regole derivate da explainers basati su regioni o segmenti (non ancora visualizzati).")
            st.json(rules)

    # ============================================================
    # COUNTERFACTUAL (REGOLA o SEMPLICE)
    # ============================================================
    elif expl_type in ["CounterfactualExplanation", "CounterfactualRuleExplanation"]:
        rules = expl.get("rules", [])
        if not rules:
            st.info("Nessuna regola controfattuale disponibile.")
            return

        if data_type == "tabular":
            st.subheader("📉 Regole controfattuali (tabular)")
            for i, rule in enumerate(rules):
                with st.expander(f"Counterfactual #{i + 1}"):
                    premises = rule.get("premises", [])
                    consequence = rule.get("consequence", {})
                    if premises:
                        st.dataframe(pd.DataFrame(premises))
                    if consequence:
                        st.markdown(f"**→ Conseguenza:** `{consequence.get('attr')} {consequence.get('op')} {consequence.get('val')}`")

        elif data_type == "image":
            st.subheader("📉 Controfatti per immagini")
            visualization = expl.get("visualization", {})
            if "heatmap_base64" in visualization:
                image_data = base64.b64decode(visualization["heatmap_base64"])
                image = Image.open(BytesIO(image_data))
                st.image(image, caption="Regione controfattuale", use_container_width=True)
            else:
                st.json(rules)

    # ============================================================
    # EXEMPLAR / CONTROESEMPLAR
    # ============================================================
    elif expl_type == "ExemplarExplanation":
        exemplars = expl.get("exemplars", [])
        counterexemplars = expl.get("counterexemplars", [])

        if data_type == "tabular":
            if exemplars:
                st.subheader("Esempi simili (Exemplars)")
                st.dataframe(pd.DataFrame(exemplars))
            if counterexemplars:
                st.subheader("Esempi contrastanti (Counterexemplars)")
                st.dataframe(pd.DataFrame(counterexemplars))

        elif data_type == "image":
            st.subheader("📸 Esempi simili (immagini)")
            for ex in exemplars:
                b64img = ex.get("image_base64")
                if b64img:
                    image_data = base64.b64decode(b64img)
                    st.image(Image.open(BytesIO(image_data)), caption="Exemplar")
            for ce in counterexemplars:
                b64img = ce.get("image_base64")
                if b64img:
                    image_data = base64.b64decode(b64img)
                    st.image(Image.open(BytesIO(image_data)), caption="Counterexemplar")

    # ============================================================
    # FALLBACK
    # ============================================================
    else:
        st.info("Tipo di spiegazione non riconosciuto, mostro contenuto grezzo:")
        st.json(expl)
