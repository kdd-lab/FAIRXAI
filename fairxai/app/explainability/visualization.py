from collections import Counter

import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt


def visualize_explanation(expl):
    """
    Visualizza un singolo oggetto di spiegazione in Streamlit.
    Riconosce automaticamente il tipo di spiegazione (FeatureImportance,
    RuleBased, Counterfactual, Exemplar, Generic, ecc.).
    """
    expl_type = expl.get("explanation_type", "Unknown")
    st.markdown(f"### 🔍 Tipo di spiegazione: `{expl_type}`")

    # ============================================================
    # Feature Importance
    # ============================================================
    if expl_type == "FeatureImportanceExplanation":
        features = expl.get("sorted_features", [])
        importances = expl.get("sorted_importances", [])
        explainer_name = expl.get("explainer_name", "Explainer sconosciuto")
        if features and importances:
            df = pd.DataFrame({"Feature": features, "Importance": importances})
            df = df.sort_values("Importance", ascending=True)
            df = df[df["Importance"] > 0]

            # Crea il grafico matplotlib
            fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.25)))
            bars = ax.barh(df["Feature"], df["Importance"], color="#1f77b4")

            # Migliora la leggibilità
            ax.set_xlabel("Importanza")
            ax.set_ylabel("Feature")
            ax.set_title(f"Importanza delle feature - {explainer_name}")
            ax.invert_yaxis()  # la feature più importante in alto
            ax.grid(axis="x", linestyle="--", alpha=0.6)

            # Mostra valori su ogni barra (solo se poche feature)
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

            # Tabella espandibile con tutte le feature
            with st.expander("📋 Tabella completa delle feature importance"):
                st.dataframe(df.sort_values("Importance", ascending=False).reset_index(drop=True))
        else:
            st.info("Nessun valore di importanza disponibile.")

    # ============================================================
    # Rule-Based Explanation
    # ============================================================
    elif expl_type == "RuleBasedExplanation":
        expl_type = expl.get("explanation_type", "RuleBasedExplanation")
        explainer_name = expl.get("explainer_name", "Explainer sconosciuto")
        rules = expl.get("rules", [])

        st.markdown(f"Regole generate da **{explainer_name}** ({expl_type})")

        if not rules:
            st.info("Nessuna regola disponibile.")
            return

        # --- Analisi statistica sulle feature coinvolte nelle regole
        all_features = []
        for rule in rules:
            for p in rule.get("premises", []):
                all_features.append(p.get("attr"))

        if all_features:
            freq = Counter(all_features)
            df_freq = pd.DataFrame(freq.items(), columns=["Feature", "Occorrenze"]).sort_values("Occorrenze",
                                                                                                ascending=True)

            fig = px.bar(
                df_freq,
                x="Occorrenze",
                y="Feature",
                orientation="h",
                color="Occorrenze",
                color_continuous_scale="Viridis",
                title="Frequenza delle feature nelle regole",
            )
            fig.update_layout(
                height=max(400, len(df_freq) * 20),
                yaxis=dict(autorange="reversed"),
                plot_bgcolor="white",
                margin=dict(l=100, r=30, t=50, b=30),
            )
            st.plotly_chart(fig, use_container_width=True)

        # --- Visualizzazione dettagliata delle regole
        st.subheader("📜 Dettaglio delle regole")

        for i, rule in enumerate(rules):
            premises = rule.get("premises", [])
            consequence = rule.get("consequence", {})

            with st.expander(f"🔹 Regola #{i + 1}"):
                # Tabella delle premesse
                if premises:
                    df_premises = pd.DataFrame(premises)
                    st.markdown("**Premesse (SE):**")
                    st.dataframe(df_premises, use_container_width=True)
                else:
                    st.text("Nessuna premessa definita.")

                # Conseguenza evidenziata
                if consequence:
                    attr = consequence.get("attr", "?")
                    op = consequence.get("op", "=")
                    val = consequence.get("val", "?")
                    st.markdown(f"**Allora (→):** `{attr} {op} {val}`")
                else:
                    st.text("Nessuna conseguenza definita.")

        # Tabella riassuntiva finale
        with st.expander("📋 Tutte le regole in formato testo"):
            rules_text = []
            for rule in rules:
                parts = [f"{p.get('attr')} {p.get('op')} {p.get('val')}" for p in rule.get("premises", [])]
                cond = " AND ".join(parts)
                cons = rule.get("consequence", {})
                cons_str = f"{cons.get('attr')} {cons.get('op')} {cons.get('val')}"
                rules_text.append(f"SE {cond} → {cons_str}")
            st.write("\n\n".join(rules_text))

    # ============================================================
    # Counterfactual Rule Explanation
    # ============================================================
    elif expl_type in ["CounterfactualExplanation", "CounterfactualRuleExplanation"]:
        rules = expl.get("rules", [])
        if not rules:
            st.info("Nessuna regola controfattuale disponibile.")
        for i, rule in enumerate(rules):
            with st.expander(f"Counterfactual #{i + 1}"):
                premises = rule.get("premises", [])
                consequence = rule.get("consequence", {})
                if premises:
                    st.markdown("**Premesse:**")
                    st.dataframe(pd.DataFrame(premises))
                st.markdown("**Conseguenza:**")
                st.json(consequence)

    # ============================================================
    # Exemplar Explanation
    # ============================================================
    elif expl_type == "ExemplarExplanation":
        exemplars = expl.get("exemplars", [])
        counterexemplars = expl.get("counterexemplars", [])

        if exemplars:
            st.subheader("Esempi simili (Exemplars)")
            df_ex = pd.DataFrame(exemplars)
            st.dataframe(df_ex)
            if {"distance", "id"}.issubset(df_ex.columns):
                fig = px.scatter(df_ex, x="id", y="distance", title="Distanza dagli exemplars")
                st.plotly_chart(fig, use_container_width=True)

        if counterexemplars:
            st.subheader("Esempi contrastanti (Counterexemplars)")
            df_ce = pd.DataFrame(counterexemplars)
            st.dataframe(df_ce)
            if {"distance", "id"}.issubset(df_ce.columns):
                fig2 = px.scatter(df_ce, x="id", y="distance", title="Distanza dai counterexemplars")
                st.plotly_chart(fig2, use_container_width=True)

    # ============================================================
    # Generic or Global Explanation
    # ============================================================
    elif expl_type in ["GenericExplanation", "GlobalExplanation"]:
        values = expl.get("values") or expl.get("summary_values")
        if isinstance(values, dict):
            df = pd.DataFrame(values.items(), columns=["Feature", "Importance"])
            fig = px.bar(
                df.sort_values("Importance", ascending=False),
                x="Importance",
                y="Feature",
                orientation="h",
                title="Spiegazione globale",
                color="Importance",
                color_continuous_scale="Viridis",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.json(values or "Nessun valore disponibile.")

    # ============================================================
    # Text Explanation
    # ============================================================
    elif expl_type == "TextExplanation":
        tokens = expl.get("tokens", [])
        importances = expl.get("importance", [])
        if tokens and importances:
            st.subheader("🧠 Importanza delle parole")
            df_text = pd.DataFrame({"Token": tokens, "Importance": importances})
            fig = px.bar(df_text, x="Token", y="Importance", title="Importanza dei token")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.json(expl)

    # ============================================================
    # Fallback per altri tipi non riconosciuti
    # ============================================================
    else:
        st.info("Tipo di spiegazione non riconosciuto, mostro contenuto grezzo:")
        st.json(expl)
