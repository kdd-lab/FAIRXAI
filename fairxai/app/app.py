from pathlib import Path

import streamlit as st

from fairxai.app import home_page, doc_page
from fairxai.app.explainability import explainability_page


def app_main():

        st.set_page_config(page_title="FairXAI Platform", layout="wide")
        logo_path = Path(__file__).resolve().parent / "assets" / "Logo_FAIR.png"


        with st.sidebar:
            if logo_path.exists():
                st.image(str(logo_path), use_container_width=True)
            st.markdown(
                "<h2 style='text-align:center; margin-top: -10px;'>FairXAI Dashboard</h2>",
                unsafe_allow_html=True
            )

            selected = st.radio(
                "Seleziona una sezione:",
                ["Home", "Explainability", "Documentazione"],
                index=0
            )

        if selected == "Home":
            home_page.show()
        elif selected == "Explainability":
            explainability_page.show()
        elif selected == "Documentazione":
            doc_page.show()


if __name__ == "__main__":
    app_main()