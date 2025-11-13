import streamlit as st
from pathlib import Path
import webbrowser

from fairxai.app import home_page, explainability_page, doc_page


def app_main():

        st.set_page_config(page_title="FairXAI Platform", layout="wide")

        st.sidebar.title("FairXAI Dashboard")
        selected = st.sidebar.radio("Seleziona una sezione:",["Home", "Explainability", "Documentazione"])

        if selected == "Home":
            home_page.show()
        elif selected == "Explainability":
            explainability_page.show()
        elif selected == "Documentazione":
            doc_page.show()


if __name__ == "__main__":
    app_main()