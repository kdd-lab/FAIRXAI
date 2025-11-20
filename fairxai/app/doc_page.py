import streamlit as st
from pathlib import Path
import webbrowser

def show():
    st.title("Documentazione tecnica")

    docs_build = Path(__file__).resolve().parents[2] / "docs" / "build" / "html" / "index.html"

    if docs_build.exists():
        if st.button("Apri la documentazione completa"):
            webbrowser.open_new_tab(docs_build.as_uri())
    else:
        st.error(f"Non trovo il file {docs_build}")
