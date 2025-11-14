import os
from pathlib import Path
import streamlit as st

from fairxai.project.project_registry import ProjectRegistry

def projects_page():

    st.header("Elenco progetti")
