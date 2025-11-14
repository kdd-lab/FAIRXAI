import json
import os
import streamlit as st
from fairxai.project.project_registry import ProjectRegistry

def results_page():
    st.header("Risultati delle spiegazioni")
