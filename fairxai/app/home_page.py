import streamlit as st

def show():
    st.title("FairXAI Platform")
    st.write("Benvenuto nella dashboard di FairXAI!")
    st.markdown("""
        FAIRXAI è una piattaforma sviluppata da Kode all'interno del bando di ricerca e sviluppo Future Artificial Intelligence in Research (FAIR), progettata per la composizione, l'esecuzione e la spiegazione di processi di AI decisio-making.
        
        Diversamente dai tradizionali strumenti di ExplainableAI, FAIRXAI permette di avere un toolkit di spiegazione, integrando diversi spiegazioni e gestendo tipi diversi di dato, distinguendosi per fruibilità di utilizzo sia in modo programmatico, sia in modo grafico grazie ad una UI essenziale.
    """)