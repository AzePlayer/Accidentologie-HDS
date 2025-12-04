import streamlit as st
import pandas as pd

st.set_page_config(page_title="Accidentologie", page_icon="🚦", layout="wide")
st.title("🚦 Dashboard Accidentologie – Accueil (root)")

# charge acc.csv si présent
try:
    df = pd.read_csv("acc.csv", sep=";", low_memory=False)
    st.success(f"{len(df):,} lignes chargées depuis acc.csv".replace(",", " "))
    st.dataframe(df.head(), use_container_width=True)
except Exception as e:
    st.info("Placez un acc.csv à la racine ou allez dans Pages ➜ 01_Accueil.")
    st.caption(f"(Détail : {e})")
