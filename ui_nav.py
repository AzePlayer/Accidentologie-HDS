# ui_nav.py
import streamlit as st

def nav_bar(active: str = "Accueil", ns: str = "nav_main"):
    """
    Affiche le bandeau de navigation.
    Paramètres:
      - active: "Accueil" | "Analyse temporelle" | "Analyse spatiale" | "Corrélations" | "Insights & recos"
      - ns: namespace pour fabriquer des clés uniques (utile si tu appelles nav_bar deux fois sur une même page)
    """
    st.markdown("")  # petit espace
    c1, c2, c3, c4, c5 = st.columns([1.1, 1.2, 1.2, 1.2, 1.3])

    # détecter une fois si switch_page est dispo
    if st.session_state.get("_has_switch_page") is None:
        try:
            _ = st.switch_page
            st.session_state["_has_switch_page"] = True
        except Exception:
            st.session_state["_has_switch_page"] = False

    def _btn(label: str, target: str, is_active: bool, key_suffix: str):
        style = {"type": "primary"} if is_active else {}
        # clé unique = namespace + suffixe
        key = f"{ns}_{key_suffix}"
        if st.button(label, key=key, use_container_width=True, **style):
            if st.session_state["_has_switch_page"]:
                st.switch_page(target)
            else:
                # Fallback: liens dans la sidebar si version Streamlit trop ancienne
                st.sidebar.page_link("pages/01_Accueil.py", label="🏠 Accueil")
                st.sidebar.page_link("pages/02_Analyse_temporelle.py", label="⏱️ Analyse temporelle")
                st.sidebar.page_link("pages/03_Analyse_spatiale.py", label="🗺️ Analyse spatiale")
                st.sidebar.page_link("pages/04_Correlations.py", label="🧮 Corrélations")
                st.sidebar.page_link("pages/05_Insights_recos.py", label="💡 Insights & recos")

    with c1: _btn("🏠 Accueil",            "pages/01_Accueil.py",            active == "Accueil",             "home")
    with c2: _btn("⏱️ Analyse temporelle","pages/02_Analyse_temporelle.py", active == "Analyse temporelle",  "time")
    with c3: _btn("🗺️ Analyse spatiale",  "pages/03_Analyse_spatiale.py",   active == "Analyse spatiale",    "spatial")
    with c4: _btn("🧮 Corrélations",       "pages/04_Correlations.py",       active == "Corrélations",        "corr")
    with c5: _btn("💡 Insights & recos",   "pages/05_Insights_recos.py",     active == "Insights & recos",    "insights")
