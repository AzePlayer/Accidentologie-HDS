

# pages/04_Correlations.py — Page complète et autonome

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# (Optionnel) barre de navigation commune
try:
    from ui_nav import nav_bar
except Exception:
    nav_bar = None

# =========================
# Config page
# =========================
st.set_page_config(
    page_title="CrashAlert – Corrélations",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("# 📊 Corrélations")
st.caption("Mesures d’association selon la nature des variables : Pearson/Spearman, Cramér’s V (cat↔cat), corrélation-ratio η² (cat→num).")
if nav_bar:
    try:
        nav_bar(active="Corrélations", ns="corr_top")
    except Exception:
        pass

# =========================
# Utilitaires
# =========================
def _read_csv_smart(path: str = "acc.csv") -> pd.DataFrame:
    """Lecture robuste ; essaye ; comme séparateur et utf-8/latin-1."""
    for sep in (";", ","):
        for enc in ("utf-8", "latin-1"):
            try:
                return pd.read_csv(path, sep=sep, encoding=enc, low_memory=False)
            except Exception:
                continue
    # fallback
    return pd.read_csv(path, low_memory=False)

@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    # session_state prioritaire
    if "df" in st.session_state and isinstance(st.session_state.df, pd.DataFrame) and not st.session_state.df.empty:
        df = st.session_state.df.copy()
    else:
        df = _read_csv_smart("acc.csv")

    if df.empty:
        return df

    # date / heure minimal
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
    if "heure" in df.columns and "heure_num" not in df.columns:
        h = df["heure"].astype(str).str.replace("h", ":", regex=False)
        df["heure_num"] = pd.to_datetime(h, errors="coerce").dt.hour
    elif "heure_num" in df.columns:
        df["heure_num"] = pd.to_numeric(df["heure_num"], errors="coerce")

    # clés temporelles rapides
    if "date" in df.columns:
        df["annee"] = df["date"].dt.year
        df["mois"]  = df["date"].dt.to_period("M").astype(str)

    # géo : departement si code_insee
    if "code_insee" in df.columns:
        s = df["code_insee"].astype(str)
        dep = s.str[:2]
        dep = np.where(s.str.startswith("2A"), "2A",
              np.where(s.str.startswith("2B"), "2B", dep))
        df["departement"] = dep

    return df

def is_num(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)

def is_cat(s: pd.Series, max_unique: int = 40) -> bool:
    # objet/catégorie ou numérique à peu de modalités ⇒ considéré comme catégoriel
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_categorical_dtype(s) or pd.api.types.is_bool_dtype(s):
        return True
    if is_num(s):
        return s.dropna().nunique() <= max_unique
    return False

def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """V de Cramér pour cat↔cat."""
    try:
        tab = pd.crosstab(x, y)
        chi2 = ss_chi2(tab)
        n = tab.to_numpy().sum()
        r, k = tab.shape
        if n == 0:
            return np.nan
        return np.sqrt((chi2 / n) / (min(k - 1, r - 1)))
    except Exception:
        return np.nan

def ss_chi2(ct: pd.DataFrame) -> float:
    """Statistique chi² (attendus classiques)."""
    obs = ct.to_numpy(dtype=float)
    row = obs.sum(axis=1, keepdims=True)
    col = obs.sum(axis=0, keepdims=True)
    n   = obs.sum()
    exp = row @ col / max(n, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        chi = (obs - exp) ** 2 / np.where(exp == 0, np.nan, exp)
    return np.nansum(chi)

def correlation_ratio(categories: pd.Series, values: pd.Series) -> float:
    """η² (cat→num)."""
    try:
        # garder lignes non-nulles
        m = categories.notna() & values.notna()
        if m.sum() < 3:
            return np.nan
        cats = categories[m]
        vals = values[m].astype(float)
        grand_mean = vals.mean()
        # variance expliquée
        groups = vals.groupby(cats)
        ss_between = ((groups.mean() - grand_mean) ** 2 * groups.size()).sum()
        ss_total = ((vals - grand_mean) ** 2).sum()
        return float(ss_between / ss_total) if ss_total > 0 else np.nan
    except Exception:
        return np.nan

def topk_categories(s: pd.Series, k: int = 12) -> pd.Series:
    """Regroupe les modalités rares en 'Autres' pour lisibilité/perf."""
    vc = s.astype(str).fillna("NR").value_counts()
    keep = set(vc.head(k).index)
    return s.astype(str).fillna("NR").apply(lambda v: v if v in keep else "Autres")

# =========================
# Données
# =========================
df = load_data()
if df.empty:
    st.error("Aucune donnée. Place `acc.csv` à la racine ou charge d’abord depuis l’accueil.")
    st.stop()

st.success(f"Données chargées : **{len(df):,}** lignes × **{df.shape[1]}** colonnes".replace(",", " "))

# =========================
# Panneau de configuration (gauche)
# =========================
with st.container(border=True):
    st.markdown("## ⚙️ Configuration de l’analyse")
    c_dep, c_x, c_y = st.columns([1, 1, 1])

    # Département
    deps = sorted([d for d in df.get("departement", pd.Series(dtype=str)).dropna().unique()]) or ["(Tous)"]
    dep_sel = c_dep.selectbox("Département :", ["(Tous)"] + deps, index=0)

    # Variables candidates
    cols = list(df.columns)
    # éviter les clés très verbeuses par défaut
    ban = {"geo_point_2d"}
    cols = [c for c in cols if c not in ban]

    var_x = c_x.selectbox("Variable X :", cols, index=min(0, len(cols)-1))
    var_y = c_y.selectbox("Variable Y :", cols, index=min(1, len(cols)-1) if len(cols) > 1 else 0)

    # Choix du graphique
    gtype = st.selectbox("Type de visualisation :", ["Diagramme en barres", "Dispersion", "Boîte (box) / violon"])

    run_btn = st.button("▶️ Générer l’analyse", use_container_width=True)

# Appliquer filtre département
df_view = df.copy()
if dep_sel != "(Tous)" and "departement" in df_view.columns:
    df_view = df_view[df_view["departement"] == dep_sel]

# =========================
# BANDEAU : Visualisation des données (3 tuiles)
# =========================
st.markdown("## 📈 Visualisation des données")
st.markdown("<hr style='margin-top:-6px;margin-bottom:12px;border:1px solid #ff6f51;'>", unsafe_allow_html=True)

if "mini_viz" not in st.session_state:
    st.session_state.mini_viz = "obs"

c1, c2, c3 = st.columns(3)
with c1:
    if st.button("🧮  **Nombre d’observations**", use_container_width=True):
        st.session_state.mini_viz = "obs"
    st.markdown("<div style='height:6px;background:#dff0ff;border-radius:6px;margin-top:6px;'></div>", unsafe_allow_html=True)

with c2:
    if st.button("🎯  **Corrélation**", use_container_width=True):
        st.session_state.mini_viz = "corr"
    st.markdown("<div style='height:6px;background:#e8f6ea;border-radius:6px;margin-top:6px;'></div>", unsafe_allow_html=True)

with c3:
    if st.button("📈  **Tendance**", use_container_width=True):
        st.session_state.mini_viz = "trend"
    st.markdown("<div style='height:6px;background:#fff3e0;border-radius:6px;margin-top:6px;'></div>", unsafe_allow_html=True)

st.write("")

if st.session_state.mini_viz == "obs":
    k1, k2, k3, k4 = st.columns(4)
    with k1: st.metric("Lignes (vue courante)", f"{len(df_view):,}".replace(",", " "))
    with k2: st.metric("Colonnes", df_view.shape[1])
    with k3:
        nb_communes = df_view.get("code_insee", pd.Series(dtype=str)).dropna().nunique()
        st.metric("Communes (INSEE)", nb_communes)
    with k4:
        part_nan = 100 * (df_view.isna().mean().mean())
        st.metric("%% manquants (moy.)", f"{part_nan:.1f}%")

    # petit top 10
    cats = [c for c in df_view.columns if is_cat(df_view[c])]
    if cats:
        top_var = st.selectbox("Top-10 d’une variable catégorielle :", cats, index=0, key="obs_top_cat")
        top_df = (df_view[top_var].fillna("Non renseigné")
                  .value_counts().head(10).rename_axis(top_var).reset_index(name="Effectif"))
        fig = px.bar(top_df, x=top_var, y="Effectif")
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=40))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Aucune variable catégorielle détectée pour un top-10.")

elif st.session_state.mini_viz == "corr":
    st.caption("Aperçu rapide : **Pearson** sur variables **numériques** (limite de colonnes pour lisibilité).")
    num_df = df_view.select_dtypes(include=["number"]).copy()
    num_df = num_df.loc[:, num_df.notna().sum() >= max(30, int(0.6 * len(num_df)))]
    if num_df.shape[1] >= 2:
        max_cols = st.slider("Nombre max. de variables à afficher", 5, 25, min(12, num_df.shape[1]))
        cols_show = list(num_df.columns)[:max_cols]
        corr = num_df[cols_show].corr(method="pearson")
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                        origin="lower", aspect="auto")
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Pas assez de variables numériques pour une matrice.")
else:
    if "mois" in df_view.columns:
        evo = df_view["mois"].value_counts().sort_index().reset_index()
        evo.columns = ["Mois", "Accidents"]
        fig = px.line(evo, x="Mois", y="Accidents", markers=True)
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Colonne `date` absente : impossible de tracer une tendance mensuelle.")

st.markdown("---")

# =========================
# Visualisation X/Y (selon gtype)
# =========================
st.markdown("## 🖼️ Visualisation choisie")

if run_btn:
    sx = df_view[var_x]
    sy = df_view[var_y]

    if gtype == "Diagramme en barres":
        # si X cat, barres de fréquence (ou empilé par Y si cat)
        if is_cat(sx):
            x_clean = topk_categories(sx, k=15)
            if is_cat(sy):
                dfp = (pd.DataFrame({"X": x_clean, "Y": topk_categories(sy, k=8)})
                       .value_counts().reset_index(name="Effectif"))
                fig = px.bar(dfp, x="X", y="Effectif", color="Y", barmode="stack")
            else:
                dfp = x_clean.value_counts().reset_index()
                dfp.columns = ["X", "Effectif"]
                fig = px.bar(dfp, x="X", y="Effectif")
        else:
            st.info("X n’est pas catégorielle : sélectionne plutôt « Dispersion » ou « Boîte/violon ».") 
            fig = None

    elif gtype == "Dispersion":
        if is_num(sx) and is_num(sy):
            fig = px.scatter(df_view, x=var_x, y=var_y, opacity=0.6, trendline="ols")
        elif is_num(sx) and is_cat(sy):
            fig = px.box(pd.DataFrame({var_y: topk_categories(sy), var_x: sx}), x=var_y, y=var_x)
        elif is_cat(sx) and is_num(sy):
            fig = px.box(pd.DataFrame({var_x: topk_categories(sx), var_y: sy}), x=var_x, y=var_y)
        else:
            st.info("Deux variables catégorielles : préfère « Diagramme en barres ».") 
            fig = None

    else:  # Boîte / violon
        if is_cat(sx) and is_num(sy):
            d = pd.DataFrame({var_x: topk_categories(sx), var_y: sy})
            fig = px.violin(d, x=var_x, y=var_y, box=True, points="suspectedoutliers")
        elif is_cat(sy) and is_num(sx):
            d = pd.DataFrame({var_y: topk_categories(sy), var_x: sx})
            fig = px.violin(d, x=var_y, y=var_x, box=True, points="suspectedoutliers")
        else:
            st.info("Ce type suppose 1 cat + 1 num. Utilise « Dispersion » ou « Barres ».") 
            fig = None

    if fig is not None:
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

# =========================
# Matrices de corrélations ciblées
# =========================
st.markdown("## 🧠 Corrélations ciblées (lisibles)")

with st.expander("Matrice Pearson (numérique ↔ numérique)", expanded=False):
    num_cols = [c for c in df_view.columns if is_num(df_view[c])]
    if len(num_cols) >= 2:
        # tri par variance décroissante (plus informatif)
        num_cols = sorted(num_cols, key=lambda c: df_view[c].var(skipna=True), reverse=True)
        maxc = st.slider("Variables max à afficher", 5, 25, min(12, len(num_cols)), key="pear_max")
        cols_show = num_cols[:maxc]
        corr = df_view[cols_show].corr("pearson")
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                        origin="lower", aspect="auto")
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Pas assez de variables numériques.")

with st.expander("V de Cramér (catégorielle ↔ catégorielle, modalités regroupées)", expanded=False):
    cat_cols = [c for c in df_view.columns if is_cat(df_view[c])]
    if len(cat_cols) >= 2:
        # regrouper raretés pour chaque colonne (pour des tables raisonnables)
        capped = {c: topk_categories(df_view[c], k=10) for c in cat_cols}
        M = np.zeros((len(cat_cols), len(cat_cols)))
        for i, ci in enumerate(cat_cols):
            for j, cj in enumerate(cat_cols):
                if j < i:
                    M[i, j] = M[j, i]
                else:
                    M[i, j] = 1.0 if i == j else cramers_v(capped[ci], capped[cj])
        fig = px.imshow(pd.DataFrame(M, index=cat_cols, columns=cat_cols),
                        color_continuous_scale="Blues", origin="lower", zmin=0, zmax=1, text_auto=".2f")
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Pas assez de variables catégorielles.")

with st.expander("Corrélation-ratio η² (catégorie → numérique)", expanded=False):
    cat_cols = [c for c in df_view.columns if is_cat(df_view[c])]
    num_cols = [c for c in df_view.columns if is_num(df_view[c])]
    if cat_cols and num_cols:
        c_sel = st.selectbox("Variable catégorielle (ex. type d’accident, luminosité…)", cat_cols, key="eta_cat")
        n_sel = st.selectbox("Variable numérique (ex. heure, âges, comptages…)", num_cols, key="eta_num")
        val = correlation_ratio(df_view[c_sel], df_view[n_sel])
        st.metric("η² (0→1)", f"{val:0.3f}")
        st.caption("Interprétation indicative : 0.01 faible • 0.06 modérée • 0.14 forte (seuils Cohen, contextuels).")
    else:
        st.info("Sélectionne au moins 1 catégorielle et 1 numérique.")

st.markdown("---")

# =========================
# Analyse narrative & statistiques
# =========================
st.markdown("## 🔎 Analyse approfondie")

with st.container(border=True):
    st.markdown("### Interprétation des résultats (auto)")
    # repérer quelques signaux : corr Pearson > 0.3 (absolu) ou Cramér > 0.2
    findings = []

    # Pearson rapide sur 8 principales numériques
    num_cols = [c for c in df_view.columns if is_num(df_view[c])]
    if len(num_cols) >= 2:
        num_cols = sorted(num_cols, key=lambda c: df_view[c].var(skipna=True), reverse=True)[:8]
        corr = df_view[num_cols].corr("pearson").abs()
        tri = (corr.where(~np.eye(len(corr), dtype=bool)).stack().sort_values(ascending=False))
        for (a, b), v in tri.head(5).items():
            if v >= 0.30:
                findings.append(f"Corrélation **Pearson** notable : **{a}** ↔ **{b}** (|r|={v:.2f}).")

    # Cramér sur 8 principales catégorielles
    cat_cols = [c for c in df_view.columns if is_cat(df_view[c])]
    if len(cat_cols) >= 2:
        cat_cols = cat_cols[:8]
        capped = {c: topk_categories(df_view[c], k=10) for c in cat_cols}
        for i in range(len(cat_cols)):
            for j in range(i+1, len(cat_cols)):
                v = cramers_v(capped[cat_cols[i]], capped[cat_cols[j]])
                if v >= 0.20:
                    findings.append(f"Association **Cramér’s V** : **{cat_cols[i]}** ↔ **{cat_cols[j]}** (V={v:.2f}).")

    if findings:
        st.markdown("- " + "\n- ".join(findings))
    else:
        st.info("Pas d’association forte détectée avec les seuils rapides (ajuste les sélections ci-dessus).")

with st.container(border=True):
    st.markdown("### Statistiques descriptives")
    # mini table stats sur X et Y
    cols_show = [var_x, var_y]
    desc = {}
    for c in cols_show:
        s = df_view[c]
        if is_num(s):
            d = s.describe(percentiles=[0.05, 0.5, 0.95])
            desc[c] = {k: float(v) for k, v in d.items()}
        else:
            vc = s.astype(str).value_counts().head(5)
            desc[c] = {f"Top{i+1}": f"{idx} ({val})" for i, (idx, val) in enumerate(vc.items())}
    st.json(desc)
