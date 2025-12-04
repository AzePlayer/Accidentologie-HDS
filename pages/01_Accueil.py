



# pages/01_Accueil.py — Accueil avec bandeau de navigation réutilisable
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from ui_nav import nav_bar

st.set_page_config(
    page_title="CrashAlert – Accueil",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===== Chargement données (session_state.df prioritaire) =====
def _load_local_csv(path="acc.csv") -> pd.DataFrame:
    try:
        df = pd.read_csv(path, sep=";", low_memory=False)
    except Exception:
        df = pd.read_csv(path, low_memory=False)

    df["date"] = pd.to_datetime(df.get("date"), errors="coerce", dayfirst=True)

    if "heure" in df.columns:
        h = df["heure"].astype(str).str.replace("h", ":", regex=False)
        df["heure_num"] = pd.to_datetime(h, errors="coerce").dt.hour
    else:
        df["heure_num"] = pd.to_numeric(df.get("heure_num"), errors="coerce")

    if "code_insee" in df.columns:
        df["code_insee"] = df["code_insee"].astype(str)
        dep = df["code_insee"].str[:2]
        dep = np.where(df["code_insee"].str.startswith("2A"), "2A",
              np.where(df["code_insee"].str.startswith("2B"), "2B", dep))
        df["departement"] = dep
    if "departement" not in df and "dep" in df:
        df["departement"] = df["dep"].astype(str)
    return df

if "df" in st.session_state and isinstance(st.session_state.df, pd.DataFrame) and not st.session_state.df.empty:
    df = st.session_state.df.copy()
else:
    df = _load_local_csv("acc.csv")

if df.empty:
    st.error("Aucune donnée trouvée. Place `acc.csv` à la racine ou charge depuis l’accueil (root).")
    st.stop()

# ===== Header (logo + titre + sous-titre) =====
col_logo, col_title = st.columns([0.10, 0.90])
with col_logo:
    st.markdown("### 🚗")
with col_title:
    st.markdown("# CrashAlert")
    st.caption("Visualiser les statistiques des accidents selon divers critères.")

# ===== Bandeau de navigation (réutilisable) =====
nav_bar(active="Accueil")

st.markdown("---")

# ===== KPIs — Jauges (titres visibles, couleur claire) =====
def gauge(value, title, max_val, color="#ff7f0e"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(value) if pd.notna(value) else 0.0,
        title={"text": title, "font": {"size": 16, "color": "#e6e6e6"}},
        number={"font": {"size": 38, "color": "#e6e6e6"}, "valueformat": ",.0f"},
        gauge={
            "axis": {"range": [0, max_val], "tickcolor": "#888"},
            "bar": {"color": color},
            "bgcolor": "rgba(0,0,0,0)"
        }
    ))
    fig.update_layout(
        height=220,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e6e6e6")
    )
    return fig

tot_acc  = len(df)
age_mean = df["age"].mean() if "age" in df.columns else (df["heure_num"].mean() if "heure_num" in df else np.nan)
tot_bless = df["nb_blesses"].sum() if "nb_blesses" in df.columns else np.nan
tot_deces = df["nb_tues"].sum()    if "nb_tues"    in df.columns else np.nan

g1, g2, g3, g4 = st.columns(4)
with g1: st.plotly_chart(gauge(tot_acc, "Total Accidents", max(100, int(tot_acc*1.1))), use_container_width=True)
with g2: st.plotly_chart(gauge(age_mean if not np.isnan(age_mean) else 0, "Âge moyen (ou heure moyenne)", 100), use_container_width=True)
with g3:
    if not np.isnan(tot_bless):
        st.plotly_chart(gauge(tot_bless, "Total Blessés", max(50, int(tot_bless*1.1))), use_container_width=True)
    else:
        st.empty()
with g4:
    if not np.isnan(tot_deces):
        st.plotly_chart(gauge(tot_deces, "Total Décès", max(10, int(tot_deces*1.1))), use_container_width=True)
    else:
        st.empty()

# ===== Contrôles =====
st.markdown("### ")
c_dep, c_var, c_type, c_pal = st.columns(4)

deps = sorted([d for d in df.get("departement", pd.Series(dtype=str)).dropna().unique()]) or ["(Tous)"]
dep_sel = c_dep.selectbox("Choisir un département :", deps, index=0)

def _categorical_candidates(data: pd.DataFrame):
    cands = []
    for c in data.columns:
        if c in ["date","heure","heure_num","annee","mois_str","jour_sem","jour","latitude","longitude","geo_point_2d"]:
            continue
        if pd.api.types.is_object_dtype(data[c]) or pd.api.types.is_categorical_dtype(data[c]):
            cands.append(c)
        elif pd.api.types.is_numeric_dtype(data[c]) and data[c].nunique(dropna=True) <= 20:
            cands.append(c)
    return sorted(list(dict.fromkeys(cands)))

cat_vars = _categorical_candidates(df)
default_var = "luminosite" if "luminosite" in cat_vars else (cat_vars[0] if cat_vars else None)
var_sel = c_var.selectbox("Choisir une variable :", cat_vars, index=cat_vars.index(default_var) if default_var in cat_vars else 0)

type_sel = c_type.selectbox("Choisir le type de graphique :", ["Camembert", "Barres", "Treemap"], index=0)

palettes = {
    "Set3": px.colors.qualitative.Set3,
    "Pastel": px.colors.qualitative.Pastel,
    "Bold": px.colors.qualitative.Bold,
    "Viridis": px.colors.sequential.Viridis
}
pal_sel = c_pal.selectbox("Choisir une palette de couleurs :", list(palettes.keys()), index=0)

# Actions
ca1, ca2, ca3 = st.columns([1,1,1])
valider = ca1.button("✅ Valider", use_container_width=True)
gen_narr = ca2.button("🖋️ Générer le narratif", use_container_width=True)
dl_placeholder = ca3.empty()

# ===== Données filtrées par département =====
df_viz = df.copy()
if dep_sel and dep_sel in df_viz.get("departement", pd.Series(dtype=str)).unique():
    df_viz = df_viz[df_viz["departement"] == dep_sel]

freq = (df_viz[var_sel]
        .fillna("Non renseigné")
        .value_counts(dropna=False)
        .rename_axis(var_sel)
        .reset_index(name="Frequence"))

# ===== Graphique =====
st.markdown(f"### Répartition de **{var_sel}**{f' – département {dep_sel}' if dep_sel else ''}")

if type_sel == "Camembert":
    fig = px.pie(freq, names=var_sel, values="Frequence", color=var_sel,
                 color_discrete_sequence=palettes[pal_sel], hole=0.0)
elif type_sel == "Barres":
    fig = px.bar(freq.sort_values("Frequence", ascending=False), x=var_sel, y="Frequence", color=var_sel,
                 color_discrete_sequence=palettes[pal_sel])
else:
    fig = px.treemap(freq, path=[var_sel], values="Frequence", color="Frequence",
                     color_continuous_scale="Viridis")

fig.update_layout(margin=dict(l=0, r=0, t=10, b=10), legend_title_text=var_sel)
st.plotly_chart(fig, use_container_width=True)

# ===== Narratif =====
def make_narrative(freq_df: pd.DataFrame, var_name: str) -> str:
    tot = int(freq_df["Frequence"].sum())
    top = freq_df.sort_values("Frequence", ascending=False).iloc[0]
    pct = 100 * top["Frequence"] / max(tot, 1)
    lines = [
        f"- Total observé : **{tot:,}** enregistrements.".replace(",", " "),
        f"- Catégorie dominante pour **{var_name}** : **{top[var_name]}** "
        f"({top['Frequence']:,} ; {pct:.1f}%).".replace(",", " "),
    ]
    if len(freq_df) >= 3:
        nxt = freq_df.sort_values("Frequence", ascending=False).iloc[1]
        pct2 = 100 * nxt["Frequence"] / max(tot, 1)
        lines.append(f"- Ensuite : **{nxt[var_name]}** ({nxt['Frequence']:,} ; {pct2:.1f}%).".replace(",", " "))
    return "\n".join(lines)

if gen_narr:
    st.markdown("#### Narratif")
    st.markdown(make_narrative(freq, var_sel))

# ===== Téléchargement du graphique =====
png_bytes = None
try:
    import plotly.io as pio  # noqa
    png_bytes = fig.to_image(format="png", scale=2)  # nécessite kaleido
except Exception:
    pass

if png_bytes is not None:
    dl_placeholder.download_button(
        "⬇️ Télécharger le graphique (PNG)",
        data=png_bytes,
        file_name=f"repartition_{var_sel}.png",
        mime="image/png",
        use_container_width=True
    )
else:
    html = fig.to_html(include_plotlyjs="cdn")
    dl_placeholder.download_button(
        "⬇️ Télécharger le graphique (HTML)",
        data=html.encode("utf-8"),
        file_name=f"repartition_{var_sel}.html",
        mime="text/html",
        use_container_width=True
    )


