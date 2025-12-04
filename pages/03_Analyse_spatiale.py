

# pages/03_Analyse_spatiale.py — version optimisée (FR)
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# Bandeau (optionnel)
try:
    from ui_nav import nav_bar
except Exception:
    nav_bar = None

# -------------------------------
# Configuration de la page
# -------------------------------
st.set_page_config(
    page_title="CrashAlert – Analyse spatiale",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("# Analyse spatiale")
st.caption("Localisation des accidents, zoom Hauts-de-Seine, heatmap, clustering et exports.")
if nav_bar:
    try:
        nav_bar(active="Analyse spatiale", ns="spatial_top")
    except Exception:
        pass

# -------------------------------
# Utilitaires
# -------------------------------
def get_secret(name: str, default=None):
    """st.secrets[name] si dispo, sinon variable d'environnement, sinon default."""
    try:
        return st.secrets[name]
    except Exception:
        return os.environ.get(name, default)

@st.cache_data(show_spinner=False)
def _load_local_csv(path: str = "acc.csv") -> pd.DataFrame:
    """Lecture + normalisation minimale ; gère 'geo_point_2d' -> latitude/longitude."""
    try:
        df = pd.read_csv(path, sep=";", low_memory=False)
    except Exception:
        df = pd.read_csv(path, low_memory=False)

    if df.empty:
        return df

    # Dates / heures
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
    else:
        df["date"] = pd.NaT

    if "heure" in df.columns and "heure_num" not in df.columns:
        h = df["heure"].astype(str).str.replace("h", ":", regex=False)
        df["heure_num"] = pd.to_datetime(h, errors="coerce").dt.hour
    elif "heure_num" in df.columns:
        df["heure_num"] = pd.to_numeric(df["heure_num"], errors="coerce")
    else:
        df["heure_num"] = np.nan

    df["annee"] = df["date"].dt.year

    # Géo : priorité lat/lon, sinon geo_point_2d "lat, lon"
    has_latlon = ("latitude" in df.columns) and ("longitude" in df.columns)
    if not has_latlon and "geo_point_2d" in df.columns:
        sp = df["geo_point_2d"].astype(str).str.split(",", n=1, expand=True)
        if sp.shape[1] == 2:
            df["latitude"]  = pd.to_numeric(sp[0].str.strip(), errors="coerce")
            df["longitude"] = pd.to_numeric(sp[1].str.strip(), errors="coerce")

    for c in ("latitude", "longitude"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "code_insee" in df.columns:
        df["code_insee"] = df["code_insee"].astype(str)

    return df

# Priorité aux données déjà chargées depuis l’Accueil
if "df" in st.session_state and isinstance(st.session_state.df, pd.DataFrame) and not st.session_state.df.empty:
    df = st.session_state.df.copy()
else:
    df = _load_local_csv("acc.csv")

if df.empty:
    st.error("Aucune donnée trouvée. Place `acc.csv` à la racine du projet ou ouvre d’abord la page Accueil.")
    st.stop()

if ("latitude" not in df.columns) or ("longitude" not in df.columns):
    st.error("Colonnes **latitude/longitude** absentes (ou `geo_point_2d` non exploitable). Impossible d’afficher la carte.")
    st.stop()

df = df.dropna(subset=["latitude", "longitude"])
if df.empty:
    st.info("Aucun point géolocalisé exploitable après nettoyage.")
    st.stop()

# -------------------------------
# Filtres (barre latérale)
# -------------------------------
st.sidebar.header("⚙️ Paramètres – Spatial")

years = sorted([int(a) for a in df["annee"].dropna().unique()]) if "annee" in df else []
y_min, y_max = st.sidebar.select_slider("Période (années)", options=years, value=(years[0], years[-1])) if years else (None, None)

hmin, hmax = st.sidebar.select_slider("Plage horaire", options=list(range(24)), value=(6, 20))

insee_opts = sorted(df["code_insee"].dropna().unique()) if "code_insee" in df else []
insee_sel = st.sidebar.multiselect("Filtre commune (code INSEE)", insee_opts, default=[])

# Options d’affichage
st.sidebar.markdown("### Mode carte (Plotly)")
base_map = st.sidebar.selectbox(
    "Fond de carte",
    ["OpenStreetMap", "Heatmap (densité)", "Mapbox Streets", "Mapbox Satellite"],
    index=0
)

# Options performance / confort
st.sidebar.markdown("### Affichage & performance")
fast_mode  = st.sidebar.toggle("⚡ Mode rapide (optimisé)", value=True)
max_points = st.sidebar.slider("Points max affichés (points bruts / clustering)", 2_000, 50_000, 15_000, step=1_000)
point_size = st.sidebar.slider("Taille des points (Plotly)", 1, 12, 3)
point_alpha = st.sidebar.slider("Opacité des points", 0.2, 1.0, 0.6)
dens_radius = st.sidebar.slider("Rayon densité (si Heatmap)", 5, 60, 18)
scroll_zoom = st.sidebar.toggle("Zoom à la molette (Plotly)", True)

# Zone d’affichage par défaut
st.sidebar.markdown("### Zone par défaut")
view_area = st.sidebar.selectbox(
    "Choisir la zone",
    ["Hauts-de-Seine (92)", "Étendue des données", "France (large)"],
    index=0
)

# -------------------------------
# Application des filtres
# -------------------------------
mask = pd.Series(True, index=df.index)
if y_min is not None:
    mask &= df["annee"].between(y_min, y_max)
mask &= df["heure_num"].between(hmin, hmax) | df["heure_num"].isna()
if insee_sel:
    mask &= df["code_insee"].isin(insee_sel)

df_f = df[mask].copy()
if df_f.empty:
    st.info("Aucune donnée avec ces filtres.")
    st.stop()

# KPIs rapides
c1, c2, c3 = st.columns(3)
c1.metric("Accidents filtrés", f"{len(df_f):,}".replace(",", " "))
zone_top = df_f["code_insee"].mode().iloc[0] if "code_insee" in df_f and df_f["code_insee"].dropna().size else "—"
c2.metric("Zone la + touchée (INSEE)", zone_top)
modal_hour = int(df_f["heure_num"].mode().iloc[0]) if df_f["heure_num"].dropna().size else "—"
c3.metric("Heure modale (global)", modal_hour)
st.markdown("---")

# -------------------------------
# Pré-agrégations utiles
# -------------------------------
has_insee = "code_insee" in df_f.columns and df_f["code_insee"].notna().any()
agg = pd.DataFrame()
if has_insee:
    agg = (
        df_f.groupby("code_insee")
        .agg(accidents=("code_insee", "size"),
             lat=("latitude", "mean"), lon=("longitude", "mean"))
        .reset_index()
    )

# -------------------------------
# Zone : centrage / zoom / découpe
# -------------------------------
HDS_CENTER = (48.85, 2.27)               # centre approximatif 92 (Nanterre)
HDS_ZOOM   = 11
HDS_BBOX   = (48.75, 48.95, 2.15, 2.35)  # lat_min, lat_max, lon_min, lon_max

def crop_to_bbox(d: pd.DataFrame, bbox=HDS_BBOX) -> pd.DataFrame:
    if {"latitude","longitude"}.issubset(d.columns):
        lat_min, lat_max, lon_min, lon_max = bbox
        return d[(d["latitude"].between(lat_min, lat_max)) &
                 (d["longitude"].between(lon_min, lon_max))]
    return d

if view_area == "Hauts-de-Seine (92)":
    center_lat, center_lon, zoom = *HDS_CENTER, HDS_ZOOM
    df_view = crop_to_bbox(df_f, HDS_BBOX)
elif view_area == "Étendue des données":
    center_lat = float(df_f["latitude"].median())
    center_lon = float(df_f["longitude"].median())
    zoom = 8
    df_view = df_f
else:
    center_lat, center_lon, zoom = 46.5, 2.2, 5
    df_view = df_f

centre = {"lat": center_lat, "lon": center_lon}

# -------------------------------
# Fonctions d’aide (perf)
# -------------------------------
def downsample(d: pd.DataFrame, nmax: int) -> pd.DataFrame:
    if len(d) <= nmax:
        return d
    return d.sample(nmax, random_state=42)

# -------------------------------
# Onglets : Plotly / Folium
# -------------------------------
tabs = st.tabs(["🧭 Plotly", "🧩 Folium (clustering)"])

# === PLOTLY ============================================================
with tabs[0]:
    st.subheader("Carte interactive (Plotly)")

    # Gestion Mapbox optionnelle
    mapbox_style = "open-street-map"
    token = get_secret("MAPBOX_TOKEN")

    if base_map.startswith("Mapbox"):
        if token:
            px.set_mapbox_access_token(token)
            mapbox_style = ("mapbox://styles/mapbox/streets-v12"
                            if "Streets" in base_map else
                            "mapbox://styles/mapbox/satellite-streets-v12")
        else:
            st.info("Aucun `MAPBOX_TOKEN` détecté → fond OpenStreetMap.")
            base_map = "OpenStreetMap"  # fallback propre

    # Heatmap (gratuite, très fluide)
    if base_map == "Heatmap (densité)":
        d = df_view[["latitude","longitude"]].dropna()
        if fast_mode and len(d) > 100_000:
            d = downsample(d, 100_000)

        fig = px.density_mapbox(
            d, lat="latitude", lon="longitude",
            radius=dens_radius, center=centre, zoom=zoom, height=620
        )
        fig.update_layout(
            mapbox_style="open-street-map" if not token else None,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        st.plotly_chart(fig, use_container_width=True,
                        config={"scrollZoom": scroll_zoom, "displaylogo": False})

    # Points bruts (échantillonnés)
    else:
        d = df_view.dropna(subset=["latitude","longitude"])
        if fast_mode:
            d = downsample(d, max_points)

        hover_cols = [c for c in ["date","heure","code_insee"] if c in d.columns]
        fig = px.scatter_mapbox(
            d, lat="latitude", lon="longitude",
            hover_data=hover_cols, opacity=point_alpha,
            height=620, zoom=zoom, center=centre
        )
        fig.update_traces(marker={"size": point_size})
        fig.update_layout(mapbox_style=mapbox_style,
                          margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True,
                        config={"scrollZoom": scroll_zoom, "displaylogo": False})

# === FOLIUM ============================================================
with tabs[1]:
    st.subheader("Carte Folium (clustering) — rapide et pratique")

    try:
        import folium
        from streamlit_folium import st_folium
        from folium.plugins import FastMarkerCluster, MiniMap, Fullscreen, MeasureControl

        # Choix du fond Folium (gratuits)
        fond = st.selectbox(
            "Fond Folium",
            ["Esri WorldImagery (Satellite)", "OpenStreetMap", "CartoDB Positron"],
            index=0
        )

        # Carte Folium
        m = folium.Map(location=[center_lat, center_lon],
                       zoom_start=int(zoom), control_scale=True, tiles=None)

        if "Esri" in fond:
            esri_sat = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
            folium.TileLayer(tiles=esri_sat, attr="Esri WorldImagery").add_to(m)
        elif "CartoDB" in fond:
            folium.TileLayer("CartoDB Positron").add_to(m)
        else:
            folium.TileLayer("OpenStreetMap").add_to(m)

        # Outils confort
        Fullscreen().add_to(m)
        MiniMap(toggle_display=True).add_to(m)
        MeasureControl(primary_length_unit='meters').add_to(m)

        # Clustering ultra-rapide
        d = df_view.dropna(subset=["latitude","longitude"])
        if fast_mode:
            d = downsample(d, max_points)

        coords = d[["latitude","longitude"]].values.tolist()
        FastMarkerCluster(data=coords).add_to(m)

        st_folium(m, height=620, width=None)
    except ModuleNotFoundError:
        st.warning("Installe `streamlit-folium` et `folium` pour activer le clustering rapide : "
                   "`pip install streamlit-folium folium`")

# -------------------------------
# Classements & Export
# -------------------------------
st.markdown("---")
st.subheader("Classements & export")
colL, colR = st.columns(2)

with colL:
    st.markdown("**Top 15 zones (INSEE) par nombre d’accidents**")
    if has_insee and not agg.empty:
        st.dataframe(agg.sort_values("accidents", ascending=False).head(15), use_container_width=True)
    else:
        st.info("Pas de tableau agrégé disponible.")

with colR:
    st.markdown("**Export**")
    st.download_button(
        "💾 Points filtrés (CSV)",
        data=df_f.to_csv(index=False).encode("utf-8"),
        file_name="acc_spatial_points.csv",
        mime="text/csv",
        use_container_width=True
    )
    if has_insee and not agg.empty:
        st.download_button(
            "💾 Agrégat INSEE (CSV)",
            data=agg.to_csv(index=False).encode("utf-8"),
            file_name="acc_spatial_agg_insee.csv",
            mime="text/csv",
            use_container_width=True
        )

with st.expander("🧪 Notes"):
    st.markdown(
        "- **Mode rapide** : échantillonne automatiquement et accélère l’affichage.\n"
        "- **Heatmap Plotly** : très fluide sans clé API.\n"
        "- **Folium** : **FastMarkerCluster** (très rapide), **plein écran**, **mini-carte** et **mesure**.\n"
        "- **Hauts-de-Seine** : rognage par *bounding box* + zoom dédié.\n"
        "- **Mapbox** : utilisé uniquement si `MAPBOX_TOKEN` est présent ; sinon OpenStreetMap."
    )
