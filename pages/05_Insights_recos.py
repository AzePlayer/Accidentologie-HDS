

# pages/05_Insights_recos.py — Synthèse Insights & recommandations

import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from ui_nav import nav_bar  # <- nav bar

st.set_page_config(
    page_title="Accidentologie HDS — Insights & recommandations",
    page_icon="💡",
    layout="wide",
)

# =========================
# Couleurs & helpers
# =========================
PALETTE_SEQ = px.colors.sequential.Blues

def safe_has(df: pd.DataFrame, cols):
    return all(c in df.columns for c in cols)

def correlation_ratio(categories, measurements):
    """
    η² (correlation ratio) pour cat↔num. Robuste aux NaN.
    """
    try:
        cats = pd.Categorical(categories)
        y = pd.to_numeric(measurements, errors="coerce")
        mask = ~cats.isna() & ~pd.isna(y)
        cats = cats[mask]
        y = y[mask]
        if y.size < 3:
            return np.nan
        grand_mean = np.nanmean(y)
        groups = [y[cats.codes == k] for k in range(cats.categories.size)]
        ss_between = np.nansum(
            [len(g) * (np.nanmean(g) - grand_mean) ** 2 for g in groups if len(g) > 0]
        )
        ss_total = np.nansum((y - grand_mean) ** 2)
        return float(ss_between / ss_total) if ss_total > 0 else np.nan
    except Exception:
        return np.nan

def cramers_v(x, y):
    """
    V de Cramér (cat↔cat) avec correction biais (Bergsma 2013).
    On regroupe automatiquement les modalités rares (<0.5%) pour stabilité.
    """
    try:
        xs, ys = x.astype(str), y.astype(str)

        def shrink(s):
            vc = s.value_counts(normalize=True)
            rare = vc[vc < 0.005].index
            return s.where(~s.isin(rare), other="Autre")

        xs, ys = shrink(xs), shrink(ys)
        table = pd.crosstab(xs, ys)
        if table.size == 0:
            return np.nan
        O = table.values
        row_sums = O.sum(axis=1, keepdims=True)
        col_sums = O.sum(axis=0, keepdims=True)
        n = O.sum()
        E = row_sums @ col_sums / max(n, 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            chi2 = np.nansum((O - E) ** 2 / np.where(E == 0, np.nan, E))
        phi2 = chi2 / max(n, 1)
        r, k = O.shape
        phi2corr = max(0, phi2 - (k - 1) * (r - 1) / max(n - 1, 1))
        rcorr = r - (r - 1) ** 2 / max(n - 1, 1)
        kcorr = k - (k - 1) ** 2 / max(n - 1, 1)
        denom = min(kcorr - 1, rcorr - 1)
        return float(np.sqrt(phi2corr / denom)) if denom > 0 else np.nan
    except Exception:
        return np.nan

def make_lum_cat(series):
    s = series.astype(str).str.lower()

    def map_lum(x: str) -> str:
        if "jour" in x:
            return "Plein jour"
        if "crépuscule" in x or "crepuscule" in x or "aube" in x:
            return "Crépuscule / Aube"
        if "nuit" in x and ("éclairage" in x or "eclairage" in x):
            return "Nuit avec éclairage"
        if "nuit" in x:
            return "Nuit sans éclairage"
        return "Non renseigné"

    return s.map(map_lum)

def fmt_int(v):
    if v is None or (isinstance(v, (float, int)) and pd.isna(v)):
        return "—"
    return f"{int(round(v)):,}".replace(",", " ")

def fmt_float(v, digits=2):
    if v is None or (isinstance(v, (float, int)) and pd.isna(v)):
        return "—"
    return f"{float(v):.{digits}f}"

def fmt_pct(v, digits=1):
    if v is None or (isinstance(v, (float, int)) and pd.isna(v)):
        return "—"
    return f"{float(v):.{digits}f} %"

def section_header(title: str, color: str, emoji: str):
    st.markdown(
        f"""
        <div style='margin-top:1.5rem; margin-bottom:0.4rem'>
          <h3 style='margin-bottom:0.2rem; font-weight:600'>
            {emoji} {title}
          </h3>
          <hr style='border-top:2px solid {color}; margin-top:0.2rem; margin-bottom:0.6rem' />
        </div>
        """,
        unsafe_allow_html=True,
    )

def kpi_card(col, label: str, value: str, icon: str = ""):
    html = f"""
    <div style='padding:0.4rem 0.2rem'>
      <div style='font-size:26px; font-weight:600;'>{value}</div>
      <div style='font-size:13px; color:#555;'>{label}</div>
      <div style='font-size:16px; margin-top:0.1rem'>{icon}</div>
    </div>
    """
    col.markdown(html, unsafe_allow_html=True)

# =========================
# Lecture & normalisation
# =========================
@st.cache_data(show_spinner=False)
def load_data(path="acc.csv") -> pd.DataFrame:
    try:
        df = pd.read_csv(path, sep=";", low_memory=False)
    except Exception:
        df = pd.read_csv(path, low_memory=False)
    if df.empty:
        return df

    df["date"] = pd.to_datetime(df.get("date"), errors="coerce", dayfirst=True)

    # heure_num
    if "heure" in df.columns and "heure_num" not in df.columns:
        h = df["heure"].astype(str).str.replace("h", ":", regex=False)
        df["heure_num"] = pd.to_datetime(h, errors="coerce").dt.hour
    else:
        df["heure_num"] = pd.to_numeric(df.get("heure_num"), errors="coerce")

    # departement
    if "code_insee" in df.columns:
        c = df["code_insee"].astype(str)
        dep = np.where(
            c.str.startswith("2A"),
            "2A",
            np.where(c.str.startswith("2B"), "2B", c.str[:2]),
        )
        df["departement"] = dep

    # jour_sem
    try:
        df["jour_sem"] = df["date"].dt.day_name(locale="fr_FR")
    except Exception:
        mapper = {
            0: "lundi",
            1: "mardi",
            2: "mercredi",
            3: "jeudi",
            4: "vendredi",
            5: "samedi",
            6: "dimanche",
        }
        df["jour_sem"] = df["date"].dt.dayofweek.map(mapper)
    ordre = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
    df["jour_sem"] = pd.Categorical(df["jour_sem"], categories=ordre, ordered=True)

    # année / mois
    df["annee"] = df["date"].dt.year
    df["mois"] = df["date"].dt.to_period("M").astype(str)
    df["mois_num"] = df["date"].dt.month

    # luminosité
    if "luminosite" in df.columns:
        df["lum_cat"] = make_lum_cat(df["luminosite"])
    else:
        df["lum_cat"] = "Non renseigné"

    # latitude/longitude depuis geo_point_2d si besoin
    if not safe_has(df, ["latitude", "longitude"]) and "geo_point_2d" in df.columns:
        sp = df["geo_point_2d"].astype(str).str.split(",", n=1, expand=True)
        if sp.shape[1] == 2:
            df["latitude"] = pd.to_numeric(sp[0].str.strip(), errors="coerce")
            df["longitude"] = pd.to_numeric(sp[1].str.strip(), errors="coerce")
    for c in ("latitude", "longitude"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # proxy de gravité (0 à 3)
    sev = pd.Series(np.nan, index=df.index)
    if "grav_usa1" in df.columns:
        g = df["grav_usa1"].astype(str).str.lower()
        sev = np.select(
            [
                g.str.contains("tu"),
                g.str.contains("hospital"),
                g.str.contains("léger") | g.str.contains("leger"),
            ],
            [3, 2, 1],
            default=0,
        )
    else:
        tu = 0
        if "nb_t_ve1" in df.columns:
            tu = np.maximum(
                tu, pd.to_numeric(df["nb_t_ve1"], errors="coerce").fillna(0)
            )
        if "nb_t_ve2" in df.columns:
            tu = np.maximum(
                tu, pd.to_numeric(df["nb_t_ve2"], errors="coerce").fillna(0)
            )
        bh = 0
        if "nb_bh_ve1" in df.columns:
            bh = pd.to_numeric(df["nb_bh_ve1"], errors="coerce").fillna(0)
        if "nb_bh_ve2" in df.columns:
            bh = np.maximum(
                bh, pd.to_numeric(df["nb_bh_ve2"], errors="coerce").fillna(0)
            )
        bl = 0
        if "nb_bnh_ve1" in df.columns:
            bl = pd.to_numeric(df["nb_bnh_ve1"], errors="coerce").fillna(0)
        if "nb_bnh_ve2" in df.columns:
            bl = np.maximum(
                bl, pd.to_numeric(df["nb_bnh_ve2"], errors="coerce").fillna(0)
            )
        sev = np.select([tu > 0, bh > 0, bl > 0], [3, 2, 1], default=0)
    df["sev_score"] = pd.to_numeric(sev, errors="coerce")
    df["sev_grave"] = (df["sev_score"] >= 2).astype(int)  # grave = hospitalisé/tué

    return df

# =========================
# Données & filtres
# =========================
if (
    "df" in st.session_state
    and isinstance(st.session_state.df, pd.DataFrame)
    and not st.session_state.df.empty
):
    df = st.session_state.df.copy()
else:
    df = load_data("acc.csv")

st.title("💡 Synthèse globale & recommandations")
nav_bar(active="Insights & recos")   # <- barre de navigation
st.markdown("---")

if df.empty:
    st.error("Aucune donnée disponible.")
    st.stop()

st.sidebar.header("⚙️ Filtres")
years = sorted(df["annee"].dropna().unique().astype(int)) if "annee" in df else []
an_sel = st.sidebar.multiselect(
    "Années", years, default=years[-1:] if years else []
)
if an_sel:
    df = df[df["annee"].isin(an_sel)]

deps = sorted(df.get("departement", pd.Series(dtype=str)).dropna().unique())
dep_sel = st.sidebar.multiselect(
    "Départements", ["(Tous)"] + deps, default=["(Tous)"]
)
if dep_sel and "(Tous)" not in dep_sel and "departement" in df.columns:
    df = df[df["departement"].isin(dep_sel)]

hmin, hmax = st.sidebar.select_slider(
    "Plage horaire", options=list(range(24)), value=(6, 20)
)
df = df[(df["heure_num"].between(hmin, hmax)) | df["heure_num"].isna()]

nmax = st.sidebar.slider(
    "Taille max échantillon (perf.)", 5000, 120000, 40000, step=5000
)
if len(df) > nmax:
    df = df.sample(nmax, random_state=42)

total = len(df)
pct_graves_global = 100 * df["sev_grave"].mean() if total else 0.0
part_nuit = 100 * (
    df["heure_num"].between(22, 23) | df["heure_num"].between(0, 5)
).mean()

# =========================
# 1. Bloc "Statistiques générales"
# =========================
death_cols = [c for c in df.columns if "nb_t_" in c.lower()]
if death_cols:
    nb_deces = (
        df[death_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum().sum()
    )
else:
    nb_deces = int((df["sev_score"] == 3).sum()) if "sev_score" in df else None

bless_cols = [
    c
    for c in df.columns
    if any(k in c.lower() for k in ["nb_bh", "nb_bnh"])
]
if bless_cols:
    nb_blesses = (
        df[bless_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum().sum()
    )
else:
    nb_blesses = None

pct_mortels = (
    100 * (df["sev_score"] == 3).mean()
    if "sev_score" in df.columns
    else None
)

section_header("Statistiques générales", "#f44336", "📊")

g1, g2, g3, g4 = st.columns(4)

kpi_card(g1, "Total des accidents", fmt_int(total), "🚗")
kpi_card(g2, "Total des décès (approx.)", fmt_int(nb_deces), "☠️")
kpi_card(g3, "Total des blessés (approx.)", fmt_int(nb_blesses), "🩺")
kpi_card(g4, "Accidents mortels (part des accidents)", fmt_pct(pct_mortels), "⚠️")

# =========================
# 2. Bloc "Analyse démographique"
# =========================

# Femmes / hommes tués : on utilise sev_score == 3 + colonne sexe
femmes_tuees = hommes_tues = None

# 1) chercher un nom exact
sex_col = None
for c in ["sexe_usa1", "sexe", "genre", "sexe_usager_1", "sexe_usager1"]:
    if c in df.columns:
        sex_col = c
        break

# 2) sinon, fallback sur n'importe quelle colonne contenant "sexe" ou "genre"
if sex_col is None:
    for c in df.columns:
        cl = c.lower()
        if "sexe" in cl or "genre" in cl:
            sex_col = c
            break

if sex_col is not None and "sev_score" in df.columns:
    s = df[sex_col].astype(str).str.strip().str.lower()
    sev = df["sev_score"]
    dead_mask = sev == 3  # 3 = tué dans notre proxy

    # si tu as Masculin/Féminin ou 1/2, ça marche pour les deux
    is_f = s.str.startswith(("f", "fé", "fe", "2"))
    is_h = s.str.startswith(("m", "mas", "ho", "1"))

    femmes_tuees = int((dead_mask & is_f).sum())
    hommes_tues = int((dead_mask & is_h).sum())

# Taux d'accidents piétons / conducteurs (approx)
def rate_for_value(df, substrings):
    substrings = [s.lower() for s in substrings]
    for c in df.columns:
        if df[c].dtype == "O" or pd.api.types.is_categorical_dtype(df[c]):
            s = df[c].astype(str).str.lower()
            if any(s.str.contains(sub).any() for sub in substrings):
                mask = False
                for sub in substrings:
                    mask = mask | s.str.contains(sub)
                return 100 * mask.mean()
    return None

taux_pieton = rate_for_value(df, ["piéton", "pieton"])
taux_cond   = rate_for_value(df, ["conducteur", "driver"])

# Âge moyen usagers
age_moy = None
for c in ["age_usa1", "age"]:
    if c in df.columns:
        age_moy = pd.to_numeric(df[c], errors="coerce").mean()
        break


occ_moy = None


section_header("Analyse démographique", "#2196f3", "👥")

d1, d2, d3 = st.columns(3)

with d1:
    kpi_card(d1, "Femmes tuées (usager 1)", fmt_int(femmes_tuees), "♀️")
    kpi_card(d1, "Taux d'accidents piétons", fmt_pct(taux_pieton), "🚶")

with d2:
    kpi_card(d2, "Hommes tués (usager 1)", fmt_int(hommes_tues), "♂️")
    kpi_card(d2, "Taux d'accidents conducteurs", fmt_pct(taux_cond), "🚗")

with d3:
    kpi_card(d3, "Âge moyen des usagers impliqués", fmt_float(age_moy, 1), "📍")


# =========================
# 3. Bloc "Répartition temporelle"
# =========================
mois_plus_risque = jour_plus_risque = None
acc_plein_jour = acc_ete = acc_hiver = nuit_sans_ecl = None

if "date" in df.columns:
    mois_counts = df["mois_num"].value_counts(dropna=True)
    if not mois_counts.empty:
        mois_plus_risque = int(mois_counts.idxmax())

    jour_mois = df["date"].dt.day
    jour_counts = jour_mois.value_counts(dropna=True)
    if not jour_counts.empty:
        jour_plus_risque = int(jour_counts.idxmax())

if "lum_cat" in df.columns:
    acc_plein_jour = int((df["lum_cat"] == "Plein jour").sum())
    nuit_sans_ecl = int((df["lum_cat"] == "Nuit sans éclairage").sum())

if "mois_num" in df.columns:
    ete_mask   = df["mois_num"].isin([6, 7, 8])
    hiver_mask = df["mois_num"].isin([12, 1, 2])
    acc_ete   = int(ete_mask.sum())
    acc_hiver = int(hiver_mask.sum())

section_header("Répartition temporelle", "#ff9800", "⏱️")

t1, t2, t3 = st.columns(3)

with t1:
    kpi_card(t1, "Mois le plus risqué (numéro)", fmt_int(mois_plus_risque), "📅")
    kpi_card(t1, "Accidents en plein jour", fmt_int(acc_plein_jour), "☀️")

with t2:
    kpi_card(t2, "Journée du mois la plus accidentogène", fmt_int(jour_plus_risque), "📍")
    kpi_card(t2, "Accidents en été (juin–août)", fmt_int(acc_ete), "🌞")

with t3:
    kpi_card(t3, "Nuit sans éclairage public", fmt_int(nuit_sans_ecl), "🌙")
    kpi_card(t3, "Accidents en hiver (déc–fév)", fmt_int(acc_hiver), "❄️")

# =========================
# 4. Bloc "Conditions environnementales"
# =========================
acc_pluie = collision_pluie = None
if "cond_atmos" in df.columns:
    atm = df["cond_atmos"].astype(str).str.lower()
    pluie_mask = atm.str.contains("pluie") | atm.str.contains("rain")
    acc_pluie = int(pluie_mask.sum())
    if "TYPE_COLLI" in df.columns and pluie_mask.any():
        mode_coll = (
            df.loc[pluie_mask, "TYPE_COLLI"].astype(str).mode(dropna=True)
        )
        if not mode_coll.empty:
            collision_pluie = mode_coll.iloc[0]

section_header("Conditions environnementales", "#4caf50", "🌦️")

e1, e2 = st.columns(2)
kpi_card(e1, "Accidents sous la pluie (approx.)", fmt_int(acc_pluie), "🌧️")
kpi_card(
    e2,
    "Collision la plus fréquente par temps de pluie",
    collision_pluie or "—",
    "💥",
)

st.markdown("---")

# =====================================================
# 5. Analyse avancée : associations, segments, profils
# =====================================================

# --- Top associations avec la gravité ---
st.subheader("Top associations avec la gravité (KPI explicatifs)")

# Candidats
num_cands, cat_cands = [], []
for c in df.columns:
    if pd.api.types.is_numeric_dtype(df[c]) and c not in ["sev_score", "sev_grave"]:
        if df[c].nunique(dropna=True) >= 5:
            num_cands.append(c)
    elif pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_categorical_dtype(df[c]):
        n_mod = df[c].nunique(dropna=True)
        if 3 <= n_mod <= 60:
            cat_cands.append(c)

assoc = []

# num ↔ sev_score : Spearman
for c in num_cands:
    s = pd.to_numeric(df[c], errors="coerce")
    mask = ~s.isna() & ~df["sev_score"].isna()
    if mask.sum() >= 100:
        rho = pd.Series(s[mask]).corr(df.loc[mask, "sev_score"], method="spearman")
        if pd.notna(rho):
            assoc.append(("Spearman num↔gravité", c, abs(float(rho))))

# cat ↔ sev_score : η²
for c in cat_cands:
    eta = correlation_ratio(df[c], df["sev_score"])
    if pd.notna(eta):
        assoc.append(("η² cat↔gravité", c, float(eta)))

# cat ↔ gravité binaire : Cramér V
for c in cat_cands:
    v = cramers_v(df[c], df["sev_grave"])
    if pd.notna(v):
        assoc.append(("V de Cramér cat↔grave", c, float(v)))

assoc_df = pd.DataFrame(assoc, columns=["Mesure", "Variable", "Force"])
assoc_df = assoc_df.sort_values("Force", ascending=False).head(8)

if assoc_df.empty:
    st.info("Pas assez de données pour calculer des associations robustes.")
else:
    assoc_show = assoc_df.copy()
    assoc_show["Force"] = assoc_show["Force"].round(3)
    st.dataframe(assoc_show, use_container_width=True)
    top = assoc_df.iloc[0]
    st.caption(
        f"Variable la plus associée à la gravité : **{top['Variable']}** "
        f"({top['Mesure']} = {top['Force']:.2f})."
    )

st.divider()

# --- Segments à risque (uplift vs moyenne) ---
st.subheader("Segments à risque (taux de cas graves > moyenne)")

prefered_cats = [
    "lum_cat",
    "TYPE_COLLI",
    "type_acci",
    "cond_atmos",
    "cat_ve1",
    "lieu",
    "commune",
    "code_insee",
]
options_base = [c for c in prefered_cats if c in df.columns]
cand_seg = [c for c in cat_cands if c not in ["sev_grave", "sev_score"]]

seen = set()
options = []
for c in options_base + cand_seg:
    if c not in seen:
        seen.add(c)
        options.append(c)

if not options:
    st.info("Aucune variable catégorielle exploitable pour construire des segments à risque.")
    var_seg = None
else:
    default_index = 0
    if "lum_cat" in options:
        default_index = options.index("lum_cat")
    var_seg = st.selectbox(
        "Variable pour segmenter le risque de gravité :", options, index=default_index
    )

if var_seg is not None:
    base = df["sev_grave"].mean()
    tab = (
        df.groupby(var_seg)["sev_grave"]
        .agg(n="count", grave="mean")
        .reset_index()
    )
    tab["uplift (points)"] = 100 * (tab["grave"] - base)
    tab["Taux graves (%)"] = 100 * tab["grave"]
    tab = tab[tab["n"] >= 50].sort_values("uplift (points)", ascending=False).head(10)

    st.caption(
        f"Taux moyen de cas graves (base) : **{100*base:.1f}%** — "
        f"segments affichés : ≥ 50 observations."
    )

    if tab.empty:
        st.info("Pas assez d’observations pour afficher des segments robustes.")
    else:
        fig_seg = px.bar(
            tab,
            x="uplift (points)",
            y=var_seg,
            orientation="h",
            color="Taux graves (%)",
            color_continuous_scale="Reds",
            labels={"uplift (points)": "Uplift (points de % vs moyenne)"},
            height=420,
        )
        fig_seg.update_layout(margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig_seg, use_container_width=True)

        with st.expander("Détail des segments à risque"):
            show_tab = tab[[var_seg, "n", "Taux graves (%)", "uplift (points)"]].copy()
            show_tab["Taux graves (%)"] = show_tab["Taux graves (%)"].round(1)
            show_tab["uplift (points)"] = show_tab["uplift (points)"].round(1)
            st.dataframe(show_tab, use_container_width=True)

st.divider()

# --- Profils clés à surveiller ---
st.subheader("Profils clés à surveiller (usagers / véhicules)")

profiles_rows = []
profiles_spec = [
    ("age_usa1", "Tranche d’âge usager 1", True),
    ("cat_ve1", "Catégorie de véhicule 1", False),
    ("TYPE_COLLI", "Type de collision", False),
    ("cond_atmos", "Conditions atmosphériques", False),
]

for var, label, is_age in profiles_spec:
    if var not in df.columns:
        continue
    tmp = df[[var, "sev_grave"]].copy()

    if is_age:
        age = pd.to_numeric(tmp[var], errors="coerce")
        tmp["modalite"] = pd.cut(
            age,
            bins=[0, 18, 25, 35, 45, 55, 65, 80, 120],
            labels=[
                "≤18",
                "19–25",
                "26–35",
                "36–45",
                "46–55",
                "56–65",
                "66–80",
                ">80",
            ],
            include_lowest=True,
        )
    else:
        tmp["modalite"] = tmp[var].astype(str)

    grp = (
        tmp.groupby("modalite")["sev_grave"]
        .agg(n="count", taux="mean")
        .reset_index()
        .dropna()
    )
    grp = grp[grp["n"] >= 50]
    if grp.empty:
        continue

    grp = grp.sort_values("taux", ascending=False).head(3)
    for _, row in grp.iterrows():
        profiles_rows.append(
            {
                "Variable": label,
                "Modalité": str(row["modalite"]),
                "Taux cas graves (%)": round(100 * row["taux"], 1),
                "N": int(row["n"]),
            }
        )

if profiles_rows:
    profiles_df = pd.DataFrame(profiles_rows)
    profiles_df = profiles_df.sort_values(
        ["Taux cas graves (%)", "N"], ascending=[False, False]
    )
    st.dataframe(profiles_df, use_container_width=True)
    st.caption(
        "Profils triés par taux de cas graves, avec un seuil de 50 observations minimum par segment."
    )
else:
    st.info(
        "Aucun profil robuste identifié avec les filtres actuels (seuil = 50 observations par segment)."
    )

st.divider()

# =========================
# 6. Narratif & recommandations
# =========================
st.subheader("Interprétation synthétique & recommandations")

findings = []

# 1) Résumé des associations
if not assoc_df.empty:
    top_assoc = assoc_df.head(3)
    desc = "; ".join(
        f"{row['Variable']} ({row['Mesure']} = {row['Force']:.2f})"
        for _, row in top_assoc.iterrows()
    )
    findings.append(
        f"Les variables les plus associées à la gravité sont : {desc}."
    )

# 2) Luminosité
if "lum_cat" in df.columns:
    t = (
        df.groupby("lum_cat")["sev_grave"]
        .agg(n="count", taux="mean")
        .reset_index()
    )
    t = t[t["n"] >= 50]
    if not t.empty:
        worst = t.sort_values("taux", ascending=False).iloc[0]
        findings.append(
            f"Les accidents survenant en **{worst['lum_cat']}** présentent un taux de cas graves "
            f"élevé (**{100*worst['taux']:.1f}%**, n={int(worst['n'])})."
        )

# 3) Horaires
if "heure_num" in df.columns:
    h = (
        df.groupby("heure_num")["sev_grave"]
        .mean()
        .sort_values(ascending=False)
        .head(3)
    )
    if not h.empty:
        findings.append(
            "Les créneaux horaires les plus sévères sont autour de : "
            + ", ".join(f"{int(k)}h ({100*v:.1f}% de cas graves)" for k, v in h.items())
            + "."
        )

# 4) Types de collision & météo
if "TYPE_COLLI" in df.columns:
    c = (
        df.groupby("TYPE_COLLI")["sev_grave"]
        .agg(n="count", taux="mean")
        .reset_index()
    )
    c = c[c["n"] >= 50].sort_values("taux", ascending=False).head(3)
    if not c.empty:
        findings.append(
            "Certains **types de collision** (ex. "
            + ", ".join(
                f"{row['TYPE_COLLI']} ({100*row['taux']:.1f}%, n={int(row['n'])})"
                for _, row in c.iterrows()
            )
            + ") concentrent une part importante des cas graves."
        )

if "cond_atmos" in df.columns:
    m = (
        df.groupby("cond_atmos")["sev_grave"]
        .agg(n="count", taux="mean")
        .reset_index()
    )
    m = m[m["n"] >= 50].sort_values("taux", ascending=False).head(3)
    if not m.empty:
        findings.append(
            "Certaines **conditions météorologiques** ("
            + ", ".join(
                f"{row['cond_atmos']} ({100*row['taux']:.1f}%, n={int(row['n'])})"
                for _, row in m.iterrows()
            )
            + ") sont particulièrement associées à des cas graves."
        )

reco = []

if "lum_cat" in df.columns:
    reco += [
        "Renforcer l’éclairage et la signalisation sur les axes où les accidents de nuit sont les plus graves.",
        "Mettre en place des campagnes ciblées sur la vitesse et les distances de sécurité de nuit / par faible visibilité.",
    ]
if "TYPE_COLLI" in df.columns:
    reco += [
        "Prioriser des aménagements (ralentisseurs, giratoires, séparateurs, feux intelligents) sur les carrefours et segments où les collisions les plus sévères sont concentrées."
    ]
if "cond_atmos" in df.columns:
    reco += [
        "Déployer des messages dynamiques et des limitations temporaires en cas de pluie / neige / verglas sur les axes identifiés comme critiques.",
    ]
if "age_usa1" in df.columns:
    reco += [
        "Adapter les actions de prévention à certains profils d’âge (jeunes conducteurs, seniors) identifiés comme plus exposés dans les cas graves.",
    ]

st.markdown("### Principaux constats")
if findings:
    st.markdown("\n".join(f"- {f}" for f in findings))
else:
    st.write("_Aucun constat robuste avec les filtres actuels._")

st.markdown("### Recommandations opérationnelles")
if reco:
    st.markdown("\n".join(f"- {r}" for r in reco))
else:
    st.write("_Aucune recommandation spécifique générée._")

# =========================
# 7. Export mini-rapport
# =========================
report_btn = st.button("📄 Exporter un mini-rapport (Markdown)")
if report_btn:
    md = io.StringIO()
    md.write(
        f"# Rapport — Insights (filtres: années={an_sel or 'toutes'}, "
        f"dep={dep_sel or 'tous'}, heures={hmin}-{hmax})\n\n"
    )
    md.write(
        f"Total observations: **{total}** — Taux cas graves: **{pct_graves_global:.1f}%**\n\n"
    )
    md.write("## Constats\n")
    if findings:
        for f in findings:
            md.write(f"- {f}\n")
    else:
        md.write("- Aucun constat robuste avec ces filtres.\n")
    md.write("\n## Recommandations\n")
    if reco:
        for r in reco:
            md.write(f"- {r}\n")
    else:
        md.write("- Aucune recommandation spécifique.\n")

    st.download_button(
        "⬇️ Télécharger le rapport",
        data=md.getvalue().encode("utf-8"),
        file_name="insights_recommandations.md",
        mime="text/markdown",
        use_container_width=True,
    )
