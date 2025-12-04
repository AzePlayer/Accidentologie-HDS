# # projet.py — Présentation des données (dataset overview) — Fiches descriptives
# import os, io, csv
# import numpy as np
# import pandas as pd
# import streamlit as st

# # ---------------------------
# # Config & titre
# # ---------------------------
# st.set_page_config(page_title="📦 Projet — Présentation des données", page_icon="📦", layout="wide")
# st.title("📦 Projet — Présentation des données")
# st.caption("Aperçu global du fichier `acc.csv` : dictionnaire, % manquants et fiches descriptives par variable (sans graphiques).")

# # ---------------------------
# # Lecture CSV robuste (réutilisable)
# # ---------------------------
# def _read_csv_smart(buffer_or_path, default_sep=";", encodings=("utf-8", "latin-1")):
#     """Lecture robuste: détecte séparateur (comma/semicolon) & essaye plusieurs encodages."""
#     def _read_bytes(raw: bytes, encs):
#         for enc in encodings:
#             try:
#                 sample = raw[:10000].decode(enc, errors="ignore")
#                 dialect = csv.Sniffer().sniff(sample, delimiters=",;")
#                 sep = dialect.delimiter if dialect.delimiter in [",", ";"] else default_sep
#                 return pd.read_csv(io.BytesIO(raw), sep=sep, encoding=enc, low_memory=False)
#             except Exception:
#                 continue
#         return pd.read_csv(io.BytesIO(raw), sep=default_sep, low_memory=False)

#     if isinstance(buffer_or_path, (str, os.PathLike)):
#         if not os.path.exists(buffer_or_path):
#             return pd.DataFrame()
#         with open(buffer_or_path, "rb") as f:
#             raw = f.read()
#         return _read_bytes(raw, encodings)
#     else:
#         raw = buffer_or_path.read()
#         return _read_bytes(raw, encodings)

# @st.cache_data(show_spinner=False)
# def load_data(default_path="acc.csv") -> pd.DataFrame:
#     df = _read_csv_smart(default_path)
#     if df.empty:
#         return df

#     # Normalisations minimales utiles
#     if "date" in df.columns:
#         df["date"] = pd.to_datetime(df["date"], errors="coerce")
#     if "heure" in df.columns and "heure_num" not in df.columns:
#         h = df["heure"].astype(str).str.replace("h", ":", regex=False)
#         df["heure_num"] = pd.to_datetime(h, errors="coerce").dt.hour
#     for c in ("latitude", "longitude"):
#         if c in df.columns:
#             df[c] = pd.to_numeric(df[c], errors="coerce")
#     return df

# # ---------------------------
# # Sidebar : import optionnel
# # ---------------------------
# st.sidebar.header("⚙️ Données")
# upl = st.sidebar.file_uploader("Importer un CSV (optionnel)", type=["csv"])
# df = _read_csv_smart(upl) if upl is not None else load_data("acc.csv")

# if df.empty:
#     st.error("Aucune donnée chargée. Place `acc.csv` à la racine du projet ou importe un CSV via la barre latérale.")
#     st.stop()

# st.success(f"✅ Données chargées : **{len(df):,}** lignes × **{df.shape[1]}** colonnes".replace(",", " "))

# # =====================================================================
# # Helpers de typage & aperçus
# # =====================================================================
# def infer_kind(s: pd.Series):
#     """Retourne (Type, Sous-type) lisibles."""
#     if pd.api.types.is_numeric_dtype(s):
#         return "Quantitative", "numérique"
#     if pd.api.types.is_datetime64_any_dtype(s):
#         return "Quantitative", "date/temps"
#     if pd.api.types.is_bool_dtype(s):
#         return "Qualitative", "booléenne"
#     nun = s.dropna().nunique()
#     if nun <= 25 or pd.api.types.is_categorical_dtype(s) or pd.api.types.is_object_dtype(s):
#         return "Qualitative", "catégorielle"
#     return "Qualitative", "texte libre"

# def preview_values(s: pd.Series, max_items: int = 3) -> str:
#     """Petit aperçu : num -> stats courtes ; cat/texte -> top modalités."""
#     s_nonan = s.dropna()
#     if s_nonan.empty:
#         return "—"
#     if pd.api.types.is_numeric_dtype(s):
#         q = np.nanpercentile(s_nonan.astype(float), [5, 50, 95]) if s_nonan.size >= 3 else [s_nonan.mean()]*3
#         return f"min={s_nonan.min():.2f} | med={q[1]:.2f} | max={s_nonan.max():.2f}"
#     vc = s_nonan.astype(str).value_counts().head(max_items)
#     parts = [f"{k} ({v})" for k, v in vc.items()]
#     return " | ".join(parts)

# # ============================================================
# # 1) Dictionnaire des variables (compact)
# # ============================================================
# st.subheader("📚 Dictionnaire des variables")

# c1, c2, c3 = st.columns([1, 1, 1.2])
# with c1:
#     show_preview = st.toggle("Aperçu concis", value=True, help="Top modalités ou stats courtes.")
# with c2:
#     sort_by = st.selectbox("Trier par", ["Nom", "Type", "Nb modalités", "% manquants"], index=0)
# with c3:
#     search = st.text_input("🔎 Filtrer (contient…)", "")

# rows = []
# for col in df.columns:
#     s = df[col]
#     typ, sub = infer_kind(s)
#     nmods = int(s.dropna().nunique())
#     miss_pct = round(100 * s.isna().mean(), 2)
#     rows.append({
#         "Variable": col,
#         "Type": typ,
#         "Sous-type": sub,
#         "Nb modalités (ou valeurs uniques)": nmods,
#         "% manquants": miss_pct,
#         "Aperçu": preview_values(s, 3 if show_preview else 5)
#     })
# dict_df = pd.DataFrame(rows)

# if search:
#     dict_df = dict_df[dict_df["Variable"].str.contains(search, case=False, na=False)]

# if sort_by == "Nom":
#     dict_df = dict_df.sort_values("Variable")
# elif sort_by == "Type":
#     dict_df = dict_df.sort_values(["Type", "Variable"])
# elif sort_by == "Nb modalités":
#     dict_df = dict_df.sort_values("Nb modalités (ou valeurs uniques)", ascending=False)
# else:
#     dict_df = dict_df.sort_values("% manquants", ascending=False)

# st.dataframe(dict_df, use_container_width=True, hide_index=True)
# st.caption("ℹ️ **Aperçu** : numérique → min/med/max ; qualitatif → top modalités (effectifs).")

# st.divider()

# # ============================================================
# # 2) Fiches descriptives (zone unique + sous-dépliants)
# # ============================================================
# st.subheader("🗂️ Fiches descriptives par variable")

# # Sélection de variables à documenter
# col_sel1, col_sel2 = st.columns([1.6, 1])
# with col_sel1:
#     selected_vars = st.multiselect(
#         "Choisir les variables à décrire",
#         options=sorted(df.columns),
#         default=[]
#     )
# with col_sel2:
#     sample_n = st.number_input("Taille de l’échantillon (table d’exemples)", 3, 50, 8, step=1)

# main_exp = st.expander("Déplier les fiches sélectionnées", expanded=bool(selected_vars))

# def render_variable_card(name: str, s: pd.Series):
#     typ, sub = infer_kind(s)
#     nun = int(s.dropna().nunique())
#     miss_pct = 100 * s.isna().mean()
#     card = st.expander(f"ℹ️ {name} – détails", expanded=False)

#     with card:
#         # Métadonnées principales
#         with st.container(border=True):
#             st.markdown(
#                 f"**Variable :** `{name}`  \n"
#                 f"**Type :** {typ}  •  **Sous-type :** {sub}  \n"
#                 f"**Valeurs uniques :** {nun:,}  \n"
#                 f"**% manquants :** {miss_pct:.2f}%"
#             )

#         # Résumé adapté au type
#         if pd.api.types.is_numeric_dtype(s):
#             s_num = pd.to_numeric(s, errors="coerce")
#             desc = s_num.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).to_frame("valeur")
#             st.markdown("**Résumé numérique**")
#             st.dataframe(desc, use_container_width=True)

#         elif pd.api.types.is_datetime64_any_dtype(s):
#             s_dt = pd.to_datetime(s, errors="coerce").dropna()
#             if s_dt.empty:
#                 st.info("Dates non exploitables.")
#             else:
#                 st.markdown("**Résumé temporel**")
#                 info = pd.DataFrame({
#                     "min": [s_dt.min()],
#                     "max": [s_dt.max()],
#                     "nb mois couverts": [s_dt.dt.to_period("M").nunique()]
#                 })
#                 st.dataframe(info, use_container_width=True)

#         else:
#             # Qualitatif / texte : top modalités
#             st.markdown("**Top modalités**")
#             top_k = s.astype(str).value_counts(dropna=False).head(20)
#             top_tbl = top_k.reset_index()
#             top_tbl.columns = [name, "effectif"]
#             top_tbl["%"] = (100 * top_tbl["effectif"] / max(len(s), 1)).round(2)
#             st.dataframe(top_tbl, use_container_width=True)

#         # Échantillon de valeurs (texte propre)
#         st.markdown("**Échantillon de valeurs (nettoyé)**")
#         sample = (
#             df[[name]]
#             .astype(str)
#             .replace({"nan": "—", "NaT": "—"})
#             .head(sample_n)
#         )
#         st.dataframe(sample, use_container_width=True, hide_index=True)

# # Rendu des cartes choisies
# with main_exp:
#     if not selected_vars:
#         st.info("Sélectionne une ou plusieurs variables ci-dessus pour afficher leurs fiches.")
#     else:
#         for v in selected_vars:
#             render_variable_card(v, df[v])

# st.divider()

# # ============================================================
# # 3) Export léger
# # ============================================================
# col_a, col_b = st.columns(2)
# col_a.download_button(
#     "💾 Télécharger le dictionnaire (CSV)",
#     data=dict_df.to_csv(index=False).encode("utf-8"),
#     file_name="dictionnaire_variables.csv",
#     mime="text/csv",
#     use_container_width=True
# )
# col_b.download_button(
#     "💾 Télécharger un échantillon (500 lignes)",
#     data=df.head(500).to_csv(index=False).encode("utf-8"),
#     file_name="echantillon_500.csv",
#     mime="text/csv",
#     use_container_width=True
# )


# projet.py — Présentation des données (labels FR affichés uniquement ici)
import os, io, csv
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------
# Config & titre
# ---------------------------
st.set_page_config(page_title="📦 Projet — Présentation des données", page_icon="📦", layout="wide")
st.title("📦 Projet — Présentation des données")
st.caption("Aperçu du fichier `acc.csv` : dictionnaire des variables (FR/CSV), types, valeurs manquantes et exploration d’une variable. "
           "Les autres pages conservent les noms techniques d’origine.")

# ---------------------------
# Mapping colonnes FR (spécifique à cette page)
# ---------------------------
COLUMNS_FR = {
  "TYPE_COLLI": "Type de collision",
  "adresse": "Adresse",
  "age_usa1": "Âge usager 1",
  "age_usa2": "Âge usager 2",
  "age_usa3": "Âge usager 3",
  "cat_route1": "Catégorie route 1",
  "cat_route2": "Catégorie route 2",
  "cat_ve1": "Catégorie véhicule 1",
  "cat_ve2": "Catégorie véhicule 2",
  "code_insee": "Code INSEE",
  "commune": "Commune",
  "cond_atmos": "Conditions atmosphériques",
  "date": "Date",
  "geo_point_2d": "Géopoint (lat, lon)",
  "grav_usa1": "Gravité usager 1",
  "grav_usa2": "Gravité usager 2",
  "grav_usa3": "Gravité usager 3",
  "heure": "Heure",
  "heure_num": "Heure (numérique)",
  "id_pv": "Identifiant PV",
  "lieu": "Lieu",
  "luminosite": "Luminosité",
  "man_ve1": "Manœuvre véhicule 1",
  "man_ve2": "Manœuvre véhicule 2",
  "nb_bh_ve1": "Blessés hospitalisés (véh. 1)",
  "nb_bh_ve2": "Blessés hospitalisés (véh. 2)",
  "nb_bnh_ve1": "Blessés non hospitalisés (véh. 1)",
  "nb_bnh_ve2": "Blessés non hospitalisés (véh. 2)",
  "nb_pie": "Nombre de piétons",
  "nb_t_ve1": "Nombre de tués (véh. 1)",
  "nb_t_ve2": "Nombre de tués (véh. 2)",
  "nb_usager": "Nombre d'usagers",
  "nb_veh": "Nombre de véhicules",
  "nom_route1": "Nom route 1",
  "nom_route2": "Nom route 2",
  "rev_route1": "Référence route 1",
  "rev_route2": "Référence route 2",
  "route_ve1": "Route véhicule 1",
  "route_ve2": "Route véhicule 2",
  "sens_ve1": "Sens de circulation (véh. 1)",
  "sens_ve2": "Sens de circulation (véh. 2)",
  "sexe_usa1": "Sexe usager 1",
  "sexe_usa2": "Sexe usager 2",
  "sexe_usa3": "Sexe usager 3",
  "type_acci": "Type d'accident",
  "usager1": "Catégorie usager 1",
  "usager2": "Catégorie usager 2",
  "usager3": "Catégorie usager 3",
  "veh_usa1": "Type de véhicule (usager 1)",
  "veh_usa2": "Type de véhicule (usager 2)",
  "veh_usa3": "Type de véhicule (usager 3)",
  "victime": "Victime"
}

def tech2fr_name(col: str) -> str:
    return COLUMNS_FR.get(col, col)

def fr_choices_from_df(df: pd.DataFrame) -> list[str]:
    return [tech2fr_name(c) for c in df.columns]

def fr2tech_lookup(label_fr: str, df: pd.DataFrame) -> str:
    for c in df.columns:
        if tech2fr_name(c) == label_fr:
            return c
    return label_fr

# ---------------------------
# Lecture CSV robuste
# ---------------------------
def _read_csv_smart(buffer_or_path, default_sep=";", encodings=("utf-8", "latin-1")):
    """Détection ; / , et multiple encodages."""
    def _read_bytes(raw: bytes):
        for enc in encodings:
            try:
                sample = raw[:10000].decode(enc, errors="ignore")
                dialect = csv.Sniffer().sniff(sample, delimiters=",;")
                sep = dialect.delimiter if dialect.delimiter in [",", ";"] else default_sep
                return pd.read_csv(io.BytesIO(raw), sep=sep, encoding=enc, low_memory=False)
            except Exception:
                continue
        return pd.read_csv(io.BytesIO(raw), sep=default_sep, low_memory=False)

    if isinstance(buffer_or_path, (str, os.PathLike)):
        if not os.path.exists(buffer_or_path):
            return pd.DataFrame()
        with open(buffer_or_path, "rb") as f:
            raw = f.read()
        return _read_bytes(raw)
    else:
        raw = buffer_or_path.read()
        return _read_bytes(raw)

@st.cache_data(show_spinner=False)
def load_data(default_path="acc.csv") -> pd.DataFrame:
    df = _read_csv_smart(default_path)
    if df.empty:
        return df
    # normalisations minimales
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
    if "heure" in df.columns and "heure_num" not in df.columns:
        h = df["heure"].astype(str).str.replace("h", ":", regex=False)
        df["heure_num"] = pd.to_datetime(h, errors="coerce").dt.hour
    for c in ("latitude", "longitude"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# ---------------------------
# Sidebar : import optionnel
# ---------------------------
st.sidebar.header("⚙️ Données")
upl = st.sidebar.file_uploader("Importer un CSV (optionnel)", type=["csv"])
df = _read_csv_smart(upl) if upl is not None else load_data("acc.csv")

if df.empty:
    st.error("Aucune donnée chargée. Place `acc.csv` à la racine du projet ou importe un CSV via la barre latérale.")
    st.stop()

st.success(f"✅ Données chargées : **{len(df):,}** lignes × **{df.shape[1]}** colonnes".replace(",", " "))

# ============================================================
# 1) Dictionnaire des variables (FR + nom d’origine)
# ============================================================
st.subheader("📚 Dictionnaire des variables")

col_opt1, col_opt2, col_opt3 = st.columns([1, 1, 1.2])
with col_opt1:
    show_preview = st.toggle("Aperçu concis (recommandé)", value=True,
                             help="Top 1–3 modalités ou stats courtes.")
with col_opt2:
    sort_by = st.selectbox("Trier par :", ["Ordre CSV", "Nom (FR)", "Type", "Nb modalités", "% manquants"], index=0)
with col_opt3:
    search = st.text_input("🔎 Filtrer (FR/CSV contient…)", "")

def _infer_kind(s: pd.Series):
    if pd.api.types.is_numeric_dtype(s): return "Quantitative", "numérique"
    if pd.api.types.is_bool_dtype(s):    return "Qualitative", "booléenne"
    if pd.api.types.is_datetime64_any_dtype(s): return "Quantitative", "date/temps"
    nun = s.dropna().nunique()
    if nun <= 25 or pd.api.types.is_categorical_dtype(s) or pd.api.types.is_object_dtype(s):
        return "Qualitative", "catégorielle"
    return "Qualitative", "texte libre"

def _preview(s: pd.Series, concise: bool = True) -> str:
    s_nonan = s.dropna()
    if s_nonan.empty: return "—"
    if pd.api.types.is_numeric_dtype(s):
        if s_nonan.size >= 3:
            if concise:
                return f"min≈{s_nonan.min():.2f} | med≈{np.median(s_nonan):.2f} | max≈{s_nonan.max():.2f}"
            q5, q50, q95 = np.nanpercentile(s_nonan.astype(float), [5, 50, 95])
            return f"q5≈{q5:.2f} | q50≈{q50:.2f} | q95≈{q95:.2f}"
        return f"moy≈{s_nonan.astype(float).mean():.2f}"
    vc = s_nonan.astype(str).value_counts().head(3 if concise else 5)
    return " | ".join([f"{k} ({v})" for k, v in vc.items()])

original_cols = list(df.columns)
rows = []
for i, col in enumerate(original_cols, start=1):
    s = df[col]
    typ, sub = _infer_kind(s)
    nmods = int(s.dropna().nunique())
    miss_pct = 100 * s.isna().mean()
    rows.append({
        "Ordre CSV": i,
        "Nom d’origine (CSV)": col,
        "Nom (FR)": tech2fr_name(col),
        "Type": typ,
        "Sous-type": sub,
        "Nb modalités (ou valeurs uniques)": nmods,
        "% manquants": round(miss_pct, 2),
        "Aperçu": _preview(s, concise=show_preview)
    })

dict_df = pd.DataFrame(rows)

if search:
    m = (
        dict_df["Nom (FR)"].str.contains(search, case=False, na=False) |
        dict_df["Nom d’origine (CSV)"].str.contains(search, case=False, na=False)
    )
    dict_df = dict_df[m]

if sort_by == "Ordre CSV":
    dict_df = dict_df.sort_values("Ordre CSV")
elif sort_by == "Nom (FR)":
    dict_df = dict_df.sort_values(["Nom (FR)", "Nom d’origine (CSV)"])
elif sort_by == "Type":
    dict_df = dict_df.sort_values(["Type", "Nom (FR)"])
elif sort_by == "Nb modalités":
    dict_df = dict_df.sort_values("Nb modalités (ou valeurs uniques)", ascending=False)
else:
    dict_df = dict_df.sort_values("% manquants", ascending=False)

st.dataframe(
    dict_df[[
        "Ordre CSV",
        "Nom d’origine (CSV)",
        "Nom (FR)",
        "Type",
        "Sous-type",
        "Nb modalités (ou valeurs uniques)",
        "% manquants",
        "Aperçu"
    ]],
    use_container_width=True,
    hide_index=True
)

non_map = [c for c in original_cols if tech2fr_name(c) == c]
if non_map:
    st.caption(f"⚠️ Colonnes sans libellé FR (affichées telles quelles) : {', '.join(non_map[:8])}{'…' if len(non_map)>8 else ''}")

st.divider()

# ============================================================
# 2) Exploration détaillée (une variable à la fois, sans graphiques)
# ============================================================
st.subheader("🔎 Exploration détaillée (une variable à la fois)")

with st.expander("Déplier l’exploration détaillée", expanded=False):
    left, right = st.columns([1.2, 1])

    with left:
        fr_choices = sorted(fr_choices_from_df(df))
        fr_default = tech2fr_name(df.columns[0]) if len(df.columns) else ""
        fr_label = st.selectbox("Choisir une variable", fr_choices,
                                index=fr_choices.index(fr_default) if fr_default in fr_choices else 0)
        var = fr2tech_lookup(fr_label, df)
        s = df[var]
        typ, sub = _infer_kind(s)
        n_unique = int(s.dropna().nunique())
        miss_pct = 100 * s.isna().mean()

        with st.container(border=True):
            st.markdown(f"**Nom (FR) :** `{fr_label}`  \n**Nom d’origine (CSV) :** `{var}`")
            st.markdown(f"- **Type :** {typ}  •  **Sous-type :** {sub}")
            st.markdown(f"- **Valeurs uniques :** {n_unique:,}".replace(",", " "))
            st.markdown(f"- **% manquants :** {miss_pct:.2f}%")

        st.markdown("#### Description")
        if pd.api.types.is_numeric_dtype(s):
            desc = s.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).to_frame("Valeur")
            st.dataframe(desc, use_container_width=True)
        elif pd.api.types.is_datetime64_any_dtype(s):
            nonan = s.dropna()
            if nonan.empty:
                st.info("Aucune date exploitable.")
            else:
                st.write({
                    "min": str(nonan.min()),
                    "max": str(nonan.max()),
                    "nb. jours couverts (approx.)": int((nonan.max() - nonan.min()).days)
                })
        else:
            k = st.slider("Top K modalités", 5, 30, 15, key="topk_cat_overview")
            vc = s.astype(str).value_counts(dropna=False).head(k).reset_index()
            vc.columns = [fr_label, "Effectif"]
            st.dataframe(vc, use_container_width=True)

    with right:
        st.markdown("#### Échantillon (nettoyé)")
        sample_size = st.slider("Taille de l’échantillon", 5, 50, 10, key="samp_overview")
        ex = (df[[var]].dropna().astype(str).head(sample_size))
        ex.columns = [fr_label]
        st.dataframe(ex, use_container_width=True, hide_index=True)

st.divider()

# ============================================================
# 3) Export léger (optionnel)
# ============================================================
col_a, col_b = st.columns(2)
col_a.download_button(
    "💾 Télécharger le dictionnaire (CSV)",
    data=dict_df.to_csv(index=False).encode("utf-8"),
    file_name="dictionnaire_variables_fr.csv",
    mime="text/csv",
    use_container_width=True
)
col_b.download_button(
    "💾 Télécharger un échantillon (500 lignes)",
    data=df.head(500).to_csv(index=False).encode("utf-8"),
    file_name="echantillon_500.csv",
    mime="text/csv",
    use_container_width=True
)
