
# data_store.py
import json, os, pandas as pd

def load_columns_fr(path: str = "config/columns_fr.json") -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def get_label_maps(df: pd.DataFrame, fr_map: dict | None = None):
    fr_map = fr_map or load_columns_fr()
    tech2fr = {c: fr_map.get(c, c) for c in df.columns}
    fr2tech = {}
    for k, v in tech2fr.items():
        if v not in fr2tech:
            fr2tech[v] = k
    return tech2fr, fr2tech

def resolve_col(choice: str, fr2tech: dict) -> str:
    return fr2tech.get(choice, choice)




# def _normalize(df: pd.DataFrame) -> pd.DataFrame:
#     if df.empty:
#         return df.copy()

#     out = df.copy()

#     # --- géo : si geo_point_2d "lat, lon" existe, créer latitude/longitude ---
#     if "geo_point_2d" in out.columns and (("latitude" not in out.columns) or ("longitude" not in out.columns)):
#         coords = out["geo_point_2d"].astype(str).str.split(",", n=1, expand=True)
#         if coords.shape[1] == 2:
#             out["latitude"]  = pd.to_numeric(coords[0].str.strip(), errors="coerce")
#             out["longitude"] = pd.to_numeric(coords[1].str.strip(), errors="coerce")

#     # harmoniser géo
#     for c in ("latitude","longitude"):
#         if c in out.columns:
#             out[c] = pd.to_numeric(out[c], errors="coerce")

#     # --- date / heure ---
#     if "date" in out.columns:
#         out["date"] = pd.to_datetime(out["date"], dayfirst=True, errors="coerce")  # tes dates semblent au format JJ/MM/AAAA
#     else:
#         out["date"] = pd.NaT

#     if "heure" in out.columns:
#         h = out["heure"].astype(str).str.replace("h", ":", regex=False)
#         out["heure_num"] = pd.to_datetime(h, errors="coerce").dt.hour
#     else:
#         out["heure_num"] = np.nan

#     # --- clés temporelles ---
#     out["annee"]    = out["date"].dt.year
#     out["mois_str"] = out["date"].dt.to_period("M").astype(str)
#     try:
#         out["jour_sem"] = out["date"].dt.day_name(locale="fr_FR")
#     except Exception:
#         map_j = {0:"lundi",1:"mardi",2:"mercredi",3:"jeudi",4:"vendredi",5:"samedi",6:"dimanche"}
#         out["jour_sem"] = out["date"].dt.dayofweek.map(map_j)

#     # --- identifiants / textes utiles ---
#     if "code_insee" in out.columns:
#         out["code_insee"] = out["code_insee"].astype(str)

#     # normalise qq colonnes catégorielles si présentes
#     for c in ["commune", "lieu", "luminosite", "conditions_atm", "etat_surface", "agg", "catr"]:
#         if c in out.columns:
#             out[c] = out[c].astype(str).str.strip()

#     return out


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = df.copy()

    # --- géo : si geo_point_2d "lat, lon" existe, créer latitude/longitude ---
    if "geo_point_2d" in out.columns and (("latitude" not in out.columns) or ("longitude" not in out.columns)):
        coords = out["geo_point_2d"].astype(str).str.split(",", n=1, expand=True)
        if coords.shape[1] == 2:
            out["latitude"]  = pd.to_numeric(coords[0].str.strip(), errors="coerce")
            out["longitude"] = pd.to_numeric(coords[1].str.strip(), errors="coerce")

    # harmoniser géo
    for c in ("latitude","longitude"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # --- date / heure ---
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], dayfirst=True, errors="coerce")
    else:
        out["date"] = pd.NaT

    if "heure" in out.columns:
        h = out["heure"].astype(str).str.replace("h", ":", regex=False)
        out["heure_num"] = pd.to_datetime(h, errors="coerce").dt.hour
    elif "heure_num" not in out.columns:
        out["heure_num"] = np.nan

    # --- clés temporelles ---
    out["annee"]    = out["date"].dt.year
    out["mois_str"] = out["date"].dt.to_period("M").astype(str)
    try:
        out["jour_sem"] = out["date"].dt.day_name(locale="fr_FR")
    except Exception:
        map_j = {0:"lundi",1:"mardi",2:"mercredi",3:"jeudi",4:"vendredi",5:"samedi",6:"dimanche"}
        out["jour_sem"] = out["date"].dt.dayofweek.map(map_j)

    # --- identifiants / textes utiles ---
    if "code_insee" in out.columns:
        out["code_insee"] = out["code_insee"].astype(str)

    # --- Nettoyage des variables qualitatives (corrige valeurs aberrantes) ---
    cat_cols = [c for c in out.columns if out[c].dtype == "object" or out[c].dtype.name == "category"]
    for c in cat_cols:
        codebook = None
        if c in CODEBOOK_FILES and os.path.exists(CODEBOOK_FILES[c]):
            codebook = _load_json_safe(CODEBOOK_FILES[c])
        # ex: "commune_raw" si vous voulez garder l'original
        # out[c + "_raw"] = out[c]

        out[c] = _clean_categorical(
            out[c],
            mapping_codes=codebook,
            rare_min_freq=50,               # à ajuster si besoin
            label_non_ren="Non renseigné",
            label_autre="Autre"
        )

    # --- Normalise qq colonnes spécifiques si présentes (votre logique existante) ---
    for c in ["commune", "lieu", "luminosite", "conditions_atm", "etat_surface", "agg", "catr"]:
        if c in out.columns:
            out[c] = out[c].astype(str).str.strip()

    # --- Alias de colonnes en FR (sans casser l'existant) ---
    #   -> crée des colonnes "alias" FR *en plus* des originales (mode sûr)
    if os.path.exists(COLUMNS_FR_FILE):
        fr_map = _load_json_safe(COLUMNS_FR_FILE)
        for src, fr in fr_map.items():
            if src in out.columns and fr not in out.columns:
                out[fr] = out[src]

    return out
