

# from ui_nav import nav_bar
# nav_bar(active="Analyse temporelle")   # ou "Analyse spatiale", "Corrélations", "Insights & recos"

# # pages/02_Analyse_temporelle.py
# import pandas as pd
# import numpy as np
# import streamlit as st
# import plotly.express as px

# st.set_page_config(page_title="Accidentologie – Analyse temporelle", page_icon="⏱️", layout="wide")
# st.title("⏱️ Analyse temporelle")
# st.caption("Cycles journaliers/hebdomadaires/mensuels • Lissage • Comparaison de périodes • Heatmap.")

# # ========= Chargement des données =========
# def _load_local_csv(path="acc.csv") -> pd.DataFrame:
#     # lecture robuste ; d’abord ; puis ,
#     try:
#         df = pd.read_csv(path, sep=";", low_memory=False)
#     except Exception:
#         df = pd.read_csv(path, low_memory=False)

#     # normalisations minimales
#     df["date"] = pd.to_datetime(df.get("date"), errors="coerce", dayfirst=True)

#     if "heure" in df.columns:
#         h = df["heure"].astype(str).str.replace("h", ":", regex=False)
#         df["heure_num"] = pd.to_datetime(h, errors="coerce").dt.hour
#     else:
#         df["heure_num"] = pd.to_numeric(df.get("heure_num"), errors="coerce")

#     df["annee"] = df["date"].dt.year
#     df["jour"] = df["date"].dt.date
#     df["mois_str"] = df["date"].dt.to_period("M").astype(str)
#     try:
#         df["jour_sem"] = df["date"].dt.day_name(locale="fr_FR")
#     except Exception:
#         df["jour_sem"] = df["date"].dt.dayofweek.map({
#             0:"lundi",1:"mardi",2:"mercredi",3:"jeudi",4:"vendredi",5:"samedi",6:"dimanche"
#         })
#     df["sem_iso"] = df["date"].dt.isocalendar().week.astype("Int64") if df["date"].notna().any() else np.nan
#     if "code_insee" in df.columns:
#         df["code_insee"] = df["code_insee"].astype(str)
#     return df

# # 1) si app.py a mis un df en session, on l’utilise
# if "df" in st.session_state and isinstance(st.session_state.df, pd.DataFrame) and not st.session_state.df.empty:
#     df = st.session_state.df.copy()
# # 2) sinon, autonomie : lecture directe d'acc.csv
# else:
#     df = _load_local_csv("acc.csv")

# if df is None or df.empty:
#     st.error("Impossible de récupérer les données. Assure-toi d’avoir un acc.csv à la racine.")
#     st.stop()

# # ========= Sidebar filtres =========
# st.sidebar.header("⚙️ Paramètres – Temporel")

# years = sorted([int(a) for a in df["annee"].dropna().unique()]) if "annee" in df else []
# y_min, y_max = (years[0], years[-1]) if years else (None, None)
# if years:
#     y_min, y_max = st.sidebar.select_slider("Période (années)", options=years, value=(y_min, y_max), key="t_years")

# hmin, hmax = st.sidebar.select_slider("Plage horaire", options=list(range(24)), value=(6, 20), key="t_hours")

# insee_opts = sorted(df["code_insee"].dropna().unique()) if "code_insee" in df else []
# insee_sel = st.sidebar.multiselect("Commune (code INSEE)", insee_opts, default=[], key="t_insee")

# gran = st.sidebar.radio("Granularité", ["Jour", "Semaine ISO", "Mois"], horizontal=True, key="t_gran")
# win = st.sidebar.slider("Lissage (moyenne mobile – fenêtres)", 1, 30, 7, key="t_win")

# st.sidebar.markdown("---")
# st.sidebar.subheader("Comparer deux périodes")
# cmp_on = st.sidebar.checkbox("Activer", key="t_cmp_on")
# if cmp_on and years:
#     colA, colB = st.sidebar.columns(2)
#     yA0, yA1 = colA.select_slider("Période A", options=years, value=(years[0], years[-1]), key="t_A")
#     yB0, yB1 = colB.select_slider("Période B", options=years, value=(years[0], years[-1]), key="t_B")
#     mode_cmp = st.sidebar.radio("Affichage", ["Superposition", "Côte à côte"], horizontal=True, key="t_cmp_mode")

# # ========= Application des filtres =========
# mask = pd.Series(True, index=df.index)
# if y_min is not None:
#     mask &= df["annee"].between(y_min, y_max)
# mask &= df["heure_num"].between(hmin, hmax) | df["heure_num"].isna()
# if insee_sel:
#     mask &= df["code_insee"].isin(insee_sel)

# df_f = df[mask].copy()
# if df_f.empty:
#     st.info("Aucune donnée pour ces filtres. Élargis la période ou la plage horaire.")
#     st.stop()

# # ========= KPIs =========
# k1, k2, k3, k4 = st.columns(4)
# k1.metric("Accidents (filtrés)", f"{len(df_f):,}".replace(",", " "))
# if df_f["date"].notna().any():
#     span_days = max((df_f["date"].max() - df_f["date"].min()).days, 1)
#     dens = round(len(df_f) / span_days, 3)
# else:
#     dens = "—"
# k2.metric("Densité acc./jour", dens)
# modal_hour = int(df_f["heure_num"].mode().iloc[0]) if df_f["heure_num"].dropna().size else "—"
# k3.metric("Heure modale", modal_hour)
# top_day = df_f["jour_sem"].mode().iloc[0] if df_f["jour_sem"].dropna().size else "—"
# k4.metric("Jour le + accidentogène", top_day)

# st.markdown("---")

# # ========= Helpers =========
# def agg_time(data: pd.DataFrame, granularite: str) -> pd.DataFrame:
#     if data.empty:
#         return pd.DataFrame(columns=["x", "accidents"])
#     if granularite == "Jour":
#         s = data.groupby("jour").size().rename("accidents").reset_index()
#         s["x"] = pd.to_datetime(s["jour"])
#     elif granularite == "Semaine ISO":
#         tmp = data.dropna(subset=["annee","sem_iso"]).copy()
#         tmp["iso_date"] = pd.to_datetime(tmp["annee"].astype(str) + "-1", errors="coerce") \
#                           + pd.to_timedelta((tmp["sem_iso"] - 1).astype("Int64") * 7, unit="D")
#         s = tmp.groupby(["annee","sem_iso","iso_date"]).size().rename("accidents").reset_index()
#         s["x"] = s["iso_date"]
#     else:  # Mois
#         s = data.groupby("mois_str").size().rename("accidents").reset_index()
#         s["x"] = pd.to_datetime(s["mois_str"], errors="coerce")
#     return s.sort_values("x")

# def add_sma(df_line: pd.DataFrame, w: int) -> pd.DataFrame:
#     if df_line.empty:
#         return df_line
#     if w <= 1:
#         df_line["sma"] = df_line["accidents"]
#     else:
#         df_line["sma"] = df_line["accidents"].rolling(window=w, min_periods=max(1, w//2)).mean()
#     return df_line

# # ========= Série principale =========
# st.subheader("Évolution du nombre d’accidents")
# serie = add_sma(agg_time(df_f, gran), win)

# colL, colR = st.columns((1.6, 1))
# with colL:
#     fig_evo = px.line(serie, x="x", y="accidents", markers=True, labels={"x":"Temps","accidents":"Accidents"})
#     fig_evo.add_scatter(x=serie["x"], y=serie["sma"], mode="lines", name=f"SMA {win}")
#     st.plotly_chart(fig_evo, use_container_width=True)

# with colR:
#     st.markdown("**Répartition par jour de semaine**")
#     jour_order = ["lundi","mardi","mercredi","jeudi","vendredi","samedi","dimanche"]
#     dist = df_f.groupby("jour_sem").size().rename("accidents").reset_index()
#     dist["ord"] = dist["jour_sem"].apply(lambda x: jour_order.index(x) if x in jour_order else 99)
#     dist = dist.sort_values("ord")
#     st.plotly_chart(px.bar(dist, x="jour_sem", y="accidents",
#                            labels={"jour_sem":"Jour","accidents":"Accidents"}),
#                     use_container_width=True)

# st.markdown("---")

# # ========= Comparaison de périodes =========
# if cmp_on and years:
#     st.subheader("Comparaison de périodes")

#     def subperiod(data, a0, a1):
#         m = data["annee"].between(a0, a1) if a0 is not None else True
#         return data[m].copy()

#     A = subperiod(df_f, yA0, yA1)
#     B = subperiod(df_f, yB0, yB1)
#     serieA = add_sma(agg_time(A, gran), win)
#     serieB = add_sma(agg_time(B, gran), win)

#     if mode_cmp == "Superposition":
#         fig_cmp = px.line(serieA, x="x", y="accidents", markers=True, labels={"x":"Temps","accidents":"Accidents"})
#         fig_cmp.add_scatter(x=serieA["x"], y=serieA["sma"], mode="lines", name=f"A – SMA {win}")
#         fig_cmp.add_scatter(x=serieB["x"], y=serieB["accidents"], mode="lines+markers", name="B – brut")
#         fig_cmp.add_scatter(x=serieB["x"], y=serieB["sma"], mode="lines", name=f"B – SMA {win}")
#         st.plotly_chart(fig_cmp, use_container_width=True)
#     else:
#         cA, cB = st.columns(2)
#         with cA:
#             st.markdown(f"**Période A : {yA0}–{yA1}**")
#             figA = px.line(serieA, x="x", y="accidents", markers=True)
#             figA.add_scatter(x=serieA["x"], y=serieA["sma"], mode="lines", name=f"SMA {win}")
#             st.plotly_chart(figA, use_container_width=True)
#         with cB:
#             st.markdown(f"**Période B : {yB0}–{yB1}**")
#             figB = px.line(serieB, x="x", y="accidents", markers=True)
#             figB.add_scatter(x=serieB["x"], y=serieB["sma"], mode="lines", name=f"SMA {win}")
#             st.plotly_chart(figB, use_container_width=True)

#     st.caption(f"Δ (B − A) = **{len(B) - len(A):+,}** accidents".replace(",", " "))

# st.markdown("---")

# # ========= Heatmap Heure × Jour =========
# st.subheader("Heatmap – Heure × Jour de la semaine")
# df_hm = df_f.dropna(subset=["heure_num","jour_sem"]).copy()
# if df_hm.empty:
#     st.info("Données horaires insuffisantes pour la heatmap.")
# else:
#     ordre = ["lundi","mardi","mercredi","jeudi","vendredi","samedi","dimanche"]
#     df_hm["jour_sem"] = pd.Categorical(df_hm["jour_sem"], categories=ordre, ordered=True)
#     hm = df_hm.groupby(["jour_sem","heure_num"]).size().rename("accidents").reset_index()
#     mat = hm.pivot(index="jour_sem", columns="heure_num", values="accidents").fillna(0)
#     st.plotly_chart(px.imshow(mat, aspect="auto", labels=dict(x="Heure", y="Jour", color="Accidents")),
#                     use_container_width=True)

# st.markdown("---")

# # ========= Tops =========
# st.subheader("⛳ Pics temporels")
# cL, cR = st.columns(2)
# with cL:
#     st.markdown("**Top heures**")
#     top_h = (df_f.groupby("heure_num").size()
#              .rename("accidents").reset_index()
#              .sort_values("accidents", ascending=False).head(10))
#     st.dataframe(top_h, use_container_width=True)
# with cR:
#     st.markdown("**Top jours (jour de semaine)**")
#     top_d = (df_f.groupby("jour_sem").size()
#              .rename("accidents").reset_index()
#              .sort_values("accidents", ascending=False))
#     st.dataframe(top_d, use_container_width=True)

# # ========= Exports =========
# st.markdown("---")
# cE1, cE2 = st.columns(2)
# cE1.download_button(
#     "💾 Export – données filtrées (CSV)",
#     data=df_f.to_csv(index=False).encode("utf-8"),
#     file_name="acc_temporel_filtre.csv",
#     mime="text/csv",
#     use_container_width=True
# )
# subset_cols = [c for c in ["date","annee","mois_str","jour_sem","heure_num"] if c in df_f.columns]
# if subset_cols:
#     cE2.download_button(
#         "📊 Export – sous-ensemble temporel (CSV)",
#         data=df_f[subset_cols].to_csv(index=False).encode("utf-8"),
#         file_name="acc_temporel_subset.csv",
#         mime="text/csv",
#         use_container_width=True
#     )

# # ========= Notes =========
# with st.expander("🧪 Notes méthodologiques"):
#     st.markdown(
#         "- **Granularité** : agrégation par jour, semaine ISO (approx) ou mois.\n"
#         "- **Lissage** : moyenne mobile pour révéler la tendance.\n"
#         "- **Comparaison** : overlay ou côte-à-côte sur deux fenêtres temporelles.\n"
#         "- **Heatmap** : identifie les créneaux (jour×heure) les plus accidentogènes."
#     )



# # pages/02_Analyse_temporelle.py

# import pandas as pd
# import numpy as np
# import streamlit as st
# import plotly.express as px
# import plotly.graph_objects as go

# from ui_nav import nav_bar

# # ========= Config générale =========
# st.set_page_config(
#     page_title="Accidentologie – Analyse temporelle",
#     page_icon="⏱️",
#     layout="wide"
# )

# st.title("⏱️ Analyse temporelle")
# st.caption("Cycles journaliers/hebdomadaires/mensuels • Lissage • Comparaison de périodes.")

# # Barre de navigation (sous le titre)
# nav_bar(active="Analyse temporelle")
# st.markdown("---")

# # ========= Chargement des données =========
# def _load_local_csv(path="acc.csv") -> pd.DataFrame:
#     try:
#         df = pd.read_csv(path, sep=";", low_memory=False)
#     except Exception:
#         df = pd.read_csv(path, low_memory=False)

#     df["date"] = pd.to_datetime(df.get("date"), errors="coerce", dayfirst=True)

#     if "heure" in df.columns:
#         h = df["heure"].astype(str).str.replace("h", ":", regex=False)
#         df["heure_num"] = pd.to_datetime(h, errors="coerce").dt.hour
#     else:
#         df["heure_num"] = pd.to_numeric(df.get("heure_num"), errors="coerce")

#     df["annee"] = df["date"].dt.year
#     df["jour"] = df["date"].dt.date
#     df["mois_str"] = df["date"].dt.to_period("M").astype(str)
#     try:
#         df["jour_sem"] = df["date"].dt.day_name(locale="fr_FR")
#     except Exception:
#         df["jour_sem"] = df["date"].dt.dayofweek.map({
#             0: "lundi", 1: "mardi", 2: "mercredi", 3: "jeudi",
#             4: "vendredi", 5: "samedi", 6: "dimanche"
#         })
#     df["sem_iso"] = (
#         df["date"].dt.isocalendar().week.astype("Int64")
#         if df["date"].notna().any() else np.nan
#     )
#     if "code_insee" in df.columns:
#         df["code_insee"] = df["code_insee"].astype(str)
#     return df

# # 1) si app.py a mis un df en session, on l’utilise
# if "df" in st.session_state and isinstance(st.session_state.df, pd.DataFrame) and not st.session_state.df.empty:
#     df = st.session_state.df.copy()
# # 2) sinon, lecture directe d'acc.csv
# else:
#     df = _load_local_csv("acc.csv")

# if df is None or df.empty:
#     st.error("Impossible de récupérer les données. Assure-toi d’avoir un acc.csv à la racine.")
#     st.stop()

# # ========= Sidebar filtres =========
# st.sidebar.header("⚙️ Paramètres – Temporel")

# years = sorted([int(a) for a in df["annee"].dropna().unique()]) if "annee" in df else []
# y_min, y_max = (years[0], years[-1]) if years else (None, None)
# if years:
#     y_min, y_max = st.sidebar.select_slider(
#         "Période (années)", options=years, value=(y_min, y_max), key="t_years"
#     )

# hmin, hmax = st.sidebar.select_slider(
#     "Plage horaire", options=list(range(24)), value=(6, 20), key="t_hours"
# )

# insee_opts = sorted(df["code_insee"].dropna().unique()) if "code_insee" in df else []
# insee_sel = st.sidebar.multiselect(
#     "Commune (code INSEE)", insee_opts, default=[], key="t_insee"
# )

# gran = st.sidebar.radio(
#     "Granularité", ["Jour", "Semaine ISO", "Mois"],
#     horizontal=True, key="t_gran"
# )
# win = st.sidebar.slider(
#     "Lissage (moyenne mobile – fenêtres)", 1, 30, 7, key="t_win"
# )

# st.sidebar.markdown("---")
# st.sidebar.subheader("Comparer deux périodes")
# cmp_on = st.sidebar.checkbox("Activer", key="t_cmp_on")
# if cmp_on and years:
#     colA, colB = st.sidebar.columns(2)
#     yA0, yA1 = colA.select_slider(
#         "Période A", options=years, value=(years[0], years[-1]), key="t_A"
#     )
#     yB0, yB1 = colB.select_slider(
#         "Période B", options=years, value=(years[0], years[-1]), key="t_B"
#     )
#     mode_cmp = st.sidebar.radio(
#         "Affichage", ["Superposition", "Côte à côte"],
#         horizontal=True, key="t_cmp_mode"
#     )

# # ========= Application des filtres =========
# mask = pd.Series(True, index=df.index)
# if y_min is not None:
#     mask &= df["annee"].between(y_min, y_max)
# mask &= df["heure_num"].between(hmin, hmax) | df["heure_num"].isna()
# if insee_sel:
#     mask &= df["code_insee"].isin(insee_sel)

# df_f = df[mask].copy()
# if df_f.empty:
#     st.info("Aucune donnée pour ces filtres. Élargis la période ou la plage horaire.")
#     st.stop()

# # ========= KPIs =========
# k1, k2, k3, k4 = st.columns(4)
# k1.metric("Accidents (filtrés)", f"{len(df_f):,}".replace(",", " "))

# if df_f["date"].notna().any():
#     span_days = max((df_f["date"].max() - df_f["date"].min()).days, 1)
#     dens = round(len(df_f) / span_days, 3)
# else:
#     dens = "—"
# k2.metric("Densité acc./jour", dens)

# modal_hour = int(df_f["heure_num"].mode().iloc[0]) if df_f["heure_num"].dropna().size else "—"
# k3.metric("Heure modale", modal_hour)

# top_day = df_f["jour_sem"].mode().iloc[0] if df_f["jour_sem"].dropna().size else "—"
# k4.metric("Jour le + accidentogène", top_day)

# st.markdown("---")

# # ========= Helpers =========
# def agg_time(data: pd.DataFrame, granularite: str) -> pd.DataFrame:
#     if data.empty:
#         return pd.DataFrame(columns=["x", "accidents"])
#     if granularite == "Jour":
#         s = data.groupby("jour").size().rename("accidents").reset_index()
#         s["x"] = pd.to_datetime(s["jour"])
#     elif granularite == "Semaine ISO":
#         tmp = data.dropna(subset=["annee", "sem_iso"]).copy()
#         tmp["iso_date"] = (
#             pd.to_datetime(tmp["annee"].astype(str) + "-1", errors="coerce")
#             + pd.to_timedelta((tmp["sem_iso"] - 1).astype("Int64") * 7, unit="D")
#         )
#         s = tmp.groupby(["annee", "sem_iso", "iso_date"]).size().rename("accidents").reset_index()
#         s["x"] = s["iso_date"]
#     else:  # Mois
#         s = data.groupby("mois_str").size().rename("accidents").reset_index()
#         s["x"] = pd.to_datetime(s["mois_str"], errors="coerce")
#     return s.sort_values("x")

# def add_sma(df_line: pd.DataFrame, w: int) -> pd.DataFrame:
#     if df_line.empty:
#         return df_line
#     if w <= 1:
#         df_line["sma"] = df_line["accidents"]
#     else:
#         df_line["sma"] = df_line["accidents"].rolling(
#             window=w, min_periods=max(1, w // 2)
#         ).mean()
#     return df_line

# # ========= Série principale =========
# st.subheader("Évolution du nombre d’accidents")
# serie = add_sma(agg_time(df_f, gran), win)

# colL, colR = st.columns((1.6, 1))

# # Palette cohérente
# COLOR_LINE = "#4c78a8"   # bleu
# COLOR_SMA = "#f58518"    # orange
# BAR_PALETTE = px.colors.qualitative.Set3

# with colL:
#     fig_evo = px.line(
#         serie,
#         x="x",
#         y="accidents",
#         markers=True,
#         labels={"x": "Temps", "accidents": "Accidents"},
#         color_discrete_sequence=[COLOR_LINE],
#     )
#     fig_evo.update_traces(
#         line=dict(width=2.2),
#         marker=dict(size=5)
#     )
#     fig_evo.add_scatter(
#         x=serie["x"],
#         y=serie["sma"],
#         mode="lines",
#         name=f"SMA {win}",
#         line=dict(color=COLOR_SMA, width=3)
#     )
#     fig_evo.update_layout(
#         height=430,
#         margin=dict(l=0, r=0, t=40, b=0),
#         legend=dict(
#             orientation="h",
#             yanchor="bottom",
#             y=1.02,
#             xanchor="right",
#             x=1
#         )
#     )
#     st.plotly_chart(fig_evo, use_container_width=True)

# with colR:
#     st.markdown("**Répartition par jour de semaine**")
#     jour_order = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
#     dist = df_f.groupby("jour_sem").size().rename("accidents").reset_index()
#     dist["ord"] = dist["jour_sem"].apply(
#         lambda x: jour_order.index(x) if x in jour_order else 99
#     )
#     dist = dist.sort_values("ord")

#     fig_bar = px.bar(
#         dist,
#         x="jour_sem",
#         y="accidents",
#         labels={"jour_sem": "Jour", "accidents": "Accidents"},
#         color="jour_sem",
#         color_discrete_sequence=BAR_PALETTE,
#         text_auto=True
#     )
#     fig_bar.update_traces(
#         marker=dict(line=dict(width=0)),
#         opacity=0.9,
#         textfont=dict(size=11)
#     )
#     fig_bar.update_layout(
#         height=430,
#         margin=dict(l=0, r=0, t=40, b=0),
#         showlegend=False
#     )
#     st.plotly_chart(fig_bar, use_container_width=True)

# st.markdown("---")

# # ========= Comparaison de périodes =========
# if cmp_on and years:
#     st.subheader("Comparaison de périodes")

#     def subperiod(data, a0, a1):
#         m = data["annee"].between(a0, a1) if a0 is not None else True
#         return data[m].copy()

#     A = subperiod(df_f, yA0, yA1)
#     B = subperiod(df_f, yB0, yB1)
#     serieA = add_sma(agg_time(A, gran), win)
#     serieB = add_sma(agg_time(B, gran), win)

#     if mode_cmp == "Superposition":
#         fig_cmp = px.line(
#             serieA,
#             x="x",
#             y="accidents",
#             markers=True,
#             labels={"x": "Temps", "accidents": "Accidents"},
#             color_discrete_sequence=[COLOR_LINE],
#         )
#         fig_cmp.update_traces(line=dict(width=2), marker=dict(size=5), name="A – brut")
#         fig_cmp.add_scatter(
#             x=serieA["x"],
#             y=serieA["sma"],
#             mode="lines",
#             name=f"A – SMA {win}",
#             line=dict(color=COLOR_LINE, width=3, dash="dash")
#         )
#         fig_cmp.add_scatter(
#             x=serieB["x"],
#             y=serieB["accidents"],
#             mode="lines+markers",
#             name="B – brut",
#             line=dict(color=COLOR_SMA, width=2),
#             marker=dict(size=5)
#         )
#         fig_cmp.add_scatter(
#             x=serieB["x"],
#             y=serieB["sma"],
#             mode="lines",
#             name=f"B – SMA {win}",
#             line=dict(color=COLOR_SMA, width=3, dash="dot")
#         )
#         fig_cmp.update_layout(
#             height=430,
#             margin=dict(l=0, r=0, t=40, b=0),
#             legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
#         )
#         st.plotly_chart(fig_cmp, use_container_width=True)
#     else:
#         cA, cB = st.columns(2)
#         with cA:
#             st.markdown(f"**Période A : {yA0}–{yA1}**")
#             figA = px.line(
#                 serieA, x="x", y="accidents", markers=True,
#                 color_discrete_sequence=[COLOR_LINE]
#             )
#             figA.update_traces(line=dict(width=2), marker=dict(size=5))
#             figA.add_scatter(
#                 x=serieA["x"], y=serieA["sma"],
#                 mode="lines", name=f"SMA {win}",
#                 line=dict(color=COLOR_SMA, width=3)
#             )
#             figA.update_layout(height=380, margin=dict(l=0, r=0, t=40, b=0),
#                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
#             st.plotly_chart(figA, use_container_width=True)
#         with cB:
#             st.markdown(f"**Période B : {yB0}–{yB1}**")
#             figB = px.line(
#                 serieB, x="x", y="accidents", markers=True,
#                 color_discrete_sequence=[COLOR_LINE]
#             )
#             figB.update_traces(line=dict(width=2), marker=dict(size=5))
#             figB.add_scatter(
#                 x=serieB["x"], y=serieB["sma"],
#                 mode="lines", name=f"SMA {win}",
#                 line=dict(color=COLOR_SMA, width=3)
#             )
#             figB.update_layout(height=380, margin=dict(l=0, r=0, t=40, b=0),
#                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
#             st.plotly_chart(figB, use_container_width=True)

#     st.caption(f"Δ (B − A) = **{len(B) - len(A):+,}** accidents".replace(",", " "))

# st.markdown("---")

# # ========= Tops =========
# st.subheader("⛳ Pics temporels")
# cL, cR = st.columns(2)
# with cL:
#     st.markdown("**Top heures**")
#     top_h = (
#         df_f.groupby("heure_num").size()
#         .rename("accidents").reset_index()
#         .sort_values("accidents", ascending=False).head(10)
#     )
#     st.dataframe(top_h, use_container_width=True)
# with cR:
#     st.markdown("**Top jours (jour de semaine)**")
#     top_d = (
#         df_f.groupby("jour_sem").size()
#         .rename("accidents").reset_index()
#         .sort_values("accidents", ascending=False)
#     )
#     st.dataframe(top_d, use_container_width=True)

# # ========= Exports =========
# st.markdown("---")
# cE1, cE2 = st.columns(2)
# cE1.download_button(
#     "💾 Export – données filtrées (CSV)",
#     data=df_f.to_csv(index=False).encode("utf-8"),
#     file_name="acc_temporel_filtre.csv",
#     mime="text/csv",
#     use_container_width=True
# )
# subset_cols = [c for c in ["date", "annee", "mois_str", "jour_sem", "heure_num"] if c in df_f.columns]
# if subset_cols:
#     cE2.download_button(
#         "📊 Export – sous-ensemble temporel (CSV)",
#         data=df_f[subset_cols].to_csv(index=False).encode("utf-8"),
#         file_name="acc_temporel_subset.csv",
#         mime="text/csv",
#         use_container_width=True
#     )

# # ========= Notes =========
# with st.expander("🧪 Notes méthodologiques"):
#     st.markdown(
#         "- **Granularité** : agrégation par jour, semaine ISO (approx) ou mois.\n"
#         "- **Lissage** : moyenne mobile pour révéler la tendance.\n"
#         "- **Comparaison** : overlay ou côte-à-côte sur deux fenêtres temporelles.\n"
#         "- **Pics temporels** : tableaux pour identifier les heures et jours les plus accidentogènes."
#     )




# pages/02_Analyse_temporelle.py

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from ui_nav import nav_bar

# ========= Config générale =========
st.set_page_config(
    page_title="Accidentologie – Analyse temporelle",
    page_icon="⏱️",
    layout="wide"
)

st.title("⏱️ Analyse temporelle")
st.caption("Cycles journaliers/hebdomadaires/mensuels • Lissage • Comparaison de périodes.")

# Barre de navigation (sous le titre)
nav_bar(active="Analyse temporelle")
st.markdown("---")

# ========= Chargement des données =========
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

    df["annee"] = df["date"].dt.year
    df["jour"] = df["date"].dt.date
    df["mois_str"] = df["date"].dt.to_period("M").astype(str)
    try:
        df["jour_sem"] = df["date"].dt.day_name(locale="fr_FR")
    except Exception:
        df["jour_sem"] = df["date"].dt.dayofweek.map({
            0: "lundi", 1: "mardi", 2: "mercredi", 3: "jeudi",
            4: "vendredi", 5: "samedi", 6: "dimanche"
        })
    df["sem_iso"] = (
        df["date"].dt.isocalendar().week.astype("Int64")
        if df["date"].notna().any() else np.nan
    )
    if "code_insee" in df.columns:
        df["code_insee"] = df["code_insee"].astype(str)
    return df

# 1) si app.py a mis un df en session, on l’utilise
if "df" in st.session_state and isinstance(st.session_state.df, pd.DataFrame) and not st.session_state.df.empty:
    df = st.session_state.df.copy()
# 2) sinon, lecture directe d'acc.csv
else:
    df = _load_local_csv("acc.csv")

if df is None or df.empty:
    st.error("Impossible de récupérer les données. Assure-toi d’avoir un acc.csv à la racine.")
    st.stop()

# ========= Sidebar filtres =========
st.sidebar.header("⚙️ Paramètres – Temporel")

years = sorted([int(a) for a in df["annee"].dropna().unique()]) if "annee" in df else []
y_min, y_max = (years[0], years[-1]) if years else (None, None)
if years:
    y_min, y_max = st.sidebar.select_slider(
        "Période (années)", options=years, value=(y_min, y_max), key="t_years"
    )

hmin, hmax = st.sidebar.select_slider(
    "Plage horaire", options=list(range(24)), value=(6, 20), key="t_hours"
)

insee_opts = sorted(df["code_insee"].dropna().unique()) if "code_insee" in df else []
insee_sel = st.sidebar.multiselect(
    "Commune (code INSEE)", insee_opts, default=[], key="t_insee"
)

# -> on met MOIS en premier pour que ce soit la valeur par défaut
gran = st.sidebar.radio(
    "Granularité",
    ["Mois", "Semaine ISO", "Jour"],
    horizontal=True,
    key="t_gran"
)

win = st.sidebar.slider(
    "Lissage (moyenne mobile – fenêtres)", 1, 30, 7, key="t_win"
)

st.sidebar.markdown("---")
st.sidebar.subheader("Comparer deux périodes")
cmp_on = st.sidebar.checkbox("Activer", key="t_cmp_on")
if cmp_on and years:
    colA, colB = st.sidebar.columns(2)
    yA0, yA1 = colA.select_slider(
        "Période A", options=years, value=(years[0], years[-1]), key="t_A"
    )
    yB0, yB1 = colB.select_slider(
        "Période B", options=years, value=(years[0], years[-1]), key="t_B"
    )
    mode_cmp = st.sidebar.radio(
        "Affichage", ["Superposition", "Côte à côte"],
        horizontal=True, key="t_cmp_mode"
    )

# ========= Application des filtres =========
mask = pd.Series(True, index=df.index)
if y_min is not None:
    mask &= df["annee"].between(y_min, y_max)
mask &= df["heure_num"].between(hmin, hmax) | df["heure_num"].isna()
if insee_sel:
    mask &= df["code_insee"].isin(insee_sel)

df_f = df[mask].copy()
if df_f.empty:
    st.info("Aucune donnée pour ces filtres. Élargis la période ou la plage horaire.")
    st.stop()

# ========= KPIs =========
k1, k2, k3, k4 = st.columns(4)
k1.metric("Accidents (filtrés)", f"{len(df_f):,}".replace(",", " "))

if df_f["date"].notna().any():
    span_days = max((df_f["date"].max() - df_f["date"].min()).days, 1)
    dens = round(len(df_f) / span_days, 3)
else:
    dens = "—"
k2.metric("Densité acc./jour", dens)

modal_hour = int(df_f["heure_num"].mode().iloc[0]) if df_f["heure_num"].dropna().size else "—"
k3.metric("Heure modale", modal_hour)

top_day = df_f["jour_sem"].mode().iloc[0] if df_f["jour_sem"].dropna().size else "—"
k4.metric("Jour le + accidentogène", top_day)

st.markdown("---")

# ========= Helpers =========
def agg_time(data: pd.DataFrame, granularite: str) -> pd.DataFrame:
    """Agrège le nombre d'accidents selon la granularité choisie."""
    if data.empty:
        return pd.DataFrame(columns=["x", "accidents"])
    if granularite == "Jour":
        s = data.groupby("jour").size().rename("accidents").reset_index()
        s["x"] = pd.to_datetime(s["jour"])
    elif granularite == "Semaine ISO":
        tmp = data.dropna(subset=["annee", "sem_iso"]).copy()
        tmp["iso_date"] = (
            pd.to_datetime(tmp["annee"].astype(str) + "-1", errors="coerce")
            + pd.to_timedelta((tmp["sem_iso"] - 1).astype("Int64") * 7, unit="D")
        )
        s = tmp.groupby(["annee", "sem_iso", "iso_date"]).size().rename("accidents").reset_index()
        s["x"] = s["iso_date"]
    else:  # Mois
        s = data.groupby("mois_str").size().rename("accidents").reset_index()
        s["x"] = pd.to_datetime(s["mois_str"], errors="coerce")
    return s.sort_values("x")

def add_sma(df_line: pd.DataFrame, w: int) -> pd.DataFrame:
    """Ajoute une colonne de moyenne mobile (SMA)."""
    if df_line.empty:
        return df_line
    if w <= 1 or len(df_line) < w:
        df_line["sma"] = df_line["accidents"]
    else:
        df_line["sma"] = df_line["accidents"].rolling(
            window=w, min_periods=max(1, w // 2)
        ).mean()
    return df_line

# ========= Série principale =========
st.subheader("Évolution du nombre d’accidents")

serie = agg_time(df_f, gran)
serie = add_sma(serie, win)

# on trace UNE SEULE courbe façon Excel :
#   - y = SMA si possible, sinon la série brute
y_col = "sma" if "sma" in serie.columns else "accidents"
label_y = {
    "Jour": "Nombre d'accidents par jour",
    "Semaine ISO": "Nombre d'accidents par semaine",
    "Mois": "Nombre d'accidents par mois",
}.get(gran, "Nombre d'accidents")

colL, colR = st.columns((1.6, 1))

COLOR_LINE = "#4c78a8"   # bleu
BAR_PALETTE = px.colors.qualitative.Set3

with colL:
    fig_evo = go.Figure()
    fig_evo.add_trace(go.Scatter(
        x=serie["x"],
        y=serie[y_col],
        mode="lines+markers",
        name="Accidents (lissés)",
        line=dict(color=COLOR_LINE, width=2.5),
        marker=dict(size=5)
    ))
    fig_evo.update_layout(
        xaxis_title="Temps",
        yaxis_title=label_y,
        height=430,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    st.plotly_chart(fig_evo, use_container_width=True)

with colR:
    st.markdown("**Répartition par jour de semaine**")
    jour_order = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
    dist = df_f.groupby("jour_sem").size().rename("accidents").reset_index()
    dist["ord"] = dist["jour_sem"].apply(
        lambda x: jour_order.index(x) if x in jour_order else 99
    )
    dist = dist.sort_values("ord")

    fig_bar = px.bar(
        dist,
        x="jour_sem",
        y="accidents",
        labels={"jour_sem": "Jour", "accidents": "Accidents"},
        color="jour_sem",
        color_discrete_sequence=BAR_PALETTE,
        text_auto=True
    )
    fig_bar.update_traces(
        marker=dict(line=dict(width=0)),
        opacity=0.9,
        textfont=dict(size=11)
    )
    fig_bar.update_layout(
        height=430,
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False
    )
    st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")

# ========= Comparaison de périodes =========
if cmp_on and years:
    st.subheader("Comparaison de périodes")

    def subperiod(data, a0, a1):
        m = data["annee"].between(a0, a1) if a0 is not None else True
        return data[m].copy()

    A = subperiod(df_f, yA0, yA1)
    B = subperiod(df_f, yB0, yB1)
    serieA = add_sma(agg_time(A, gran), win)
    serieB = add_sma(agg_time(B, gran), win)

    if mode_cmp == "Superposition":
        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Scatter(
            x=serieA["x"], y=serieA["sma"],
            mode="lines+markers", name=f"A ({yA0}-{yA1})",
            line=dict(color="#4c78a8", width=2.5),
            marker=dict(size=5)
        ))
        fig_cmp.add_trace(go.Scatter(
            x=serieB["x"], y=serieB["sma"],
            mode="lines+markers", name=f"B ({yB0}-{yB1})",
            line=dict(color="#f58518", width=2.5),
            marker=dict(size=5)
        ))
        fig_cmp.update_layout(
            xaxis_title="Temps",
            yaxis_title=label_y,
            height=430,
            margin=dict(l=0, r=0, t=40, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_cmp, use_container_width=True)
    else:
        cA, cB = st.columns(2)
        with cA:
            st.markdown(f"**Période A : {yA0}–{yA1}**")
            figA = go.Figure()
            figA.add_trace(go.Scatter(
                x=serieA["x"], y=serieA["sma"],
                mode="lines+markers", name="A",
                line=dict(color="#4c78a8", width=2.5),
                marker=dict(size=5)
            ))
            figA.update_layout(
                xaxis_title="Temps",
                yaxis_title=label_y,
                height=380,
                margin=dict(l=0, r=0, t=40, b=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(figA, use_container_width=True)
        with cB:
            st.markdown(f"**Période B : {yB0}–{yB1}**")
            figB = go.Figure()
            figB.add_trace(go.Scatter(
                x=serieB["x"], y=serieB["sma"],
                mode="lines+markers", name="B",
                line=dict(color="#f58518", width=2.5),
                marker=dict(size=5)
            ))
            figB.update_layout(
                xaxis_title="Temps",
                yaxis_title=label_y,
                height=380,
                margin=dict(l=0, r=0, t=40, b=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(figB, use_container_width=True)

    st.caption(f"Δ (B − A) = **{len(B) - len(A):+,}** accidents".replace(",", " "))

st.markdown("---")

# ========= Pics temporels =========
st.subheader("⛳ Pics temporels")
cL, cR = st.columns(2)
with cL:
    st.markdown("**Top heures**")
    top_h = (
        df_f.groupby("heure_num").size()
        .rename("accidents").reset_index()
        .sort_values("accidents", ascending=False).head(10)
    )
    st.dataframe(top_h, use_container_width=True)
with cR:
    st.markdown("**Top jours (jour de semaine)**")
    top_d = (
        df_f.groupby("jour_sem").size()
        .rename("accidents").reset_index()
        .sort_values("accidents", ascending=False)
    )
    st.dataframe(top_d, use_container_width=True)

# ========= Exports =========
st.markdown("---")
cE1, cE2 = st.columns(2)
cE1.download_button(
    "💾 Export – données filtrées (CSV)",
    data=df_f.to_csv(index=False).encode("utf-8"),
    file_name="acc_temporel_filtre.csv",
    mime="text/csv",
    use_container_width=True
)
subset_cols = [c for c in ["date", "annee", "mois_str", "jour_sem", "heure_num"] if c in df_f.columns]
if subset_cols:
    cE2.download_button(
        "📊 Export – sous-ensemble temporel (CSV)",
        data=df_f[subset_cols].to_csv(index=False).encode("utf-8"),
        file_name="acc_temporel_subset.csv",
        mime="text/csv",
        use_container_width=True
    )

# ========= Notes =========
with st.expander("🧪 Notes méthodologiques"):
    st.markdown(
        "- **Granularité** : agrégation par jour, semaine ISO (approx) ou mois.\n"
        "- **Lissage** : moyenne mobile appliquée à la série agrégée.\n"
        "- **Comparaison** : courbes lissées par période (même granularité).\n"
        "- **Pics temporels** : tableaux pour identifier les heures et jours les plus accidentogènes."
    )





