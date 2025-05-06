#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import unicodedata
from pathlib import Path
from sklearn.impute import SimpleImputer
def clean(col):
    import unicodedata, re
    col = unicodedata.normalize("NFKD", col).encode("ascii", "ignore").decode()
    col = re.sub(r"\s+", "_", col.strip().lower())   # espaces → _
    return col
# -------------------------------------------------------------------
# 1) Fichiers sources
# -------------------------------------------------------------------
F1      = "donnees_economique/données eco 2017.xlsx"
F2      = "donnees_economique/données eco 2021.xlsx"
F3      ='structure_population_2017_tranches_age.csv'
F4      ='structure_population_2021_tranches_age.csv'
F5      ='niveau_diplome_non_scolarise_2017.csv'
F6      ='niveau_diplome_non_scolarise_2021.csv'
F7      ='Population___Crimes__Format_Long_ 2.csv'
F8      ='immigration.csv'
ELECTORAL = "ensemble_elections_socio3.csv"
OUT     = "dataset_final.csv"
OUT_model     = "model_to_train.csv"

# -------------------------------------------------------------------
# 2) Lecture Excel avec type explicite
# -------------------------------------------------------------------
dtypes_excel = {"CODGEO": "string"}          # seule colonne clé
dtypes_elec = {c: "string" for c in pd.read_csv(ELECTORAL, nrows=0).columns}
df1 = pd.read_excel(F1, dtype=dtypes_excel)
df2 = pd.read_excel(F2, dtype=dtypes_excel)
df3 = pd.read_csv(F3, sep=";", dtype=str)
df4 = pd.read_csv(F4, sep=";", dtype=str)
df5 = pd.read_csv(F5, sep=";", dtype=str)
df6 = pd.read_csv(F6, sep=";", dtype=str)
df7 = pd.read_csv(F7, sep=";", dtype=str)
df8 = pd.read_csv(F8, sep=";", dtype=str)
# Nettoie les intitulés

print("df3 :", df5.columns.tolist())
pop17 = df3.merge(df5, on="codecommune", how="left")
pop17["annee"] = "2017"

# 2021
pop21 = df4.merge(df6, on="codecommune", how="left")
pop21["annee"] = "2022"


df_concat = pd.concat([df1, df2], ignore_index=True)
df_pop_etude = pd.concat([pop17,pop21], ignore_index=True)

# -------------------------------------------------------------------
# 3) Conversion des colonnes numériques
# -------------------------------------------------------------------
num_cols = ["Logement", "salaire mediane",
            "logement fiscaux", "nombre de personne dans les logement fiscaux"]

df_concat[num_cols] = df_concat[num_cols].apply(
    pd.to_numeric, errors="coerce"
)

num_imp = SimpleImputer(strategy="median")
df_concat[num_cols] = num_imp.fit_transform(df_concat[num_cols])

# -------------------------------------------------------------------
# 4) Renommer CODGEO -> codgeo (clé commune)
# -------------------------------------------------------------------
df_concat = df_concat.rename(columns={"CODGEO": "codgeo"})
df_pop_etude = df_pop_etude.rename(columns={"codecommune": "codgeo"})
df_concat["annee"] = df_concat["annee"].astype("string")
# -------------------------------------------------------------------
# 5) Lecture des données électorales (toutes colonnes texte)
#    -> évite DtypeWarning sur colonnes mixtes
# -------------------------------------------------------------------
dtypes_elec = {c: "string" for c in pd.read_csv(ELECTORAL, nrows=0).columns}
vainqueurs = pd.read_csv(ELECTORAL, dtype=dtypes_elec, low_memory=False)

# -------------------------------------------------------------------
# 6) Fusion sur codgeo + annee
# -------------------------------------------------------------------
vainqueurs["annee"] = vainqueurs["annee"].astype("string")
final = vainqueurs.merge(df_concat, on=["codgeo", "annee"], how="left")

# -------------------------------------------------------------------
# 7) Export
# -------------------------------------------------------------------
med_cols = ["Logement", "logement fiscaux", "salaire mediane"]
final[med_cols] = (
    final
      .groupby("annee")[med_cols]                     # par année
      .transform(lambda s: s.fillna(s.median()))      # médiane du groupe
)

# ---- 3. MOYENNE pour la densité par logement fiscal -------------
mean_cols = ["nombre de personne dans les logement fiscaux"]
final[mean_cols] = (
    final
      .groupby("annee")[mean_cols]
      .transform(lambda s: s.fillna(s.mean()))
)
df_pop_etude = df_pop_etude.drop(columns="unnamed:_9", errors="ignore")

final["annee"] = final["annee"].astype("string")
df_pop_etude["annee"] = df_pop_etude["annee"].astype("string")
df_pop_etude["codgeo"] = df_pop_etude["codgeo"].astype("string")
final["codgeo"] = final["codgeo"].astype("string")
dipl_cols = ["DIPLMIN", "NSCOL15P", "BEPC", "CAPBEP",
             "SUP34", "SUP2", "BAC", "SUP5","POP0014","POP","POP3044","POP4559","POP6074","POP90P","POP1529","POP7589","NB","NombreCrimes"]


df_pop_etude = df_pop_etude.drop(columns="unnamed:_9", errors="ignore")

final = final.merge(df_pop_etude, on=["codgeo", "annee"], how="left")
final = final.merge(df7,on=['codgeo','annee'], how="left")
final = final.merge(df8,on=['codgeo','annee'], how="left")
final[dipl_cols] = final[dipl_cols].apply(
    pd.to_numeric, errors="coerce"
)
final[dipl_cols] = (
    final
      .groupby("annee")[dipl_cols]
      .transform(lambda s: s.fillna(s.mean()))
)

report = (final.isna()
            .agg(['sum', 'mean'])      # deux lignes : somme & moyenne
            .T                          # transpose pour lire en colonnes
            .rename(columns={'sum':'nb_nan', 'mean':'pct_nan'}))
report['pct_nan'] = (report['pct_nan']*100).round(1)

print(report.sort_values('nb_nan', ascending=False))
final = final.drop(columns="candidat", errors="ignore")
model_to_train = final.drop(columns=['voix','votants_municipal','voix_gagnant_municipal','elu_municipal','parti_politique','positionnement_politique'])
final.to_csv(OUT, index=False, encoding="utf-8")
model_to_predict = model_to_train[model_to_train["region"] == "Bretagne"].reset_index(drop=True)
model_to_train   = model_to_train[model_to_train["region"] != "Bretagne"].reset_index(drop=True)
OUT_predict = 'model_to_predict.csv'
model_to_train.to_csv(OUT_model, index=False, encoding="utf-8")
model_to_predict.to_csv(OUT_predict, index=False, encoding="utf-8")
print(f"✅ {OUT_model} créé ({len(model_to_train)} lignes, {len(model_to_train.columns)} colonnes)")
print(f"✅ {OUT} créé ({len(final)} lignes, {len(final.columns)} colonnes)")

