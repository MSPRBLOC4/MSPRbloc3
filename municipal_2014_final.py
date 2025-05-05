#!/usr/bin/env python
# -*- coding: utf-8 -*-


import urllib.request
from pathlib import Path
import pandas as pd
import pyarrow.dataset as ds

DATA_DIR       = Path("data_muni2014")
DATA_DIR.mkdir(exist_ok=True)

PARQUET_CANDID = DATA_DIR / "candidats_results.parquet"
PARQUET_GEN    = DATA_DIR / "general_results.parquet"
NUANCES_FILE   = DATA_DIR / "nuances.csv"

ORIENT_EXCEL   = Path(r"C:/Users/HP/ELECTION/MSPR/data_election_municipale/oriontation nuance.xlsx")

OUT_CSV        = "municipales_2014_complete.csv"


URL_CANDID  = ("https://object.files.data.gouv.fr/data-pipeline-open/prod/"
               "elections/candidats_results.parquet")
URL_GENERAL = ("https://object.files.data.gouv.fr/data-pipeline-open/prod/"
               "elections/general_results.parquet")
URL_NUANCES = ("https://static.data.gouv.fr/resources/"
               "donnees-des-elections-agregees/20240425-102414/nuances.csv")

def download(url: str, dest: Path):
    if not dest.exists():
        print(f"Téléchargement {dest.name} …")
        urllib.request.urlretrieve(url, dest)
        print("   → OK")
    else:
        print(f"✓ {dest.name} déjà présent")

download(URL_CANDID,  PARQUET_CANDID)
download(URL_GENERAL, PARQUET_GEN)
download(URL_NUANCES, NUANCES_FILE)

print("Lecture des résultats candidats…")
RAW_COLS_CANDID = ["Code de la commune", "Code du département",
                   "Nom", "Prénom", "Nuance", "Voix", "id_election"]

ds_candid = ds.dataset(PARQUET_CANDID, format="parquet")
filter_muni = (ds.field("id_election") == "2014_muni_t2") | \
              (ds.field("id_election") == "2014_muni_t1")

cand_df = (
    ds_candid
      .to_table(filter=filter_muni, columns=RAW_COLS_CANDID)
      .to_pandas()
      .rename(columns={
          "Code de la commune":  "code_com",
          "Code du département": "code_dept",
          "Nom":    "nom",
          "Prénom": "prenom",
          "Nuance": "code_nuance_2014",
          "Voix":   "voix_gagnant"
      })
      .astype({"code_com": "string",
               "code_dept":    "string",
               "code_nuance_2014": "string"})
)


cand_df = cand_df.sort_values(["code_com", "id_election", "voix_gagnant"],
                              ascending=[True, True, False])

vainqueurs = (
    cand_df
      .groupby(["code_dept", "code_com"], as_index=False)
      .first()
)

vainqueurs["elu_muni2014"] = (
    vainqueurs["nom"].str.title() + " " + vainqueurs["prenom"].str.title()
)


print("Lecture des résultats généraux…")
RAW_COLS_GEN = ["Code du département", "Code de la commune",
                "Inscrits", "Votants", "id_election"]

ds_gen = ds.dataset(PARQUET_GEN, format="parquet")
gen_df = (
    ds_gen
      .to_table(filter=filter_muni, columns=RAW_COLS_GEN)
      .to_pandas()
      .rename(columns={
          "Code de la commune":  "code_com",
          "Code du département": "code_dept",
      })
      .astype({"Inscrits": "Int64", "Votants": "Int64"})
      .sort_values(["code_com", "id_election"], ascending=[True, True])
      .groupby(["code_dept", "code_com"], as_index=False)
      .first()
      .rename(columns={"Inscrits": "inscrits_muni2014",
                       "Votants":  "votants_muni2014"})
)

vainqueurs = vainqueurs.merge(gen_df,
                              on=["code_dept", "code_com"],
                              how="left")


print(" Ajout de l’orientation politique…")
orientation_df = (
    pd.read_excel(ORIENT_EXCEL, dtype=str)
      .rename(columns={"code_nuance": "code_nuance_2014",
                       "orientation": "orientation_muni2014"})
)

vainqueurs = vainqueurs.merge(orientation_df,
                              on="code_nuance_2014",
                              how="left")


cols_finales = ["code_dept", "code_com",
                "elu_muni2014", "code_nuance_2014", "orientation_muni2014",
                "inscrits_muni2014", "votants_muni2014", "voix_gagnant"]

vainqueurs = vainqueurs[cols_finales]

vainqueurs.to_csv(OUT_CSV, index=False, encoding="utf-8")
print(f" {OUT_CSV} créé ({len(vainqueurs)} communes)")
