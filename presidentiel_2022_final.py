import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
df_pres22 = pd.read_csv("presidentielle_2022_final.csv", encoding="utf-8")
df_muni20 = pd.read_csv("municipales_2020_complete.csv",  encoding="utf-8")

for col in ["code_dept", "code_com"]:
    df_pres22[col] = df_pres22[col].astype("string").str.zfill(5 if col == "code_com" else 2)
    df_muni20[col] = df_muni20[col].astype("string").str.zfill(5 if col == "code_com" else 2)

df_muni14_reduit = (
    df_muni20[["code_dept", "code_com", "elu_muni2020", "code_nuance_2020","orientation_muni2020",
               "inscrits_muni2020","votants_muni2020","voix_gagnant"]]
      .rename(columns={
          "elu_muni2020":       "elu_municipal",
          "code_nuance_2020": "code_nuance_municipal",
          "orientation_muni2020":"orientation_municipal",
          "inscrits_muni2020":"inscrits_municipal",
          "votants_muni2020":"votants_municipal",
          "voix_gagnant":"voix_gagnant_municipal"
      })
)

df_final = (
    df_pres22
      .merge(df_muni14_reduit, on=["code_dept", "code_com"], how="left")
)

# ----------------------------------------------------
# 0) Nettoyage des en-têtes
# ----------------------------------------------------
df_final.columns = df_final.columns.str.strip()

# ----------------------------------------------------
# 1) Listes de colonnes
# ----------------------------------------------------
num_cols = [
    "inscrits", "voix",
    "inscrits_municipal", "votants_municipal", "voix_gagnant_municipal"
]
cat_cols = [
    "dept", "com", "meilleur_candidat", "Candidat",
    "Parti politique", "Positionnement politique",
    "code_nuance_municipal", "orientation_municipal"
]

# ----------------------------------------------------
# 2) Conversion numérique
# ----------------------------------------------------
df_final[num_cols] = df_final[num_cols].apply(
    pd.to_numeric, errors="coerce"
)

# ----------------------------------------------------
# 3) Mise à NaN des incohérences
#     ─ pour les 3 colonnes générales
#     ─ pour les 3 colonnes municipales (déjà faits)
# ----------------------------------------------------
# A) générales
mask_votants_gt_inscrits_gen = df_final["voix"] > df_final["inscrits"]


df_final.loc[mask_votants_gt_inscrits_gen, ["voix"]] = np.nan


# B) municipales (rappel)
mask_votants_gt_inscrits_mun = df_final["votants_municipal"] > df_final["inscrits_municipal"]
mask_voix_gt_votants_mun     = df_final["voix_gagnant_municipal"] > df_final["votants_municipal"]

df_final.loc[mask_votants_gt_inscrits_mun, ["votants_municipal"]]      = np.nan
df_final.loc[mask_voix_gt_votants_mun,     ["voix_gagnant_municipal"]] = np.nan

# ----------------------------------------------------
# 4) Suppression des lignes sans élu municipal
# ----------------------------------------------------
df_final = df_final.dropna(subset=["elu_municipal"]).reset_index(drop=True)

# ----------------------------------------------------
# 5) Imputation NaN (médiane / modalité la + fréquente)
# ----------------------------------------------------
num_imp = SimpleImputer(strategy="median")
cat_imp = SimpleImputer(strategy="most_frequent", fill_value="Inconnu")

df_final[num_cols] = num_imp.fit_transform(df_final[num_cols])
df_final[cat_cols] = cat_imp.fit_transform(df_final[cat_cols])
df_final["annee"] = 2022
df_final.to_csv("fusion_pres22_muni20.csv", index=False, encoding="utf-8")
print("Fusion réalisée : 'fusion_pres22_muni20.csv' créé avec",
      df_final.shape[0], "lignes et", df_final.shape[1], "colonnes")
