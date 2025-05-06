#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt

# 1) Chargement du jeu de données final
df = pd.read_csv("dataset_final.csv", low_memory=False)

# 2) Filtrer pour l'année 2022 et ne garder que les communes gagnées
df22 = df[df["annee"] == 2022].copy()
# 3) Agréger le nombre de communes remportées par candidat et par région
agg = (
    df22
      .groupby(["region", "meilleur_candidat"], as_index=False)
      .size()
      .rename(columns={"size": "n_communes"})
)

# 4) Déterminer les 5 candidats les plus “présents” (globalement) et tagguer les autres
top5 = (
    agg
      .groupby("meilleur_candidat")["n_communes"]
      .sum()
      .sort_values(ascending=False)
      .head(5)
      .index
      .tolist()
)
agg["candidat_cat"] = agg["meilleur_candidat"].where(
    agg["meilleur_candidat"].isin(top5),
    other="Autres"
)

# 5) Re-agréger en prenant en compte le regroupement “Autres”
agg2 = (
    agg
      .groupby(["region", "candidat_cat"], as_index=False)
      .agg({"n_communes": "sum"})
)

# 6) Pivot pour avoir Regions en index, candidats en colonnes
pivot = agg2.pivot(index="region",
                   columns="candidat_cat",
                   values="n_communes"
                  ).fillna(0)

# 7) Tracer le bar chart empilé
plt.figure(figsize=(12, 8))
pivot.plot(kind="bar", stacked=True, width=0.8, ax=plt.gca())
plt.title("Répartition des communes gagnées par région (Présidentielle 2022)")
plt.xlabel("Région")
plt.ylabel("Nombre de communes remportées")
plt.xticks(rotation=45, ha="right")
plt.legend(title="Candidat", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()
