import pandas as pd
import matplotlib.pyplot as plt
# ---------------------------------------------------------
# 1) Charger ou réutiliser le DataFrame final
# ---------------------------------------------------------
df_final = pd.read_csv('dataset_final.csv')

# ---------------------------------------------------------
# 2) Agrégation Crimes par région + année
# ---------------------------------------------------------

agg = (
    df_final
      .groupby(["annee", "region"], as_index=False)["NombreCrimes"]
      .sum()
)

# 3) Pour chaque année, récupérer le top 5
top5_by_year = (
    agg
      .sort_values(["annee","NombreCrimes"], ascending=[True, False])
      .groupby("annee")
      .head(5)
)

# 4) Pivot pour avoir en colonnes chaque année
pivot = top5_by_year.pivot(index="region", columns="annee", values="NombreCrimes").fillna(0)

# 5) Graphique groupé
ax = pivot.plot(
    kind="bar",
    figsize=(10, 6),
    width=0.8
)
ax.set_title("Top 5 régions par nombre de Crimes (2017 vs 2022)")
ax.set_ylabel("Nombre de Crimes")
ax.set_xlabel("Région")
plt.xticks(rotation=45, ha="right")
plt.legend(title="Année")
plt.tight_layout()
plt.show()
#-------------------------visualisation sur toute la france par année
# 1) Somme des NombreCrimes par année
agg = df_final.groupby('annee', as_index=False)['NombreCrimes'].sum()

years  = agg['annee'].astype(str)
values = agg['NombreCrimes']

# 4) Plot
fig, ax = plt.subplots(figsize=(8,5))
bars = ax.bar(years, values, color=['#4C72B0', '#55A868'])

# 5) Data‐labels : afficher la valeur exacte au‐dessus de chaque barre
for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + values.max()*0.01,
            f"{round(h):,}", ha='center', va='bottom')

# 6) Annotation de la différence
diff = round(values.iloc[1] - values.iloc[0])
pct  = diff / round(values.iloc[0]) * 100
ax.annotate(f"Δ = {diff:,} ({pct:.1f} %)",
            xy=(1, round(values.iloc[1])),
            xytext=(0.5, values.max()*1.08),
            ha='center',
            arrowprops=dict(arrowstyle='->', lw=1.5))

# 7) Titres et style
ax.set_title("Comparaison du nombre de Crimes\n2017 vs 2022", pad=15)
ax.set_xlabel("Année")
ax.set_ylabel("Nombre de Crimes")
plt.ylim(0, values.max()*1.15)
plt.tight_layout()
plt.show()
# visualisation pour la bretagne
df_bret = df_final[df_final["region"] == "Bretagne"].reset_index(drop=True)
agg = df_bret.groupby('annee', as_index=False)['NombreCrimes'].sum()

years  = agg['annee'].astype(str)
values = agg['NombreCrimes']

# 4) Plot
fig, ax = plt.subplots(figsize=(8,5))
bars = ax.bar(years, values, color=['#4C72B0', '#55A868'])

# 5) Data‐labels : afficher la valeur exacte au‐dessus de chaque barre
for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + values.max()*0.01,
            f"{round(h):,}", ha='center', va='bottom')

# 6) Annotation de la différence
diff = round(values.iloc[1] - values.iloc[0])
pct  = diff / round(values.iloc[0]) * 100
ax.annotate(f"Δ = {diff:,} ({pct:.1f} %)",
            xy=(1, round(values.iloc[1])),
            xytext=(0.5, values.max()*1.08),
            ha='center',
            arrowprops=dict(arrowstyle='->', lw=1.5))

# 7) Titres et style
ax.set_title("Comparaison du nombre de Crimes En Bretagne\n2017 vs 2022", pad=15)
ax.set_xlabel("Année")
ax.set_ylabel("Nombre de Crimes")
plt.ylim(0, values.max()*1.15)
plt.tight_layout()
plt.show()