import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# 1) Charger les données
df = pd.read_csv("dataset_final.csv", low_memory=False)

# 2) Préparer la figure
plt.figure(figsize=(8, 6))

# 3) Pour chaque année, tracer scatter + droite de régression
for year, color in zip([2017, 2022], ["C0", "C1"]):
    sub = df[df["annee"] == year]
    # X : chômage, y : crimes
    X = sub["chomeurs15_64ans"].astype(float).values.reshape(-1, 1)
    y = sub["NombreCrimes"].astype(float).values

    # ajustement
    model = LinearRegression().fit(X, y)

    # nuage de points
    plt.scatter(X, y, alpha=0.5, label=f"{year}", color=color)

    # droite de régression
    x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_line = model.predict(x_line)
    plt.plot(x_line, y_line, color=color, linewidth=2,
             label=f"Trend {year}")

# 4) Mise en forme
plt.xlabel("Taux de chômage (15–64 ans)")
plt.ylabel("Nombre de crimes")
plt.title("Chômage vs Crime par année")
plt.legend()
plt.tight_layout()
plt.show()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Charger les données
df = pd.read_csv("dataset_final.csv", low_memory=False)

# Calculer le taux de chômage (%)
df['taux_chomage'] = df['chomeurs15_64ans'] / df['POP'] * 100

# Tracer les boxplots par candidat
plt.figure(figsize=(14, 6))
sns.boxplot(
    x='meilleur_candidat',
    y='taux_chomage',
    data=df,
    showfliers=False
)
plt.xticks(rotation=45, ha='right')
plt.xlabel("Candidat")
plt.ylabel("Taux de chômage (%)")
plt.title("Distribution du taux de chômage par candidat")
plt.tight_layout()
plt.show()
# compare les contextes socio-économiques des territoires où chaque candidat est fort,
# démontrant la pertinence des variables chômage et population.