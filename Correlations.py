import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Charger le dataset
df = pd.read_csv("ensemble_donnees_economique.csv", sep=",")

# Supprimer les colonnes non numériques ou identifier les utiles
cols_to_use = [
    'voix', 'chomeurs15_64ans', 'nombres_artisant', 'nombre_de_salarie',
    'chomeurs15_24ans', 'Logement', 'salaire mediane', 'logement fiscaux',
    'nombre de personne dans les logement fiscaux', 'POP', 'POP0014',
    'POP1529', 'POP3044', 'POP4559', 'POP6074', 'POP7589', 'POP90P',
    'NSCOL15P', 'DIPLMIN', 'BEPC', 'CAPBEP', 'BAC', 'SUP2', 'SUP34',
    'SUP5', 'NombreCrimes'
]

# Calcul de la matrice de corrélation
correlation_matrix = df[cols_to_use].corr()

# Affichage de la heatmap
plt.figure(figsize=(16, 12))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matrice de corrélation avec les voix")
plt.show()
