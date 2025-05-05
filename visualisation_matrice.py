import pandas as pd
import matplotlib.pyplot as plt

# 1. Charger les données
df = pd.read_csv("model_to_train.csv", low_memory=False)

# 2. Sélectionner les colonnes numériques
num_cols = df.select_dtypes(include=["int64", "float64"]).columns

# 3. Calculer la matrice de corrélation
corr = df[num_cols].corr()

# 4. Afficher la matrice dans la console
print("Matrice de corrélation :")
print(corr)

# 5. Visualiser la matrice sous forme de heatmap
plt.figure(figsize=(12, 10))
plt.imshow(corr, aspect='auto')
plt.colorbar(label='Coefficient de corrélation')
plt.xticks(range(len(num_cols)), num_cols, rotation=90)
plt.yticks(range(len(num_cols)), num_cols)
plt.title("Matrice de corrélation des variables numériques")
plt.tight_layout()
plt.show()
