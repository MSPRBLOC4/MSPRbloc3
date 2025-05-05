import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

# Chargement des données
df = pd.read_csv("ensemble_donnees_economique.csv", sep=",", low_memory=False)
df.columns = df.columns.str.strip()

# Colonnes numériques
colonnes_numeriques = [
    'chomeurs15_64ans', 'nombres_artisant', 'nombre_de_salarie', 'chomeurs15_24ans',
    'Logement', 'salaire mediane', 'logement fiscaux',
    'nombre de personne dans les logement fiscaux', 'POP', 'POP0014', 'POP1529',
    'POP3044', 'POP4559', 'POP6074', 'POP7589', 'POP90P', 'NSCOL15P', 'DIPLMIN',
    'BEPC', 'CAPBEP', 'BAC', 'SUP2', 'SUP34', 'SUP5', 'NombreCrimes', 'NB'
]

# Conversion en float
for col in colonnes_numeriques:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Suppression des lignes incomplètes
df = df.dropna()

# Encodage de la variable cible
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["meilleur_candidat"].astype(str))

# Colonnes catégorielles autorisées
colonnes_categorielle_utiles = ['elu_municipal', 'orientation_municipal', 'annee']

# Variables explicatives
X = df[colonnes_numeriques + colonnes_categorielle_utiles].copy()

# Encodage des colonnes catégorielles
label_encoders = {}
for col in colonnes_categorielle_utiles:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# PCA
n_composantes = min(len(X), len(X.columns)) - 1
pca = PCA(n_components=n_composantes)
X_reduced = pca.fit_transform(X)

# Séparation des données Bretagne / hors Bretagne
df_non_bretagne = df[df["region"] != "Bretagne"].copy()
df_bretagne = df[df["region"] == "Bretagne"].copy()

X_non_bretagne = df_non_bretagne[colonnes_numeriques + colonnes_categorielle_utiles].copy()
y_non_bretagne = y[df["region"] != "Bretagne"]

for col in colonnes_categorielle_utiles:
    le = label_encoders[col]
    X_non_bretagne[col] = le.transform(X_non_bretagne[col].astype(str))

X_non_bretagne_reduced = pca.transform(X_non_bretagne)

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_non_bretagne_reduced, y_non_bretagne, test_size=0.2, random_state=42
)

# Modèle Random Forest
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# Évaluation hors Bretagne
y_pred = model.predict(X_test)
print("=== Rapport de classification (Hors Bretagne) ===")
print(classification_report(y_test, y_pred))

cm_test = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_test, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title("Matrice de confusion - Test Hors Bretagne")
plt.xlabel("Prédictions")
plt.ylabel("Réel")
plt.show()

# Données Bretagne
X_bretagne = df_bretagne[colonnes_numeriques + colonnes_categorielle_utiles].copy()
for col in colonnes_categorielle_utiles:
    le = label_encoders[col]
    X_bretagne[col] = le.transform(X_bretagne[col].astype(str))

X_bretagne_reduced = pca.transform(X_bretagne)

# Prédictions Bretagne
y_bretagne_pred = model.predict(X_bretagne_reduced)
y_bretagne_true = label_encoder.transform(df_bretagne["meilleur_candidat"].astype(str))

# Accuracy Bretagne
accuracy_bretagne = accuracy_score(y_bretagne_true, y_bretagne_pred)
print(f"\n=== Accuracy sur les données de Bretagne : {accuracy_bretagne:.4f} ===")

# Rapport de classification Bretagne
print("\n=== Rapport de classification (Bretagne) ===")
print(classification_report(y_bretagne_true, y_bretagne_pred))

# Matrice de confusion Bretagne
cm_bretagne = confusion_matrix(y_bretagne_true, y_bretagne_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_bretagne, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title("Matrice de confusion - Bretagne")
plt.xlabel("Prédictions")
plt.ylabel("Réel")
plt.show()

# Résultats Bretagne
df_bretagne["prédictions"] = label_encoder.inverse_transform(y_bretagne_pred)
df_bretagne["réel"] = label_encoder.inverse_transform(y_bretagne_true)

print("\n=== Prédictions par commune (Bretagne) ===")
print(df_bretagne[["com", "prédictions", "réel"]].head())

# Top 5 candidats prédits
print("\n=== Candidats les plus prédits en Bretagne ===")
print(df_bretagne["prédictions"].value_counts().head())

# Comparaison réel vs prédictions
print("\n=== Table croisée Réel vs Prédictions (Bretagne) ===")
comparaison = df_bretagne.groupby(["réel", "prédictions"]).size().unstack().fillna(0)
print(comparaison)

# Importance des features (via PCA)
importances = model.feature_importances_
features = [f"PC{i+1}" for i in range(X_train.shape[1])]
importance_df = pd.DataFrame({'feature': features, 'importance': importances})
importance_df = importance_df.sort_values(by='importance', ascending=False)

print("\n=== Composantes principales les plus importantes ===")
print(importance_df.head(10))

plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df.head(10), x='importance', y='feature', palette="viridis")
plt.title("Top 10 des composantes principales influentes")
plt.xlabel("Importance")
plt.ylabel("Composante principale (PCA)")
plt.tight_layout()
plt.show()

# Sauvegarde CSV
df_bretagne[["com", "prédictions"]].to_csv("resultats_bretagne.csv", index=False)
print("\nRésultats sauvegardés dans 'resultats_bretagne.csv'")
