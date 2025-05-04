import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA

# 📥 Chargement propre du fichier CSV
df = pd.read_csv("ensemble_donnees_economique.csv", sep=",", low_memory=False)
df.columns = df.columns.str.strip()  # Nettoie les noms de colonnes

# 🔍 Convertir les colonnes numériques
colonnes_numeriques = [
    'chomeurs15_64ans', 'nombres_artisant', 'nombre_de_salarie', 'chomeurs15_24ans',
    'Logement', 'salaire mediane', 'logement fiscaux',
    'nombre de personne dans les logement fiscaux', 'POP', 'POP0014', 'POP1529',
    'POP3044', 'POP4559', 'POP6074', 'POP7589', 'POP90P', 'NSCOL15P', 'DIPLMIN',
    'BEPC', 'CAPBEP', 'BAC', 'SUP2', 'SUP34', 'SUP5', 'NombreCrimes', 'NB'
]

# Conversion des colonnes numériques en type float
for col in colonnes_numeriques:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 🧹 Supprimer les lignes avec des valeurs manquantes
df = df.dropna()

# 🎯 Cible : meilleur_candidat
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["meilleur_candidat"].astype(str))

# 🧠 Sélection des features pertinentes
colonnes_a_encoder = [
    'parti_politique', 'positionnement_politique', 'elu_municipal',
    'orientation_municipal', 'annee'
]

# 🧪 Variables explicatives (X)
X = df[colonnes_numeriques + colonnes_a_encoder]

# 🔄 Encodage des colonnes catégorielles avec LabelEncoder
label_encoders = {}
for col in colonnes_a_encoder:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))  # Applique LabelEncoder à chaque colonne
    label_encoders[col] = le  # Stocke le LabelEncoder pour chaque colonne si nécessaire pour l'inverse

# 🎯 Réduction de la dimensionnalité avec PCA
pca = PCA(n_components=30)  # Réduire à 30 composants
X_reduced = pca.fit_transform(X)

# 🧪 Séparation apprentissage/test
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

# 🌳 Modèle Random Forest
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# 📊 Évaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# 🔮 Prédiction sur la région Bretagne
df_bretagne = df[df["region"] == "Bretagne"].copy()
X_bretagne = df_bretagne[colonnes_numeriques + colonnes_a_encoder]

# Encodage des colonnes catégorielles pour les données de la Bretagne
for col in colonnes_a_encoder:
    le = label_encoders[col]
    X_bretagne[col] = le.transform(X_bretagne[col].astype(str))  # Utilisation des encodeurs sauvegardés

# Réduction de la dimensionnalité des données de la Bretagne
X_bretagne_reduced = pca.transform(X_bretagne)

# Prédiction
y_bretagne_pred = model.predict(X_bretagne_reduced)

# Décodage des candidats prédits
predictions = label_encoder.inverse_transform(y_bretagne_pred)

# Affichage des résultats pour la Bretagne
df_bretagne['prédictions'] = predictions
print(df_bretagne[['com', 'prédictions']])  # Affiche les prédictions pour chaque commune en Bretagne
