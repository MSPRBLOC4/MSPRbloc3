#from pyspark.sql import SparkSession
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt

delinquenceFr=pd.read_csv("donnee-data.gouv-2023-geographie2024-produit-le2024-07-05.csv/donnee-data.gouv-2023-geographie2024-produit-le2024-07-05.csv",sep=';', encoding='utf-8',low_memory=False)

delinquenceFr = delinquenceFr.drop_duplicates()
delinquenceFr = delinquenceFr.dropna(subset=['tauxpourmille'])
delinquenceFr_grouped = delinquenceFr.drop(columns=['millPOP','complementinfoval', 'complementinfotaux','faits','unité.de.compte','valeur.publiée','tauxpourmille','LOG','millLOG'])

delinquenceFr_grouped5 = delinquenceFr.groupby('classe')['POP'].mean().sort_values(ascending=False)
delinquenceFr_grouped5=delinquenceFr_grouped5.head(5)
print(delinquenceFr_grouped5.head())

plt.figure(figsize=(12,6))
plt.bar(delinquenceFr_grouped5.index, delinquenceFr_grouped5.values, color='b')
plt.xticks(rotation=15)
plt.xlabel('depart')
plt.ylabel('taux pour mille')
plt.show()


delinquenceFr_grouped_dep = delinquenceFr.groupby('CODGEO_2024')['POP'].mean().sort_values(ascending=False)
delinquenceFr_grouped_dep5 = delinquenceFr_grouped_dep.head(5)
plt.figure(figsize=(12,12))
plt.bar(delinquenceFr_grouped_dep5.index, delinquenceFr_grouped_dep5.values, color='b')
plt.xticks(rotation=45)
plt.xlabel('classe')
plt.ylabel('taux pour mille')
plt.show()
####grouper les classes du dataframe

delinquenceFr_grouped_classe=delinquenceFr_grouped.groupby(['CODGEO_2024','annee'])['POP'].sum().reset_index()
delinquenceFr_grouped_electorale=delinquenceFr_grouped_classe[delinquenceFr_grouped_classe['annee'].isin([16,21])]
print(delinquenceFr_grouped_electorale)
delinquenceFr_grouped_classe.to_csv("test2.csv",index=False)
print("succes")
"""
geolocalisation_gendarmerie=pd.read_csv("export-gn2 ouverture gendermerie.csv",sep=";", encoding="utf-8")
geolocalisation_gendarmerie_subset=geolocalisation_gendarmerie[["identifiant_public_unite","departement","geocodage_x_GPS","geocodage_y_GPS"]]
geolocalisation_gendarmerie_subset.drop_duplicates()

geolocalisation_police=pd.read_csv("export-pn ouverture police.csv",sep=";", encoding="utf-8")
geolocalisation_police_subset=geolocalisation_police[["service","departement","geocodage_x_GPS","geocodage_y_GPS"]]
geolocalisation_police_subset.drop_duplicates()
# Utiliser un fond de carte "terrain"
satellite_map = cimgt.GoogleTiles(style='satellite')

# --- Carte Police ---
fig1 = plt.figure(figsize=(10, 8))
ax1 = plt.axes(projection=satellite_map.crs)
ax1.set_extent([-5, 10, 41, 52], crs=ccrs.PlateCarree())
ax1.add_image(satellite_map, 8)

for _, row in geolocalisation_police_subset.iterrows():
    lat = row['geocodage_y_GPS']
    lon = row['geocodage_x_GPS']
    ax1.plot(lon, lat, marker='o', color='red', markersize=5, transform=ccrs.PlateCarree())

ax1.set_title("Carte des commissariats de police")

# --- Carte Gendarmerie ---
fig2 = plt.figure(figsize=(10, 8))
ax2 = plt.axes(projection=satellite_map.crs)
ax2.set_extent([-5, 10, 41, 52], crs=ccrs.PlateCarree())
ax2.add_image(satellite_map, 8)

for _, row in geolocalisation_gendarmerie_subset.iterrows():
    lat = row['geocodage_y_GPS']
    lon = row['geocodage_x_GPS']
    ax2.plot(lon, lat, marker='o', color='blue', markersize=5, transform=ccrs.PlateCarree())

ax2.set_title("Carte des brigades de gendarmerie")

# Afficher les deux cartes
plt.show()"""




"""""
delinquance=pd.read_csv("bases statistiques regionale de la delinquance enregistree par la police et la gendarmerie nationales.csv",sep=";", encoding="utf-8")
print(delinquance.dtypes)
print(delinquance.count())
"""


""""
import os

# Chemin du dossier contenant les fichiers CSV
dossier_csv = "chemin/vers/ton/dossier"

# Liste pour stocker tous les DataFrames
tous_les_csv = []

# Parcourir tous les fichiers du dossier
for fichier in os.listdir(dossier_csv):
    if fichier.endswith(".csv"):
       chemin_fichier = os.path.join(dossier_csv, fichier)
        try:
            # Lire le CSV et l'ajouter à la liste
            df = pd.read_csv(chemin_fichier, sep=';', encoding='utf-8', error_bad_lines=False)
            tous_les_csv.append(df)
        except Exception as e:
            print(f"Problème avec le fichier {fichier} : {e}")

# Concaténer tous les DataFrames en un seul
#df_final = pd.concat(tous_les_csv, ignore_index=True)

# Afficher les premières lignes pour vérifier
#print(df_final.head())
"""