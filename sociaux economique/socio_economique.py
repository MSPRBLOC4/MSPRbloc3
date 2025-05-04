import pandas as pd
import matplotlib.pyplot as plt


immigrationFr2017=pd.read_csv("immigration_2017.CSV",sep=';', encoding='utf-8',low_memory=False)
immigrationFr2017 = immigrationFr2017[['CODGEO','NB']]
immigrationFr2017 = immigrationFr2017.drop_duplicates()
immigrationFr2017 = immigrationFr2017.dropna()
immigrationFr2017['ANNEE'] = 2017

immigrationFr_grouped2017 = immigrationFr2017.groupby('CODGEO')['NB'].mean().sort_values(ascending=False)

print(immigrationFr_grouped2017.count())

immigrationFr2022=pd.read_csv("immigration_2021.CSV",sep=';', encoding='utf-8',low_memory=False)

immigrationFr2022 = immigrationFr2022[['CODGEO','NB']].drop_duplicates()
immigrationFr2022 = immigrationFr2022.dropna()
immigrationFr2022['ANNEE'] = 2022

immigrationFr_grouped2022 = immigrationFr2022.groupby('CODGEO')['NB'].mean().sort_values(ascending=False)
print(immigrationFr_grouped2022.count())

immigrationFusion = pd.concat([immigrationFr2017, immigrationFr2022], ignore_index=True)

immigrationFusion_sorted = immigrationFusion.sort_values(by='CODGEO', ascending=True)


immigrationFusion_sorted = immigrationFusion_sorted.reset_index(drop=True)
immigrationFusion_sorted=immigrationFusion_sorted.drop_duplicates(subset=['CODGEO','ANNEE'])
immigrationFusion_sorted.to_csv("immigrationFusion_sorted.csv", index=False)

print(immigrationFusion_sorted.count())