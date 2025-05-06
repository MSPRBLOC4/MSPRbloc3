import pandas as pd

# Lire le fichier complet
fichier = "data_election_presidentiel/Presidentielle_2017_Resultats_Communes_Tour_1.xls"
orientation_df = pd.read_excel("data_election_presidentiel/parti condidat 2017.xlsx")
df = pd.read_excel(fichier)

# Fonction pour extraire le candidat ayant le plus de voix
def meilleur_score(row):
    max_voix = -1
    meilleur_candidat = None
    for i in range(17):
        start_col = 19 + i * 7
        end_col = start_col + 7
        if end_col > len(row) - 1:
            break
        bloc = row.iloc[start_col:end_col]
        nom_candidat = bloc.iloc[1].strip().title()
        if pd.isna(nom_candidat):
            break
        #code_nuance = bloc.iloc[1]
        prenom_candidat = bloc.iloc[2].strip().title()
        try:
            voix = float(bloc.iloc[5])
        except:
            voix = 0.0
        if voix > max_voix:
            max_voix = voix
            elu_nom_prenom = f"{prenom_candidat} {nom_candidat}"

    return pd.Series({
        "code_dept": row["Code du département"],
        "dept": row["Libellé du département"],
        "code_com": row["Code de la commune"],
        "com": row["Libellé de la commune"],
        "inscrits": row["Inscrits"],
        "meilleur_candidat": elu_nom_prenom,
        "voix": max_voix
    })

# Appliquer la fonction
df_meilleurs = df.apply(meilleur_score, axis=1)
def merge_dataframe(df_winners):
    df_winners["meilleur_candidat_lower"] = df_winners["meilleur_candidat"].str.lower()
    orientation_df["Candidat_lower"] = orientation_df["Candidat"].str.lower()

    df_final = df_winners.merge(
        orientation_df,
        left_on="meilleur_candidat_lower",
        right_on="Candidat_lower",
        how="left"
    )
    df_final.drop(columns=["meilleur_candidat_lower", "Candidat_lower"], inplace=True)
    return df_final
# Afficher un extrait
print(df_meilleurs.head())
print(merge_dataframe(df_meilleurs).head())
df_final = merge_dataframe(df_meilleurs)
df_final.to_csv("presidentielle_2017_final.csv", index=False, encoding="utf-8")

