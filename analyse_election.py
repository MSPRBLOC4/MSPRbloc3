from logging import exception

import pandas as pd

def clean_df(df):
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
    df = df.dropna(how='all')
    return df
def clean_excel(file_path):
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        print(f"Erreur lors du chargement de {file_path} : {e}")
        return None

    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
    df = df.dropna(how='all').drop_duplicates()
    return df
def clean_csv(file_path):
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Erreur lors du chargement de {file_path} : {e}")
        return None
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
    df = df.dropna(how='all').drop_duplicates()
    return df
def merge_df(data_frame1,data_frame2,key):
    try :
        df_merge = pd.merge(data_frame1,data_frame2,on=key,how="left")
    except Exception as e:
        print(f'Erruer lors du merge des dataframe {data_frame2} and {data_frame2} avec la clé {key}')
        return None
    return df_merge
def get_map(df_index,df_to_map,key,key_to_add):
    try:
        mapping_dept_region = pd.Series(df_index.région.values, index=df_index.code_du_département).to_dict()
        df_to_map[key_to_add] = df_to_map[key].map(mapping_dept_region)
    except Exception as e:
        print('there is a probleme')
    return df_to_map
file_departement_region = "departement par region.xlsx"
file_elec_regional = "reg-resultats-par-region-1-.csv"
file_elec_pr_niveau_dpt = "resultats-par-niveau-dpt-t1-france-entiere.xlsx"
file_elus_sen_dpt = "2020-09-28-resultats-avec-elus.xlsx"

df_departement_region = clean_excel(file_departement_region)
df_elec_regional = clean_csv(file_elec_regional)
df_elec_pr_niveau_dpt = clean_excel(file_elec_pr_niveau_dpt)
df_elus_sen_dpt = clean_excel(file_elus_sen_dpt)
print(df_departement_region.columns)
print(df_elec_pr_niveau_dpt.columns)
df_elec_pr_merged = get_map(df_departement_region,df_elec_pr_niveau_dpt,'code_du_département','région')
df_elec_pr_grouped = df_elec_pr_merged.groupby('région').sum().reset_index()
df_list = []
# sheets_dict = pd.read_excel(file_elus_sen_dpt, sheet_name=None)
# for sheet_name, df in sheets_dict.items():
#     print(f"Nettoyage de la sheet : {sheet_name}")
#     df_clean = clean_df(df)
#     df_list.append(df_clean)
# df_merged = pd.concat(df_list, ignore_index=True)
# df_merged = df_merged.drop_duplicates()
#
# print("Aperçu du DataFrame final fusionné et nettoyé :")
# print(df_merged.head())




