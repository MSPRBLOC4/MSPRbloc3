import pandas as pd
import unicodedata

F1      = "fusion_pres17_muni14.csv"
F2      = "fusion_pres22_muni20.csv"
REG_MAP = "departement par region.xlsx"
OUT     = "ensemble_elections.csv"

df1 = pd.read_csv(F1, dtype=str)
df2 = pd.read_csv(F2, dtype=str)
df_concat = pd.concat([df1, df2], ignore_index=True)

def clean(col: str) -> str:
    """minuscule, accents supprimés, espaces -> '_'"""
    col = unicodedata.normalize("NFKD", col).encode("ascii", "ignore").decode()
    return col.strip().lower().replace(" ", "_")

reg_df = pd.read_excel(REG_MAP, dtype=str)
reg_df.columns = [clean(c) for c in reg_df.columns]
reg_df = reg_df.rename(columns={"code_du_departement": "code_dept"})
deps_valides = set(reg_df["code_dept"])
avant = len(df_concat)
df_concat = df_concat[df_concat["code_dept"].isin(deps_valides)].reset_index(drop=True)
apres = len(df_concat)
print(f" Départements inconnus supprimés : {avant - apres}")
df_concat = df_concat.merge(reg_df[["code_dept", "region"]], on="code_dept", how="left")
cols = ["region"] + [c for c in df_concat.columns if c != "region"]
df_concat = df_concat[cols]
df_concat.to_csv(OUT, index=False, encoding="utf-8")
print(f"{OUT} créé ({len(df_concat)} lignes, {len(df_concat.columns)} colonnes)")
