#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import warnings, joblib
from sklearn.preprocessing   import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose         import ColumnTransformer
from sklearn.pipeline        import Pipeline
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_validate
warnings.filterwarnings("ignore", category=UserWarning)

# ------------------------------------------------------------------ #
# 1. Lecture & cible                                                 #
# ------------------------------------------------------------------ #
df     = pd.read_csv("model_to_train.csv",  sep=",", dtype=str,low_memory=False)
TARGET = "meilleur_candidat"
y_str  = df[TARGET].astype(str)
df_predict = pd.read_csv("model_to_predict.csv", sep=",", dtype=str, low_memory=False)
print(df_predict.columns.tolist())
print(f"Lignes Bretagne   : {len(df_predict)}")

# ------------------------------------------------------------------ #
# 2. Retirer les classes trop rares ( < 2 obs.)                      #
# ------------------------------------------------------------------ #
counts        = y_str.value_counts().sort_values()
rare_classes  = counts[counts < 2].index
if len(rare_classes):
    print(" Suppression classes unitaires :", list(rare_classes))
    keep = ~y_str.isin(rare_classes)
    df   = df.loc[keep].reset_index(drop=True)
    y_str = df[TARGET].astype(str)

# ------------------------------------------------------------------ #
# 3. Encodage label → entiers 0…K‑1                                  #
# ------------------------------------------------------------------ #
le = LabelEncoder()
y  = le.fit_transform(y_str)
X  = df.drop(columns=[TARGET])

# ------------------------------------------------------------------ #
# 4. Pré‑traitement                                                 #
# ------------------------------------------------------------------ #
num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object", "category"]).columns

preprocess = ColumnTransformer([
    ("num", StandardScaler(with_mean=False), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

# ------------------------------------------------------------------ #
# 5. Pipeline LightGBM                                              #
# ------------------------------------------------------------------ #
pipe_lgbm = Pipeline([
    ("prep", preprocess),
    ("clf", LGBMClassifier(
            objective     ="multiclass",
            n_estimators  =400,
            learning_rate =0.05,
            subsample     =0.8,
            colsample_bytree=0.8,
            random_state  =42))
])
def print_metrics(y_true, y_pred, prefix=""):
    acc  = accuracy_score (y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score   (y_true, y_pred, average="macro", zero_division=0)
    f1   = f1_score       (y_true, y_pred, average="macro", zero_division=0)
    print(f"{prefix}Accuracy = {acc:.3f} | "
          f"Precision_macro = {prec:.3f} | "
          f"Recall_macro = {rec:.3f} | "
          f"F1_macro = {f1:.3f}")
# ------------------------------------------------------------------ #
# 6. Évaluation : CV si possible, sinon split simple                 #
# ------------------------------------------------------------------ #
min_class = y_str.value_counts().min()
if min_class >= 2:
    n_splits = min(5, min_class)        # 2 ≤ k ≤ 5
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scoring = {"acc": "accuracy",
               "prec": "precision_macro",
               "rec": "recall_macro",
               "f1": "f1_macro"}

    cv_results = cross_validate(pipe_lgbm, X, y, cv=cv,
                                scoring=scoring, n_jobs=-1)
    print(f"\nLightGBM | {n_splits}‑fold CV (moyenne ± écart‑type) :")
    print(f"  Accuracy        = {cv_results['test_acc'].mean():.3f} ± {cv_results['test_acc'].std():.3f}")
    print(f"  Precision_macro = {cv_results['test_prec'].mean():.3f} ± {cv_results['test_prec'].std():.3f}")
    print(f"  Recall_macro    = {cv_results['test_rec'].mean():.3f} ± {cv_results['test_rec'].std():.3f}")
    print(f"  F1_macro        = {cv_results['test_f1'].mean():.3f} ± {cv_results['test_f1'].std():.3f}")
else:
    print("⚠ Toujours des classes unitaires → split 80/20")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    pipe_lgbm.fit(X_tr, y_tr)
    y_val_pred = pipe_lgbm.predict(X_val)
    print_metrics(y_val, y_val_pred, prefix="LightGBM (hold‑out) | ")

# ------------------------------------------------------------------ #
# 7. Entraînement complet & export                                   #
# ------------------------------------------------------------------ #
pipe_lgbm.fit(X, y)
joblib.dump({"pipeline": pipe_lgbm, "label_encoder": le},
            "model_lgbm.joblib")
print(" modèle LightGBM enregistré : model_lgbm.joblib")
# --- mêmes listes qu’au départ -------------------------------
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols     = X.select_dtypes(include=["object", "category"]).columns

# ---------- préparation df_predict ---------------------------parti_politique,positionnement_politique
print(df_predict.columns.tolist())
df_predict = df_predict.copy()        # évite le SettingWithCopyWarning

# 1. Cast : toutes les colonnes catégorielles en *string*
df_predict[cat_cols] = (
    df_predict[cat_cols]
        .astype(str)                  # ex. 12  -> '12'
        .fillna("missing")            # NaN     -> 'missing'
        .replace({"nan": "missing"})  # 'nan'   -> 'missing'
)

# 2. Même chose pour X_train avant le fit (sécurité)
X[cat_cols] = (
    X[cat_cols]
        .astype(str)
        .fillna("missing")
        .replace({"nan": "missing"})
)
X_pred = df_predict.drop(columns=[TARGET], errors="ignore")  # le target n’existe pas pour 2022
probas = pipe_lgbm.predict_proba(X_pred)
pred_labels = le.inverse_transform(probas.argmax(axis=1))

df_predict["prediction_meilleur_candidat"] = pred_labels
df_predict["proba_max"] = probas.max(axis=1)
pred_counts = (
    df_predict["prediction_meilleur_candidat"]
      .value_counts()
      .sort_values(ascending=False)    # optionnel : ordre décroissant
)

print(" Nombre de prédictions par candidat :")
print(pred_counts)
# éventuel export
df_predict.to_csv("predictions_bretagne.csv", index=False, encoding="utf-8")
print(" predictions_bretagne.csv créé")