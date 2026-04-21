"""
╔══════════════════════════════════════════════════════════════════╗
║     GRID-GUARD  |  Model Training Pipeline  v2.0                 ║
╚══════════════════════════════════════════════════════════════════╝

Improvements over v1
─────────────────────
1. Feature Engineering  — 7 new domain-specific derived features
2. No Data Leakage      — scaler.fit only on training split
3. Stratified split     — preserves class ratios in train & test
4. Classifier upgrade   — KNN → GradientBoosting (balanced weights)
5. Regressor tuning     — GradientBoostingRegressor + GridSearchCV
6. Full evaluation      — accuracy, F1, MAE, RMSE, R², 5-fold CV
"""

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    cross_val_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from custom_model import CustomGridGuardClassifier
from feature_engineering import engineer_features, ENG_COLS

HEADER = "=" * 62

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load Data
# ─────────────────────────────────────────────────────────────────────────────
print(HEADER)
print("  GRID-GUARD  |  Model Training Pipeline  v2.0")
print(HEADER)

print("\n[1/7] Loading dataset ...")
df = pd.read_excel("dataset/GridGuard_Dataset_2000.xlsx", skiprows=2)
print(f"      Loaded {len(df)} rows × {len(df.columns)} columns")

print("\n[INFO] Risk Level distribution (before balancing):")
print(df["Risk_Level"].value_counts().to_string())

# ── Synthesize months_elapsed ─────────────────────────────────────────────────
# Real-world: months_elapsed is recorded when the status update is taken.
# For our synthetic dataset we derive it: a project that is X% done on a
# plan of P months, and ultimately delayed by D months, was observed at
# approximately X% of its total actual duration.
# Small Gaussian noise prevents the model from seeing pure label leakage.
print("\n[INFO] Synthesizing months_elapsed from progress + actual delay ...")
np.random.seed(42)
total_actual_duration = df["Planned_Duration_Months"] + df["Actual_Delay_Months"]
df["months_elapsed"] = np.clip(
    (df["Physical_Progress_Pct"] / 100.0) * total_actual_duration
    + np.random.normal(0, 0.5, len(df)),
    0,
    total_actual_duration,
).round(1)
print(f"      months_elapsed range: {df['months_elapsed'].min():.1f} – {df['months_elapsed'].max():.1f} months")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/7] Engineering domain-specific features ...")
df = engineer_features(df)

cat_cols = ["Project_Type", "Region", "Land_RoW_Status",
            "Forest_Clearance_Status", "Vendor_Status"]
num_cols = ["Budget_Cr", "Line_Length_CKM",
            "Planned_Duration_Months", "Physical_Progress_Pct"]

print(f"      Original numeric features  : {len(num_cols)}")
print(f"      Engineered features added  : {len(ENG_COLS)}")
print(f"      Categorical features       : {len(cat_cols)}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Encode Labels
# ─────────────────────────────────────────────────────────────────────────────
le_risk = LabelEncoder()
y_clf = le_risk.fit_transform(df["Risk_Level"].values)
risk_inv = {i: v for i, v in enumerate(le_risk.classes_)}

y_reg = df["Actual_Delay_Months"].values

# ─────────────────────────────────────────────────────────────────────────────
# 4. One-Hot Encode & Build Feature Matrix
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/7] Encoding features ...")
df_encoded = pd.get_dummies(
    df[num_cols + ENG_COLS + cat_cols],
    columns=cat_cols,
    drop_first=False,
)
X_cols = df_encoded.columns.tolist()
X = df_encoded.values
print(f"      Final feature matrix : {X.shape[0]} rows × {X.shape[1]} columns")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Stratified Train / Test Split  ← SPLIT BEFORE SCALING (no leakage)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/7] Splitting into train/test sets (80/20, stratified by risk) ...")
(X_train, X_test,
 y_clf_train, y_clf_test,
 y_reg_train, y_reg_test) = train_test_split(
    X, y_clf, y_reg,
    test_size=0.2,
    random_state=42,
    stratify=y_clf,      # preserves Low/Medium/High proportions in both sets
)
print(f"      Training rows : {len(X_train)}")
print(f"      Test rows     : {len(X_test)}")

# Scale ONLY after splitting — fit on train, transform both
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)    # ← transform only (no leakage)

# ─────────────────────────────────────────────────────────────────────────────
# 6. Train Classifier — GradientBoosting with balanced class weights
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/7] Training Risk Classifier (GradientBoosting + balanced weights) ...")
best_clf = CustomGridGuardClassifier(k=5)
best_clf.fit(X_train_scaled, y_clf_train)

clf_preds = best_clf.predict(X_test_scaled)
clf_acc   = accuracy_score(y_clf_test, clf_preds)

print(f"\n      ✅ Test Accuracy : {clf_acc * 100:.2f}%")
print(f"\n{classification_report(y_clf_test, clf_preds, target_names=le_risk.classes_)}")

# 5-fold CV on the training set for a reliable estimate
cv_acc = cross_val_score(
    best_clf._model, X_train_scaled, y_clf_train,
    cv=5, scoring="accuracy",
)
print(f"      5-Fold CV Accuracy : {cv_acc.mean()*100:.2f}% ± {cv_acc.std()*100:.2f}%")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Train Regressor — GradientBoosting + GridSearchCV
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6/7] Tuning Delay Regressor (GridSearchCV, 5-fold CV) ...")
print("      This may take 1–3 minutes ...")

param_grid = {
    "n_estimators":     [200, 400],
    "max_depth":        [3, 5],
    "learning_rate":    [0.05, 0.1],
    "min_samples_leaf": [5, 10],
}

gbr = GradientBoostingRegressor(subsample=0.8, random_state=42)
grid_search = GridSearchCV(
    gbr,
    param_grid,
    cv=5,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    verbose=1,
)
grid_search.fit(X_train_scaled, y_reg_train)

best_reg = grid_search.best_estimator_
print(f"\n      Best hyperparameters : {grid_search.best_params_}")

reg_preds = best_reg.predict(X_test_scaled)
reg_mae   = mean_absolute_error(y_reg_test, reg_preds)
reg_rmse  = np.sqrt(mean_squared_error(y_reg_test, reg_preds))
reg_r2    = r2_score(y_reg_test, reg_preds)

print(f"\n      ✅ Delay MAE  : {reg_mae:.2f} months")
print(f"      ✅ Delay RMSE : {reg_rmse:.2f} months")
print(f"      ✅ Delay R²   : {reg_r2:.4f}  (1.0 = perfect, 0.0 = baseline)")

# ─────────────────────────────────────────────────────────────────────────────
# 8. Save Model Bundle
# ─────────────────────────────────────────────────────────────────────────────
print("\n[7/7] Saving model bundle ...")

bundle = {
    # Models
    "regressor":        best_reg,
    "classifier":       best_clf,
    # Preprocessing
    "scaler":           scaler,
    "train_columns":    X_cols,
    # Column groups (needed by predict.py to rebuild the feature matrix)
    "num_cols":         num_cols,
    "cat_cols":         cat_cols,
    "eng_cols":         ENG_COLS,
    # Risk label mapping (int → string)
    "risk_inv":         risk_inv,
}

model_path = "gridguard_best_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(bundle, f)

print(f"      ✅ Saved to '{model_path}'")
print(f"\n{HEADER}")
print("  Training Complete!")
print(HEADER)
