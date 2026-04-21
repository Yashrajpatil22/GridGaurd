"""
╔══════════════════════════════════════════════════════════════════╗
║     GRID-GUARD  |  Model Training Pipeline  v3.0                 ║
╚══════════════════════════════════════════════════════════════════╝

Changes in v3.0
─────────────────────
1. Dataset upgraded    — GridGuard_Dataset_50000.csv (50 000 rows)
2. months_elapsed      — uses real Time_Elapsed_Months column (no synthesis)
3. Classifier upgrade  — LightGBM (fastest + best accuracy on 50K rows)
4. Regressor upgrade   — LightGBM Regressor (replaces sklearn GBM)
5. Fallback            — auto-falls back to GradientBoosting if lgbm absent
6. Full evaluation      — accuracy, F1, MAE, RMSE, R², 5-fold CV
"""

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
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
from sklearn.utils.class_weight import compute_sample_weight

try:
    from xgboost import XGBClassifier, XGBRegressor
    USE_XGB = True
    print("[INFO] XGBoost detected — will use XGBoost models (best on benchmark)")
except ImportError:
    USE_XGB = False
    print("[WARN] XGBoost not installed — falling back to GradientBoosting")
    print("       Install with: pip install xgboost")

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
df = pd.read_csv("dataset/GridGuard_Dataset_50000.csv")
print(f"      Loaded {len(df)} rows × {len(df.columns)} columns")

print("\n[INFO] Risk Level distribution (before balancing):")
print(df["Risk_Level"].value_counts().to_string())

# ── Use the REAL Time_Elapsed_Months column from the 50K dataset ──────────────
# Unlike the 2000-row XLSX, this CSV contains actual recorded elapsed time.
# We just rename it to what feature_engineering.py expects.
print("\n[INFO] Using real Time_Elapsed_Months from dataset (no synthesis needed) ...")
if "Time_Elapsed_Months" in df.columns:
    df.rename(columns={"Time_Elapsed_Months": "months_elapsed"}, inplace=True)
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
# 6. Train Classifier — LightGBM (best on 50K rows) or GBM fallback
# ─────────────────────────────────────────────────────────────────────────────
if USE_XGB:
    print("\n[5/7] Training Risk Classifier (XGBoost — benchmark winner 96.56%) ...")
    best_clf_model = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
    )
    best_clf_model.fit(X_train_scaled, y_clf_train)

    class _XGBWrapper:
        """Thin wrapper so bundle stays compatible with predict.py API."""
        def __init__(self, m, classes):
            self._model  = m
            self.classes_ = classes
        def predict(self, X):
            return self._model.predict(X)
        def predict_proba(self, X):
            return self._model.predict_proba(X)

    best_clf = _XGBWrapper(best_clf_model, np.unique(y_clf_train))
else:
    print("\n[5/7] Training Risk Classifier (GradientBoosting + balanced weights) ...")
    best_clf = CustomGridGuardClassifier(k=5)
    best_clf.fit(X_train_scaled, y_clf_train)
    best_clf_model = best_clf._model

clf_preds = best_clf.predict(X_test_scaled)
clf_acc   = accuracy_score(y_clf_test, clf_preds)

print(f"\n      ✅ Test Accuracy : {clf_acc * 100:.2f}%")
print(f"\n{classification_report(y_clf_test, clf_preds, target_names=le_risk.classes_)}")

# 5-fold CV on the training set for a reliable estimate
cv_acc = cross_val_score(
    best_clf_model, X_train_scaled, y_clf_train,
    cv=5, scoring="accuracy", n_jobs=-1,
)
print(f"      5-Fold CV Accuracy : {cv_acc.mean()*100:.2f}% ± {cv_acc.std()*100:.2f}%")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Train Regressor — LightGBM (fast on 50K) or GBM GridSearchCV fallback
# ─────────────────────────────────────────────────────────────────────────────
if USE_XGB:
    print("\n[6/7] Training Delay Regressor (XGBoost) ...")
    best_reg = XGBRegressor(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    best_reg.fit(X_train_scaled, y_reg_train)
else:
    print("\n[6/7] Tuning Delay Regressor (GridSearchCV, 5-fold CV) ...")
    print("      This may take 2-5 minutes on 50K rows ...")

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
