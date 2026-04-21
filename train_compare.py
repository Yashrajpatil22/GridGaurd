"""
╔══════════════════════════════════════════════════════════════════╗
║  GRID-GUARD  |  Multi-Model Benchmark  v1.0                     ║
║  Tests 6 classifiers on GridGuard_Dataset_50000.csv             ║
║  Prints accuracy, F1, and cross-val scores for each             ║
╚══════════════════════════════════════════════════════════════════╝

RUN:  python train_compare.py
      Results saved to model_comparison_results.csv
"""

import pandas as pd
import numpy as np
import warnings
import time
warnings.filterwarnings("ignore")

from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)
from sklearn.utils.class_weight import compute_sample_weight

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("  [WARN] xgboost not installed — skipping XGBoost")

try:
    import lightgbm as lgb
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("  [WARN] lightgbm not installed — skipping LightGBM")

try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
except ImportError:
    HAS_CAT = False
    print("  [WARN] catboost not installed — skipping CatBoost")

from feature_engineering import engineer_features, ENG_COLS

HEADER = "=" * 72

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load Dataset (50K CSV — the REAL dataset)
# ─────────────────────────────────────────────────────────────────────────────
print(HEADER)
print("  GRID-GUARD  |  Multi-Model Benchmark")
print(HEADER)

print("\n[1/5] Loading GridGuard_Dataset_50000.csv ...")
df = pd.read_csv("dataset/GridGuard_Dataset_50000.csv")
print(f"      Loaded {len(df)} rows × {len(df.columns)} columns")

# Rename the real column to what feature_engineering.py expects
if "Time_Elapsed_Months" in df.columns:
    df.rename(columns={"Time_Elapsed_Months": "months_elapsed"}, inplace=True)

print("\n[INFO] Risk Level distribution:")
print(df["Risk_Level"].value_counts().to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 2. Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/5] Engineering domain-specific features ...")
df = engineer_features(df)

cat_cols = ["Project_Type", "Region", "Land_RoW_Status",
            "Forest_Clearance_Status", "Vendor_Status"]
num_cols = ["Budget_Cr", "Line_Length_CKM",
            "Planned_Duration_Months", "Physical_Progress_Pct"]

# ─────────────────────────────────────────────────────────────────────────────
# 3. Encode & Build Feature Matrix
# ─────────────────────────────────────────────────────────────────────────────
print("[3/5] Encoding features ...")
le_risk = LabelEncoder()
y = le_risk.fit_transform(df["Risk_Level"].values)
class_names = le_risk.classes_

df_encoded = pd.get_dummies(
    df[num_cols + ENG_COLS + cat_cols],
    columns=cat_cols,
    drop_first=False,
)
X = df_encoded.values
print(f"      Feature matrix: {X.shape[0]} rows × {X.shape[1]} columns")
print(f"      Classes: {list(class_names)}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Stratified Train/Test Split
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/5] Splitting data (80/20 stratified) ...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"      Train: {len(X_train)}   Test: {len(X_test)}")

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Define Models
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/5] Training and evaluating models ...\n")

models = {
    "GradientBoost (current)": GradientBoostingClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        min_samples_leaf=8, subsample=0.8, random_state=42,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=None, class_weight="balanced",
        random_state=42, n_jobs=-1,
    ),
    "Logistic Regression": LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=42,
    ),
}

if HAS_XGB:
    # XGBoost expects labels 0..N-1
    scale_pos = dict(zip(np.unique(y_train),
                         [sum(y_train == c) for c in np.unique(y_train)]))
    models["XGBoost"] = XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="mlogloss", use_label_encoder=False,
        random_state=42, n_jobs=-1,
    )

if HAS_LGB:
    models["LightGBM"] = LGBMClassifier(
        n_estimators=500, max_depth=-1, learning_rate=0.05,
        num_leaves=63, min_child_samples=20,
        is_unbalance=True,
        random_state=42, n_jobs=-1, verbose=-1,
    )

if HAS_CAT:
    models["CatBoost"] = CatBoostClassifier(
        iterations=500, depth=6, learning_rate=0.05,
        auto_class_weights="Balanced",
        random_seed=42, verbose=0,
    )

# ─────────────────────────────────────────────────────────────────────────────
# 6. Train, Evaluate, Compare
# ─────────────────────────────────────────────────────────────────────────────
results = []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, clf in models.items():
    print(f"  >>  Training {name} ...")
    t0 = time.time()

    if "GradientBoost" in name:
        clf.fit(X_train_s, y_train, sample_weight=sample_weights)
    elif "CatBoost" in name:
        clf.fit(X_train_s, y_train)
    elif "LightGBM" in name:
        clf.fit(X_train_s, y_train)
    else:
        clf.fit(X_train_s, y_train)

    elapsed = time.time() - t0

    preds = clf.predict(X_test_s)
    acc   = accuracy_score(y_test, preds)
    f1    = f1_score(y_test, preds, average="weighted")

    # 5-fold CV on training set only
    cv_scores = cross_val_score(clf, X_train_s, y_train, cv=skf, scoring="accuracy", n_jobs=-1)
    cv_mean   = cv_scores.mean()
    cv_std    = cv_scores.std()

    print(f"       Test Acc: {acc*100:.2f}%  |  Weighted F1: {f1:.4f}  |  5-CV: {cv_mean*100:.2f}% ± {cv_std*100:.2f}%  |  Time: {elapsed:.1f}s")
    print(f"\n{classification_report(y_test, preds, target_names=class_names)}")

    results.append({
        "Model":          name,
        "Test_Accuracy":  round(acc * 100, 2),
        "Weighted_F1":    round(f1, 4),
        "CV_Accuracy":    round(cv_mean * 100, 2),
        "CV_Std":         round(cv_std * 100, 2),
        "Train_Time_s":   round(elapsed, 1),
    })

# ─────────────────────────────────────────────────────────────────────────────
# 7. Summary Table
# ─────────────────────────────────────────────────────────────────────────────
results_df = pd.DataFrame(results).sort_values("Test_Accuracy", ascending=False)
results_df.to_csv("model_comparison_results.csv", index=False)

print("\n" + HEADER)
print("  BENCHMARK RESULTS (ranked by Test Accuracy)")
print(HEADER)
print(results_df.to_string(index=False))
print(HEADER)
print(f"\n✅ Best Model: {results_df.iloc[0]['Model']}  ({results_df.iloc[0]['Test_Accuracy']}% accuracy)")
print(f"\n📄 Full results saved to: model_comparison_results.csv")
print()
