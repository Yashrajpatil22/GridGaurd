"""
╔══════════════════════════════════════════════════════════════════╗
║     GRID-GUARD  |  Custom Risk Classifier  v3.0                  ║
╚══════════════════════════════════════════════════════════════════╝

v3 upgrades the classifier to XGBoost (benchmark winner: 96.56%).
The XGBWrapper class is defined HERE (not inside train_final.py) so
that pickle/unpickle works correctly from ANY script (app.py, predict.py).

WHY THE WRAPPER EXISTS:
  The saved .pkl bundle stores the classifier under key "classifier".
  predict.py calls classifier.predict() and classifier.predict_proba().
  XGBClassifier has these methods natively, so the wrapper is thin —
  it just adds the .classes_ attribute that our predict.py reads.

The PUBLIC API (fit / predict / predict_proba / classes_) is IDENTICAL
across all versions, so predict.py and app.py require ZERO changes.
"""

import numpy as np
import pandas as pd
import pickle

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight


# ─────────────────────────────────────────────────────────────────────────────
# XGBWrapper — wraps XGBClassifier with the API predict.py expects
# MUST live in this module (custom_model.py) so pickle can resolve it
# from any calling script (app.py, train_final.py, predict.py, etc.)
# ─────────────────────────────────────────────────────────────────────────────
class XGBWrapper:
    """
    Thin wrapper around XGBClassifier that adds a .classes_ attribute.

    Pickle-safe: because this class is defined at the TOP LEVEL of
    custom_model.py (an importable module), Python can always find it
    regardless of which script does the unpickling.
    """

    def __init__(self, xgb_model, classes):
        self._model   = xgb_model
        self.classes_ = np.array(classes)

    def predict(self, X):
        return self._model.predict(np.array(X))

    def predict_proba(self, X):
        return self._model.predict_proba(np.array(X))


class CustomGridGuardClassifier:
    """
    GridGuard Risk Classifier — Gradient Boosting Ensemble.

    Classifies power-grid projects into Low / Medium / High risk by
    training a gradient boosting tree ensemble on historical project data.
    Balanced sample weights ensure the model does not ignore the rare
    "Low Risk" class.

    Parameters
    ----------
    k : int
        Kept for backward-compatibility with v1 API. Not used internally.
    """

    def __init__(self, k=5):
        # k is kept so any existing code that passes k=5 doesn't break
        self.k = k
        self.classes_ = None

        # ── Internal GBM model ────────────────────────────────────────────────
        # Hyperparameters chosen to balance bias and variance on a ~2000-row
        # dataset. Lower learning_rate + more trees generalise better.
        self._model = GradientBoostingClassifier(
            n_estimators=300,      # 300 boosting rounds
            max_depth=4,           # Shallow trees → less overfit
            learning_rate=0.05,    # Small steps → better generalisation
            min_samples_leaf=8,    # Require ≥8 samples per leaf
            subsample=0.8,         # Stochastic boosting — samples 80% each round
            random_state=42,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(self, X, y):
        """
        Train the classifier on historical project data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features) — scaled feature matrix
        y : array-like, shape (n_samples,)             — integer-encoded risk labels
        """
        X = np.array(X)
        y = np.array(y)
        self.classes_ = np.unique(y)

        # Balanced weights: rare classes (e.g. "Low") get a higher weight so
        # the model is penalised more for missing them.
        sample_weights = compute_sample_weight(class_weight="balanced", y=y)

        self._model.fit(X, y, sample_weight=sample_weights)

        # Class distribution info for transparency
        unique, counts = np.unique(y, return_counts=True)
        dist_str = "  |  ".join(
            [f"Class {u}: {c} samples" for u, c in zip(unique, counts)]
        )
        print(
            f"[CustomGridGuardClassifier] Trained GradientBoosting on "
            f"{len(X)} projects\n"
            f"  Class distribution: {dist_str}"
        )

    def predict(self, X):
        """Return the most likely risk class for each input row."""
        return self._model.predict(np.array(X))

    def predict_proba(self, X):
        """Return class probability estimates, shape (n_samples, n_classes)."""
        return self._model.predict_proba(np.array(X))


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.metrics import accuracy_score, classification_report
    from feature_engineering import engineer_features, ENG_COLS

    print("─" * 60)
    print("  CustomGridGuardClassifier — Standalone Evaluation")
    print("─" * 60)

    df = pd.read_csv("dataset/GridGuard_Dataset_50000.csv")
    if "Time_Elapsed_Months" in df.columns:
        df.rename(columns={"Time_Elapsed_Months": "months_elapsed"}, inplace=True)

    print("\n[INFO] Risk Level distribution:")
    print(df["Risk_Level"].value_counts().to_string())

    cat_cols = ["Project_Type", "Region", "Land_RoW_Status",
                "Forest_Clearance_Status", "Vendor_Status"]
    num_cols = ["Budget_Cr", "Line_Length_CKM",
                "Planned_Duration_Months", "Physical_Progress_Pct"]

    # Feature engineering
    df = engineer_features(df)

    le_risk = LabelEncoder()
    y_clf = le_risk.fit_transform(df["Risk_Level"].values)

    df_encoded = pd.get_dummies(
        df[num_cols + ENG_COLS + cat_cols], columns=cat_cols, drop_first=False
    )
    X = df_encoded.values

    # Fix: split BEFORE scaling to avoid leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_clf, test_size=0.2, random_state=42, stratify=y_clf
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    model = CustomGridGuardClassifier(k=5)
    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, predictions)

    print(f"\n[EVALUATION ON HELD-OUT TEST DATA]")
    print(f"  Accuracy : {acc * 100:.2f}%")
    print(f"\n{classification_report(y_test, predictions, target_names=le_risk.classes_)}")

    # Cross-validation for a reliable estimate
    cv_scores = cross_val_score(
        model._model, X_train_scaled, y_train, cv=5, scoring="accuracy"
    )
    print(f"[5-Fold CV Accuracy] {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
