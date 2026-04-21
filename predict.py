"""
╔══════════════════════════════════════════════════════════════════╗
║        GRID-GUARD  |  Delay Prediction — Standalone Predictor   ║
║        Load the trained model and predict new projects           ║
╚══════════════════════════════════════════════════════════════════╝

USAGE:
    1. Run this script directly for demo predictions.
    2. Import predict_project() into any pipeline.
    3. Use batch_predict() for a DataFrame of new projects.
"""

import pickle
import pandas as pd
import numpy as np
import os

# ── Path to the saved model bundle ──
MODEL_PATH = "gridguard_rf_model.pkl"  # change path if needed


def load_model(path=MODEL_PATH):
    """Load the saved Grid-Guard model bundle."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found at '{path}'.\n"
            "Run train_model_new.py first to train and save the model."
        )
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    return bundle


def predict_project(
    project_type,
    region,
    budget_cr,
    line_length_ckm,
    planned_duration_months,
    physical_progress_pct,
    land_row_status,
    forest_clearance_status,
    vendor_status,
    model_bundle=None,
    verbose=True,
):
    """
    Predict delay and risk for a single new project.

    Parameters
    ----------
    project_type            : str  — '765kV Transmission Line' | '400kV Substation' | '220kV D/C Line'
    region                  : str  — 'Northern' | 'Western' | 'Southern' | 'Eastern' | 'North-Eastern'
    budget_cr               : float — Budget in Crore INR (100–5000)
    line_length_ckm         : float — Circuit kilometres (10–500)
    planned_duration_months : int   — Original schedule (12–60)
    physical_progress_pct   : float — % completion so far (0–100)
    land_row_status         : str  — 'Clear' | 'Pending Local' | 'Disputed'
    forest_clearance_status : str  — 'Approved' | 'Stage-I Awaited' | 'Stage-II Awaited'
    vendor_status           : str  — 'On Track' | 'Equipment Delayed' | 'Insolvent'
    model_bundle            : dict — pre-loaded bundle (loads from disk if None)
    verbose                 : bool — print result card

    Returns
    -------
    dict with keys:
        predicted_delay_months  : float
        risk_level              : str  ('Low' | 'Medium' | 'High')
        risk_probabilities      : dict  {'Low': %, 'Medium': %, 'High': %}
        severity_note           : str
    """
    if model_bundle is None:
        model_bundle = load_model()

    rf_reg   = model_bundle["rf_regressor"]
    rf_clf   = model_bundle["rf_classifier"]
    encoders = model_bundle["label_encoders"]
    risk_inv = model_bundle["risk_inv"]

    # ── Validate categorical inputs (Relaxed for free text) ────────
    col_map = {
        "project_type": "Project_Type",
        "region": "Region",
        "land_row_status": "Land_RoW_Status",
        "forest_clearance_status": "Forest_Clearance_Status",
        "vendor_status": "Vendor_Status",
    }
    inputs = {
        "project_type": project_type,
        "region": region,
        "land_row_status": land_row_status,
        "forest_clearance_status": forest_clearance_status,
        "vendor_status": vendor_status,
    }

    # ── Encode (with unseen category fallback) ────────────────────
    enc_vals = {}
    for py_key, col in col_map.items():
        le = encoders[col]
        val = inputs[py_key]
        if val in le.classes_:
            enc_vals[py_key] = int(le.transform([val])[0])
        else:
            enc_vals[py_key] = 0  # Fallback to a default enc if text is mismatched

    # ── Build feature vector ──────────────────────────────────────
    X_new = np.array([[
        float(budget_cr),
        float(line_length_ckm),
        float(planned_duration_months),
        float(physical_progress_pct),
        enc_vals["project_type"],
        enc_vals["region"],
        enc_vals["land_row_status"],
        enc_vals["forest_clearance_status"],
        enc_vals["vendor_status"],
    ]])

    # ── Predict ──────────────────────────────────────────────────
    delay_pred   = float(rf_reg.predict(X_new)[0])
    delay_pred   = max(0.0, round(delay_pred, 1))
    risk_enc     = int(rf_clf.predict(X_new)[0])
    risk_level   = risk_inv[risk_enc]
    risk_proba   = rf_clf.predict_proba(X_new)[0]
    class_order  = [risk_inv[i] for i in range(3)]
    risk_proba_d = {cls: round(float(p) * 100, 1) for cls, p in zip(class_order, risk_proba)}

    # ── Severity note ─────────────────────────────────────────────
    if vendor_status == "Insolvent":
        severity_note = "CRITICAL: Vendor insolvency is the primary driver. Emergency re-tendering should be initiated immediately."
    elif forest_clearance_status == "Stage-I Awaited" and land_row_status == "Disputed":
        severity_note = "HIGH: Stacked regulatory and RoW blockages detected. Escalation to Ministry of Power recommended."
    elif forest_clearance_status == "Stage-I Awaited":
        severity_note = "HIGH: Forest clearance at Stage-I is the primary bottleneck. DFO liaison should be expedited."
    elif land_row_status == "Disputed":
        severity_note = "MEDIUM-HIGH: Active RoW dispute is delaying progress. District Collector intervention required."
    elif vendor_status == "Equipment Delayed":
        severity_note = "MEDIUM: Equipment supply chain disruption detected. Alternate vendor sourcing recommended."
    elif forest_clearance_status == "Stage-II Awaited":
        severity_note = "MEDIUM: Stage-II forest clearance pending. FAC meeting should be scheduled urgently."
    elif land_row_status == "Pending Local":
        severity_note = "LOW-MEDIUM: Local RoW negotiations in progress. Panchayat-level engagement should be accelerated."
    else:
        if risk_level == "High":
             severity_note = "HIGH: ML Model detects high risk of delay primarily due to poor progress-to-time ratio compared to schedule."
        elif risk_level == "Medium":
             severity_note = "MEDIUM: Moderate risk pattern detected by the prediction engine based on current project pace."
        else:
             severity_note = "LOW: No critical impediments detected. Project is on track for timely commissioning."

    result = {
        "predicted_delay_months": delay_pred,
        "risk_level": risk_level,
        "risk_probabilities": risk_proba_d,
        "severity_note": severity_note,
    }

    if verbose:
        print("\n" + "═" * 58)
        print("  GRID-GUARD  |  Delay Prediction Result")
        print("═" * 58)
        print(f"  Project Type      : {project_type}")
        print(f"  Region            : {region}")
        print(f"  Budget            : ₹{budget_cr:,.0f} Cr")
        print(f"  Line Length       : {line_length_ckm} CKM")
        print(f"  Planned Duration  : {planned_duration_months} months")
        print(f"  Progress so far   : {physical_progress_pct}%")
        print(f"  RoW Status        : {land_row_status}")
        print(f"  Forest Clearance  : {forest_clearance_status}")
        print(f"  Vendor Status     : {vendor_status}")
        print("─" * 58)
        print(f"  ▶  Predicted Delay  : {delay_pred:.1f} months")
        risk_emoji = "🔴" if risk_level == "High" else ("🟡" if risk_level == "Medium" else "🟢")
        print(f"  ▶  Risk Level       : {risk_emoji} {risk_level}")
        print(f"  ▶  Risk Probability : Low {risk_proba_d.get('Low', 0)}%  |  "
              f"Medium {risk_proba_d.get('Medium', 0)}%  |  "
              f"High {risk_proba_d.get('High', 0)}%")
        print(f"  ⚠  Advisory        : {severity_note}")
        print("═" * 58)

    return result


def batch_predict(df_new, model_bundle=None):
    """
    Predict delays for a DataFrame of new projects.

    df_new must have columns:
        Project_Type, Region, Budget_Cr, Line_Length_CKM,
        Planned_Duration_Months, Physical_Progress_Pct,
        Land_RoW_Status, Forest_Clearance_Status, Vendor_Status

    Returns the same DataFrame with three new columns appended:
        Predicted_Delay_Months, Predicted_Risk_Level, Risk_Probabilities
    """
    if model_bundle is None:
        model_bundle = load_model()

    results = []
    for _, row in df_new.iterrows():
        r = predict_project(
            project_type=row["Project_Type"],
            region=row["Region"],
            budget_cr=row["Budget_Cr"],
            line_length_ckm=row["Line_Length_CKM"],
            planned_duration_months=row["Planned_Duration_Months"],
            physical_progress_pct=row["Physical_Progress_Pct"],
            land_row_status=row["Land_RoW_Status"],
            forest_clearance_status=row["Forest_Clearance_Status"],
            vendor_status=row["Vendor_Status"],
            model_bundle=model_bundle,
            verbose=False,
        )
        results.append(r)

    df_out = df_new.copy()
    df_out["Predicted_Delay_Months"] = [r["predicted_delay_months"] for r in results]
    df_out["Predicted_Risk_Level"]   = [r["risk_level"] for r in results]
    df_out["Risk_Probabilities"]     = [str(r["risk_probabilities"]) for r in results]
    df_out["Advisory"]               = [r["severity_note"] for r in results]
    return df_out


# ════════════════════════════════════════════════════════════════
#  DEMO PREDICTIONS — run this file directly to see examples
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    print("\n" + "═" * 58)
    print("  GRID-GUARD  |  New Project Prediction Demo")
    print("═" * 58)

    bundle = load_model()

    # ── CASE 1: Worst-case scenario ───────────────────────────────
    print("\n📋 CASE 1 — Worst-case (Insolvent vendor + Forest + RoW dispute)")
    predict_project(
        project_type            = "765kV Transmission Line",
        region                  = "North-Eastern",
        budget_cr               = 3200,
        line_length_ckm         = 420,
        planned_duration_months = 48,
        physical_progress_pct   = 18.5,
        land_row_status         = "Disputed",
        forest_clearance_status = "Stage-I Awaited",
        vendor_status           = "Insolvent",
        model_bundle            = bundle,
    )

    # ── CASE 2: Moderate risk ─────────────────────────────────────
    print("\n📋 CASE 2 — Moderate (Equipment delayed, Stage-II clearance)")
    predict_project(
        project_type            = "400kV Substation",
        region                  = "Western",
        budget_cr               = 850,
        line_length_ckm         = 65,
        planned_duration_months = 36,
        physical_progress_pct   = 52.0,
        land_row_status         = "Pending Local",
        forest_clearance_status = "Stage-II Awaited",
        vendor_status           = "Equipment Delayed",
        model_bundle            = bundle,
    )

    # ── CASE 3: On-track project ───────────────────────────────────
    print("\n📋 CASE 3 — On track (All clearances obtained)")
    predict_project(
        project_type            = "220kV D/C Line",
        region                  = "Southern",
        budget_cr               = 320,
        line_length_ckm         = 145,
        planned_duration_months = 24,
        physical_progress_pct   = 78.0,
        land_row_status         = "Clear",
        forest_clearance_status = "Approved",
        vendor_status           = "On Track",
        model_bundle            = bundle,
    )

    # ── CASE 4: Custom — user's own project ───────────────────────
    print("\n📋 CASE 4 — Custom project (edit values as needed)")
    predict_project(
        project_type            = "765kV Transmission Line",
        region                  = "Eastern",
        budget_cr               = 2100,
        line_length_ckm         = 310,
        planned_duration_months = 42,
        physical_progress_pct   = 35.0,
        land_row_status         = "Disputed",
        forest_clearance_status = "Approved",
        vendor_status           = "Equipment Delayed",
        model_bundle            = bundle,
    )

    # ── BATCH PREDICTION DEMO ─────────────────────────────────────
    print("\n" + "═" * 58)
    print("  BATCH PREDICTION DEMO (5 new projects)")
    print("═" * 58)

    new_projects = pd.DataFrame([
        {"Project_Type": "765kV Transmission Line", "Region": "Northern",
         "Budget_Cr": 4500, "Line_Length_CKM": 480, "Planned_Duration_Months": 54,
         "Physical_Progress_Pct": 12, "Land_RoW_Status": "Disputed",
         "Forest_Clearance_Status": "Stage-I Awaited", "Vendor_Status": "Insolvent"},

        {"Project_Type": "400kV Substation", "Region": "Western",
         "Budget_Cr": 600, "Line_Length_CKM": 45, "Planned_Duration_Months": 30,
         "Physical_Progress_Pct": 60, "Land_RoW_Status": "Clear",
         "Forest_Clearance_Status": "Stage-II Awaited", "Vendor_Status": "On Track"},

        {"Project_Type": "220kV D/C Line", "Region": "Southern",
         "Budget_Cr": 280, "Line_Length_CKM": 120, "Planned_Duration_Months": 20,
         "Physical_Progress_Pct": 88, "Land_RoW_Status": "Clear",
         "Forest_Clearance_Status": "Approved", "Vendor_Status": "On Track"},

        {"Project_Type": "765kV Transmission Line", "Region": "Eastern",
         "Budget_Cr": 3800, "Line_Length_CKM": 395, "Planned_Duration_Months": 48,
         "Physical_Progress_Pct": 22, "Land_RoW_Status": "Pending Local",
         "Forest_Clearance_Status": "Stage-I Awaited", "Vendor_Status": "Equipment Delayed"},

        {"Project_Type": "400kV Substation", "Region": "North-Eastern",
         "Budget_Cr": 1200, "Line_Length_CKM": 88, "Planned_Duration_Months": 40,
         "Physical_Progress_Pct": 44, "Land_RoW_Status": "Disputed",
         "Forest_Clearance_Status": "Approved", "Vendor_Status": "Insolvent"},
    ])

    results_df = batch_predict(new_projects, model_bundle=bundle)

    print(results_df[[
        "Project_Type", "Region",
        "Land_RoW_Status", "Forest_Clearance_Status", "Vendor_Status",
        "Predicted_Delay_Months", "Predicted_Risk_Level"
    ]].to_string(index=False))
    print()
