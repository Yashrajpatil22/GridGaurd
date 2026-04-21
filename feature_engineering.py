"""
╔══════════════════════════════════════════════════════════════════╗
║     GRID-GUARD  |  Feature Engineering — Shared Module  v2.1    ║
║     Used by BOTH train_final.py AND predict.py                   ║
╚══════════════════════════════════════════════════════════════════╝

v2.1 adds three schedule-adherence features derived from months_elapsed.
These are the most powerful signals for predicting delay in completed or
near-completion projects — the model now knows whether a project is
already running late at the time of prediction.
"""

import pandas as pd
import numpy as np

# ── Ordinal Risk Maps ─────────────────────────────────────────────────────────
LAND_RISK_MAP = {
    "Clear":         0,
    "Pending Local": 1,
    "Disputed":      2,
}

FOREST_RISK_MAP = {
    "Approved":         0,
    "Stage-II Awaited": 1,
    "Stage-I Awaited":  2,
}

VENDOR_RISK_MAP = {
    "On Track":          0,
    "Equipment Delayed": 1,
    "Insolvent":         2,
}

_LAND_DEFAULT   = 1
_FOREST_DEFAULT = 1
_VENDOR_DEFAULT = 1


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 10 domain-specific engineered features to the input DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain:
            Budget_Cr, Line_Length_CKM, Planned_Duration_Months,
            Physical_Progress_Pct, Land_RoW_Status,
            Forest_Clearance_Status, Vendor_Status

        Optional:
            months_elapsed  (int/float, 0 = not provided)

    Returns
    -------
    pd.DataFrame — original df with new columns appended (copy, not in-place)
    """
    df = df.copy()

    # Ensure months_elapsed exists (0 = not provided by user)
    if "months_elapsed" not in df.columns:
        df["months_elapsed"] = 0
    df["months_elapsed"] = df["months_elapsed"].fillna(0)

    # ── Original 7 features ───────────────────────────────────────────────────

    # 1. Budget intensity
    df["budget_per_km"] = df["Budget_Cr"] / (df["Line_Length_CKM"].clip(lower=1))

    # 2. Progress rate (progress per planned month)
    df["progress_rate"] = (
        df["Physical_Progress_Pct"] / df["Planned_Duration_Months"].clip(lower=1)
    )

    # 3. Remaining work pressure
    df["remaining_work_ratio"] = (
        (100.0 - df["Physical_Progress_Pct"]) / df["Planned_Duration_Months"].clip(lower=1)
    )

    # 4–6. Ordinal bottleneck scores
    df["land_risk_score"] = (
        df["Land_RoW_Status"].map(LAND_RISK_MAP).fillna(_LAND_DEFAULT).astype(int)
    )
    df["forest_risk_score"] = (
        df["Forest_Clearance_Status"].map(FOREST_RISK_MAP).fillna(_FOREST_DEFAULT).astype(int)
    )
    df["vendor_risk_score"] = (
        df["Vendor_Status"].map(VENDOR_RISK_MAP).fillna(_VENDOR_DEFAULT).astype(int)
    )

    # 7. Composite bottleneck severity (0–6)
    df["composite_risk_score"] = (
        df["land_risk_score"] + df["forest_risk_score"] + df["vendor_risk_score"]
    )

    # ── NEW v2.1: Schedule-adherence features from months_elapsed ─────────────
    # These are the most important signals for ongoing / near-complete projects.
    # When months_elapsed = 0 (unknown), we fall back to a neutral estimate.

    planned = df["Planned_Duration_Months"].clip(lower=1)
    elapsed = df["months_elapsed"]
    progress = df["Physical_Progress_Pct"]
    has_elapsed = elapsed > 0

    # 8. Schedule adherence ratio
    #    > 1.0 means the clock has run past the original planned duration
    #    When unknown, use progress/100 as a proxy (assumes on-schedule)
    df["schedule_adherence"] = np.where(
        has_elapsed,
        elapsed / planned,
        progress / 100.0,
    )

    # 9. Progress deficit
    #    How far behind is actual progress vs where it should be at this moment?
    #    Positive = behind schedule. Negative = ahead.
    expected_pct = df["schedule_adherence"] * 100.0
    df["progress_deficit"] = np.clip(expected_pct - progress, -100, 100)

    # 10. Months already behind schedule
    #     If the project has consumed more time than progress justifies, we
    #     know the delay has already started accumulating.
    #     When months_elapsed = 0: 0 (neutral, unknown)
    expected_months_for_progress = (progress / 100.0) * planned
    df["months_behind_schedule"] = np.where(
        has_elapsed,
        np.maximum(0.0, elapsed - expected_months_for_progress),
        0.0,
    )

    return df


# Column names for all engineered features (used to select correct columns)
ENG_COLS = [
    # Original 7
    "budget_per_km",
    "progress_rate",
    "remaining_work_ratio",
    "land_risk_score",
    "forest_risk_score",
    "vendor_risk_score",
    "composite_risk_score",
    # New v2.1 — schedule adherence
    "schedule_adherence",
    "progress_deficit",
    "months_behind_schedule",
]
