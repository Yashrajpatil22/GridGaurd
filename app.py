import streamlit as st
import pandas as pd
from predict import load_model, predict_project

# Set page configuration
st.set_page_config(page_title="GridGuard Predictor", page_icon="⚡", layout="centered")

st.title("⚡ GRID-GUARD AI")
st.subheader("Transmission Project Delay Prediction Dashboard")

st.markdown("""
Enter the project details below to evaluate potential risks and predict timeline delays using our trained Machine Learning model.
""")

# Load model once using Streamlit caching to prevent reloading on every interaction
@st.cache_resource
def get_model():
    return load_model()

try:
    bundle = get_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ── Valid category options (must match training data exactly) ─────────────────
PROJECT_TYPES       = ["765kV Transmission Line", "400kV Substation", "220kV D/C Line"]
REGIONS             = ["Northern", "Western", "Southern", "Eastern", "North-Eastern"]
LAND_ROW_OPTIONS    = ["Clear", "Pending Local", "Disputed"]
FOREST_OPTIONS      = ["Approved", "Stage-II Awaited", "Stage-I Awaited"]
VENDOR_OPTIONS      = ["On Track", "Equipment Delayed", "Insolvent"]

# ── Build the layout for the input form ──────────────────────────────────────
st.markdown("### 📋 Project Details")
col1, col2 = st.columns(2)

with col1:
    project_type = st.selectbox("Project Type", options=PROJECT_TYPES, index=0)
    region = st.selectbox("Region", options=REGIONS, index=2)
    budget_cr = st.number_input("Budget (Crore INR)", min_value=0.0, value=490.0, step=50.0)
    line_length_ckm = st.number_input("Line Length (CKM)", min_value=0.0, value=100.0, step=10.0)
    planned_duration_months = st.number_input("Planned Duration (Months)", min_value=1, value=24, step=1)

with col2:
    physical_progress_pct = st.slider("Physical Progress (%)", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
    land_row_status = st.selectbox("Land/RoW Status", options=LAND_ROW_OPTIONS, index=0)
    forest_clearance_status = st.selectbox("Forest Clearance Status", options=FOREST_OPTIONS, index=0)
    vendor_status = st.selectbox("Vendor Status", options=VENDOR_OPTIONS, index=0)

# ── Months Elapsed (optional but highly recommended) ─────────────────────────
st.markdown("### 🕐 Schedule Status *(optional but improves accuracy significantly)*")

months_elapsed = st.number_input(
    "Months Elapsed Since Project Start",
    min_value=0,
    max_value=120,
    value=0,
    step=1,
    help=(
        "How many months have passed since the project officially started? "
        "Set to 0 if unknown. Providing this allows the model to detect if the "
        "project is already running behind schedule."
    ),
)

if months_elapsed > 0:
    expected_progress = min(100.0, (months_elapsed / planned_duration_months) * 100)
    deficit = expected_progress - physical_progress_pct
    if deficit > 5:
        st.warning(
            f"⚠️ **Schedule gap detected:** At {months_elapsed} months elapsed, "
            f"expected progress is **{expected_progress:.1f}%** but actual is "
            f"**{physical_progress_pct:.1f}%** — project is **{deficit:.1f}% behind schedule**."
        )
    elif deficit < -5:
        st.success(
            f"✅ **Ahead of schedule:** At {months_elapsed} months elapsed, "
            f"expected progress is **{expected_progress:.1f}%** but actual is "
            f"**{physical_progress_pct:.1f}%** — project is **{abs(deficit):.1f}% ahead**."
        )
    else:
        st.info(f"📊 Project is roughly on schedule at {months_elapsed} months elapsed.")

st.divider()

# ── Prediction button ─────────────────────────────────────────────────────────
if st.button("Predict Project Delay & Risk", type="primary", use_container_width=True):
    with st.spinner("Analyzing project data..."):
        result = predict_project(
            project_type=project_type,
            region=region,
            budget_cr=budget_cr,
            line_length_ckm=line_length_ckm,
            planned_duration_months=planned_duration_months,
            physical_progress_pct=physical_progress_pct,
            land_row_status=land_row_status,
            forest_clearance_status=forest_clearance_status,
            vendor_status=vendor_status,
            months_elapsed=months_elapsed,
            model_bundle=bundle,
            verbose=False,
        )

        st.subheader("📊 Prediction Results")

        # Display key metrics side-by-side
        rc1, rc2 = st.columns(2)
        rc1.metric(label="Predicted Delay", value=f"{result['predicted_delay_months']} Months")

        risk = result['risk_level']
        risk_color = "🔴" if risk == "High" else "🟡" if risk == "Medium" else "🟢"
        rc2.metric(label="Risk Level", value=f"{risk_color} {risk}")

        st.write("")
        st.write("**Risk Probabilities:**")
        probs = result['risk_probabilities']

        # Show mini progress bars for probabilities
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            st.metric("Low Risk", f"{probs.get('Low', 0)}%")
        with col_p2:
            st.metric("Medium Risk", f"{probs.get('Medium', 0)}%")
        with col_p3:
            st.metric("High Risk", f"{probs.get('High', 0)}%")

        st.write("")
        # Highlighted advisory message based on project risk
        if risk == "High":
            st.error(f"**🚨 Advisory:** {result['severity_note']}")
        elif risk == "Medium":
            st.warning(f"**⚠️ Advisory:** {result['severity_note']}")
        else:
            st.success(f"**✅ Advisory:** {result['severity_note']}")

        # ── Explainability (SHAP Plot) ─────────────────────────────────────────
        if result.get("shap_explanation") is not None:
            st.divider()
            st.subheader("🧠 Why this delay? (AI Explainability)")
            st.info("The waterfall chart below shows exactly how much each factor added or subtracted from the base delay. **Red bars** increase the delay, **blue bars** decrease it.")
            
            import matplotlib.pyplot as plt
            import shap
            
            # Create a larger figure to ensure labels aren't cut off
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.plots.waterfall(result["shap_explanation"], max_display=10, show=False)
            plt.tight_layout()
            
            st.pyplot(fig)
            plt.clf()

