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

# Build the layout for the input form
st.markdown("### 📋 Project Details")
col1, col2 = st.columns(2)

with col1:
    project_type = st.text_input("Project Type", value="765kV Transmission Line")
    region = st.text_input("Region", value="Northern")
    budget_cr = st.number_input("Budget (Crore INR)", min_value=0.0, value=1000.0, step=50.0)
    line_length_ckm = st.number_input("Line Length (CKM)", min_value=0.0, value=100.0, step=10.0)
    planned_duration_months = st.number_input("Planned Duration (Months)", min_value=1, value=24, step=1)

with col2:
    physical_progress_pct = st.slider("Physical Progress (%)", min_value=0.0, max_value=100.0, value=10.0, step=1.0)
    land_row_status = st.text_input("Land/RoW Status", value="Clear")
    forest_clearance_status = st.text_input("Forest Clearance Status", value="Approved")
    vendor_status = st.text_input("Vendor Status", value="On Track")

st.divider()

# Prediction button
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
            model_bundle=bundle,
            verbose=False
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
