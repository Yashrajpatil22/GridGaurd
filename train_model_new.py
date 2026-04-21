import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import os

print("Loading dataset...")
# Load the actual new dataset
df = pd.read_excel('dataset/GridGuard_Dataset_2000.xlsx', skiprows=2)

print(f"Dataset loaded: {df.shape[0]} rows.")

# Categorical features and target columns
cat_cols = ['Project_Type', 'Region', 'Land_RoW_Status', 'Forest_Clearance_Status', 'Vendor_Status']
num_cols = ['Budget_Cr', 'Line_Length_CKM', 'Planned_Duration_Months', 'Physical_Progress_Pct']

# 1. Fit Label Encoders for inputs
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Prepare features
X = df[num_cols + cat_cols].values

# 2. Target for Regression: Actual_Delay_Months
y_reg = df['Actual_Delay_Months'].values

# 3. Target for Classification: Risk_Level
y_clf_raw = df['Risk_Level'].values
le_risk = LabelEncoder()
y_clf = le_risk.fit_transform(y_clf_raw)

# Risk inversion mapping
risk_inv = {i: v for i, v in enumerate(le_risk.classes_)}

print("Training Regressor...")
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X, y_reg)

print("Training Classifier...")
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X, y_clf)

# Save the bundle
bundle = {
    "rf_regressor": rf_reg,
    "rf_classifier": rf_clf,
    "label_encoders": encoders,
    "risk_inv": risk_inv
}

model_path = "gridguard_rf_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(bundle, f)

print(f"Model successfully saved to {model_path}!")
