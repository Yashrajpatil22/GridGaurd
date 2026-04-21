import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from custom_model import CustomGridGuardClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

print("Training the finalized BEST model (Linear / Logistic Regression Pipeline)...")

# 1. Load Data
df = pd.read_excel('dataset/GridGuard_Dataset_2000.xlsx', skiprows=2)

cat_cols = ['Project_Type', 'Region', 'Land_RoW_Status', 'Forest_Clearance_Status', 'Vendor_Status']
num_cols = ['Budget_Cr', 'Line_Length_CKM', 'Planned_Duration_Months', 'Physical_Progress_Pct']

# 2. Risk Label Encoder
le_risk = LabelEncoder()
y_clf = le_risk.fit_transform(df['Risk_Level'].values)
risk_inv = {i: v for i, v in enumerate(le_risk.classes_)}

y_reg = df['Actual_Delay_Months'].values

# 3. Features & One-Hot Encoding
df_encoded = pd.get_dummies(df[num_cols + cat_cols], columns=cat_cols, drop_first=False)
X_cols = df_encoded.columns.tolist()  # Save these to align new inputs later!
X = df_encoded.values

# 4. Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Train Regressor & Classifier
best_reg = RandomForestRegressor(n_estimators=100, random_state=42)
best_reg.fit(X_scaled, y_reg)

best_clf = CustomGridGuardClassifier(k=5)
best_clf.fit(X_scaled, y_clf)

# 6. Save Bundle
bundle = {
    "regressor": best_reg,
    "classifier": best_clf,
    "scaler": scaler,
    "train_columns": X_cols,
    "num_cols": num_cols,
    "cat_cols": cat_cols,
    "risk_inv": risk_inv
}

model_path = "gridguard_best_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(bundle, f)

print(f"✅ Final Best Model saved to {model_path}!")
