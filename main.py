from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os



from fastapi.middleware.cors import CORSMiddleware

# 1. Initialize the FastAPI app

app = FastAPI(title="Grid-Guard Risk Radar API")

# ADD THIS BLOCK TO ALLOW FRONTEND CONNECTIONS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for local testing
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (POST, GET, etc.)
    allow_headers=["*"],
)
# ... (Keep the rest of your code exactly the same) ...

# 2. Load the trained model and preprocessors from the new /models directory
try:
    model = joblib.load('models/gridguard_model.pkl')
    scaler = joblib.load('models/numerical_scaler.pkl')
    tfidf = joblib.load('models/tfidf_vectorizer.pkl')
    print("Machine Learning assets loaded successfully from the /models folder!")
except Exception as e:
    print(f"Error loading models: {e}")

# 3. Define the expected incoming JSON data structure
class ProjectData(BaseModel):
    cost_deviation: float
    worker_count: int
    equipment_utilization_rate: float
    material_usage: float
    vendor_remarks: str

# 4. Create the prediction endpoint
@app.post("/predict")
def predict_risk(data: ProjectData):
    # Convert incoming numerical data into a DataFrame
    num_data = pd.DataFrame([{
        'cost_deviation': data.cost_deviation,
        'worker_count': data.worker_count,
        'equipment_utilization_rate': data.equipment_utilization_rate,
        'material_usage': data.material_usage
    }])
    
    # Scale the numerical data using your saved scaler
    scaled_nums = scaler.transform(num_data)
    num_df = pd.DataFrame(scaled_nums, columns=num_data.columns)
    
    # Process the text data using your saved TF-IDF vectorizer
    text_features = tfidf.transform([data.vendor_remarks]).toarray()
    text_df = pd.DataFrame(text_features, columns=tfidf.get_feature_names_out())
    
    # Combine everything into the final format the model expects
    X_input = pd.concat([num_df, text_df], axis=1)
    
    # Make the prediction
    prediction = model.predict(X_input)[0]
    
    # Format the output
    result = "Risk/Delayed" if prediction == 1 else "Safe/On Time"
    
    return {
        "prediction": result,
        "raw_status": int(prediction),
        "input_remark": data.vendor_remarks
    }

# A simple check to ensure the server is running
@app.get("/")
def read_root():
    return {"message": "Grid-Guard API is running. Send POST requests to /predict"}