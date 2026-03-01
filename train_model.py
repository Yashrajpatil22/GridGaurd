import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib # Used for saving the model later

print("Loading dataset and preprocessing...")

# 1. Load the Data
df = pd.read_csv('./dataset/GridGuard_Hybrid_Dataset.csv')

# 2. Define the Target Variable (Risk = 1, Safe = 0)
df['Risk_Flag'] = (df['time_deviation'] > 0).astype(int)

# 3. Process Text Data (TF-IDF Vectorization)
tfidf = TfidfVectorizer(max_features=100, stop_words='english')
text_features = tfidf.fit_transform(df['Vendor_Remarks']).toarray()
text_df = pd.DataFrame(text_features, columns=tfidf.get_feature_names_out())

# 4. Process Numerical Data (Scaling)
num_cols = ['cost_deviation', 'worker_count', 'equipment_utilization_rate', 'material_usage']
scaler = StandardScaler()
scaled_nums = scaler.fit_transform(df[num_cols])
num_df = pd.DataFrame(scaled_nums, columns=num_cols)

# 5. Combine Features (X) and Target (y)
X = pd.concat([num_df, text_df], axis=1)
y = df['Risk_Flag']

# 6. Train-Test Split (80% for training, 20% for testing)
print("Splitting data for training and testing...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Initialize and Train the Random Forest Model
print("Training the Random Forest model (this might take a few seconds)...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 8. Evaluate the Model
print("Evaluating the model on unseen test data...")
y_pred = rf_model.predict(X_test)

print("\n=== Model Performance Report ===")
print(f"Overall Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print(classification_report(y_test, y_pred, target_names=['Safe (0)', 'Risk/Delayed (1)']))

# 9. Save the Model and Preprocessors for the API (Phase 4)
joblib.dump(rf_model, './models/gridguard_model.pkl')
joblib.dump(tfidf, './models/tfidf_vectorizer.pkl')
joblib.dump(scaler, './models/numerical_scaler.pkl')
print("\nModel and preprocessors saved successfully for deployment!")