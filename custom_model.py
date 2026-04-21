import numpy as np
import pandas as pd
import pickle

class CustomGridGuardClassifier:
    """
    A 100% custom-built Classification Model built from scratch using pure Math (NumPy).
    It classifies project risk by finding the most mathematically similar historical projects 
    (K-Nearest Neighbors algorithm). No external ML libraries used.
    """
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None
        self.classes_ = None

    def fit(self, X, y):
        # Store historical records for mathematical distance checks
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        self.classes_ = np.unique(y)
        print(f"[Custom Classifier] Memorized {len(self.X_train)} projects training facts.")

    def predict(self, X):
        X = np.array(X)
        predictions = []
        
        # For every new project, calculate geometric Euclidean distance to all historical projects
        for x_new in X:
            distances = np.sqrt(np.sum((self.X_train - x_new) ** 2, axis=1))
            
            # Find the 'K' most similar historical projects
            nearest_indices = np.argsort(distances)[:self.k]
            nearest_labels = self.y_train[nearest_indices]
            nearest_distances = distances[nearest_indices]
            
            # Distance-Weighted Voting: Closer projects have a much stronger vote
            weights = 1.0 / (nearest_distances + 1e-6)
            
            # Tally weights for each class
            class_scores = {}
            for label, weight in zip(nearest_labels, weights):
                class_scores[label] = class_scores.get(label, 0) + weight
                
            best_guess = max(class_scores, key=class_scores.get)
            predictions.append(best_guess)
            
        return np.array(predictions)

    def predict_proba(self, X):
        X = np.array(X)
        probabilities = []
        
        for x_new in X:
            distances = np.sqrt(np.sum((self.X_train - x_new) ** 2, axis=1))
            nearest_indices = np.argsort(distances)[:self.k]
            nearest_labels = self.y_train[nearest_indices]
            nearest_distances = distances[nearest_indices]
            
            # Distance-Weighted Probabilities
            weights = 1.0 / (nearest_distances + 1e-6)
            total_weight = np.sum(weights)
            
            prob_row = []
            for c in self.classes_:
                prob = np.sum(weights[nearest_labels == c]) / total_weight
                prob_row.append(prob)
            probabilities.append(prob_row)
            
        return np.array(probabilities)


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.metrics import accuracy_score

    print("--- Training Custom Model on REAL GridGuard Dataset ---")
    
    # 1. Load Real Data
    df = pd.read_excel('dataset/GridGuard_Dataset_2000.xlsx', skiprows=2)

    cat_cols = ['Project_Type', 'Region', 'Land_RoW_Status', 'Forest_Clearance_Status', 'Vendor_Status']
    num_cols = ['Budget_Cr', 'Line_Length_CKM', 'Planned_Duration_Months', 'Physical_Progress_Pct']

    # 2. Encode Labels
    le_risk = LabelEncoder()
    y_clf = le_risk.fit_transform(df['Risk_Level'].values)

    # 3. Features & One-Hot Encoding
    df_encoded = pd.get_dummies(df[num_cols + cat_cols], columns=cat_cols, drop_first=False)
    X = df_encoded.values

    # 4. Train/Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y_clf, test_size=0.2, random_state=42)

    # 5. Scaling (Extremely important for distance-based mathematics)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 6. Initialize our own model
    my_model = CustomGridGuardClassifier(k=5)
    
    # 7. Train it
    my_model.fit(X_train_scaled, y_train)
    
    # 8. Predict on the 400 unseen real test projects
    print("Evaluating...")
    predictions = my_model.predict(X_test_scaled)
    
    # 9. Evaluate Accuracy
    acc = accuracy_score(y_test, predictions)
    print(f"\n[EVALUATION ON REAL TEST DATA]")
    print(f"Custom Mathematical Model Accuracy: {acc*100:.2f}%")
