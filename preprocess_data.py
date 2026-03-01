import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

# 1. Load your newly generated hybrid dataset
df = pd.read_csv('./dataset/GridGuard_Hybrid_Dataset.csv')

# 2. Define the Target Variable (What we want to predict)
# If time_deviation > 0, it's a Risk (1). Otherwise, Safe (0).
df['Risk_Flag'] = (df['time_deviation'] > 0).astype(int)

# 3. Process the Text Data (NLP)
# Initialize TF-IDF Vectorizer (limiting to top 100 keywords to keep it efficient)
tfidf = TfidfVectorizer(max_features=100, stop_words='english')
text_features = tfidf.fit_transform(df['Vendor_Remarks']).toarray()

# Create a DataFrame for the text features
text_df = pd.DataFrame(text_features, columns=tfidf.get_feature_names_out())

# 4. Process the Numerical Data
# Select key numerical columns for the risk radar
num_cols = ['cost_deviation', 'worker_count', 'equipment_utilization_rate', 'material_usage']
scaler = StandardScaler()
scaled_nums = scaler.fit_transform(df[num_cols])

# Create a DataFrame for scaled numerical features
num_df = pd.DataFrame(scaled_nums, columns=num_cols)

# 5. Combine into the Final Hybrid Feature Set (X) and Target (y)
X = pd.concat([num_df, text_df], axis=1)
y = df['Risk_Flag']

print("Phase 2 Complete! The data is now fully numerical and scaled.")
print(f"Shape of our final feature set (X): {X.shape}")
print(f"Number of Risk Flags (1) vs Safe (0):\n{y.value_counts()}")