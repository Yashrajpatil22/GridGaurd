import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("Loading Grid-Guard models to analyze the AI's brain...")

# 1. Load the trained model and the TF-IDF vectorizer
model = joblib.load('models/gridguard_model.pkl')
tfidf = joblib.load('models/tfidf_vectorizer.pkl')

# 2. Define the exact numerical columns we used
num_cols = ['cost_deviation', 'worker_count', 'equipment_utilization_rate', 'material_usage']

# 3. Extract the 100 specific text keywords the AI learned from the vendor logs
text_cols = tfidf.get_feature_names_out()

# 4. Combine them in the exact order the model was trained (Numbers first, then Text)
all_features = list(num_cols) + list(text_cols)

# 5. Extract the "Importance Scores" directly from the Random Forest
importances = model.feature_importances_

# 6. Create a DataFrame to sort them from highest impact to lowest
feature_df = pd.DataFrame({
    'Feature': all_features,
    'Importance': importances
})

# Sort by most important
feature_df = feature_df.sort_values(by='Importance', ascending=False)

# 7. Plot the Top 15 Most Important Features
plt.figure(figsize=(12, 8))
# Using seaborn to make it look highly professional and modern
sns.barplot(x='Importance', y='Feature', data=feature_df.head(15), palette='viridis')

# Format the graph
plt.title('Grid-Guard AI: Top 15 Drivers of Infrastructure Delay', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Impact Level on Final Prediction', fontsize=12, fontweight='bold')
plt.ylabel('Project Parameter / Log Keyword', fontsize=12, fontweight='bold')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()

# 8. Save it as a high-resolution image for your presentation
plt.savefig('feature_importance_graph.png', dpi=300)
print("Success! Graph saved as 'feature_importance_graph.png' in your folder.")

# Show it on the screen right now
plt.show()