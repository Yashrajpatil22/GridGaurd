import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the dataset (fallback to 70k if 50k doesn't exist depending on user's current testing state)
dataset_path = "dataset/GridGuard_Dataset_70000.csv"
if not os.path.exists(dataset_path):
    dataset_path = "dataset/GridGuard_Dataset_50000.csv"

df = pd.read_csv(dataset_path)

# Set stylistic parameters
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

output_dir = r"C:\Users\User\.gemini\antigravity\brain\7f70fbe3-d8c9-4862-9561-a6a58e61247c"

# ────────────────────────────────────────────────────────────────────────
# Graph 1: Distribution of Actual Delay by Risk Level
# ────────────────────────────────────────────────────────────────────────
plt.figure(figsize=(10, 6))
sns.kdeplot(
    data=df, 
    x="Actual_Delay_Months", 
    hue="Risk_Level", 
    fill=True, 
    common_norm=False, 
    palette={"Low": "green", "Medium": "orange", "High": "red"},
    alpha=0.4,
    linewidth=2
)
plt.title("Exponential Expansion of Delays in High-Risk Projects", fontsize=16, fontweight='bold', pad=15)
plt.xlabel("Actual Delay (Months)", fontsize=14)
plt.ylabel("Density", fontsize=14)
plt.xlim(0, 60)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "graph1_delay_distribution.png"), dpi=300)
plt.close()

# ────────────────────────────────────────────────────────────────────────
# Graph 2: The "Deceptive 100%" Edge Case Problem
# ────────────────────────────────────────────────────────────────────────
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df.sample(2000, random_state=42),  # Sample for visibility
    x="Physical_Progress_Pct", 
    y="Actual_Delay_Months", 
    hue="Risk_Level", 
    palette={"Low": "green", "Medium": "orange", "High": "red"},
    size="Budget_Cr",
    sizes=(20, 300),
    alpha=0.7,
    edgecolor="black",
    linewidth=0.5
)
plt.title("Physical Progress vs. Actual Delay (The 100% Trap)", fontsize=16, fontweight='bold', pad=15)
plt.xlabel("Physical Progress (%)", fontsize=14)
plt.ylabel("Actual Delay (Months)", fontsize=14)

# Highlight the extreme outlier zone that NLP solves
plt.axvspan(95, 102, ymin=0.3, ymax=1.0, color='red', alpha=0.1)
plt.text(70, 35, "← Extreme Outliers\n(100% complete but 20+ Mo Delay)", color='red', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "graph2_progress_outliers.png"), dpi=300)
plt.close()

print("Graphs generated successfully!")
