import pandas as pd
import random

# 1. Load the dataset you downloaded
df = pd.read_csv('construction_project_dataset.csv')

# 2. Define realistic templates for vendor/engineer logs (Infrastructure/PowerGrid focus)
delayed_remarks = [
    "Vendor reported a 3-week delay in high-voltage transformer delivery due to supply chain issues.",
    "Pending environmental clearance from local municipality delaying site excavation.",
    "Severe cement shortage from the primary supplier, holding up the foundation pouring.",
    "Subcontractor labor dispute causing a significant backlog in tower erection.",
    "Unexpected heavy rainfall halted crane operations and delayed structural assembly.",
    "Quality inspection failed for the imported steel cables; waiting for replacements.",
    "Permit renewal for heavy machinery transportation is currently stalled."
]

on_time_remarks = [
    "Routine site inspection completed. Concrete pouring proceeding as scheduled.",
    "All transmission line materials arrived on site; workers are proceeding on schedule.",
    "Vendor delivery confirmed on time. All quality checks passed successfully.",
    "Phase milestone achieved without any resource or labor bottlenecks.",
    "Optimal weather conditions allowing for uninterrupted grid stringing work.",
    "No safety incidents reported. Equipment utilization is running at maximum efficiency.",
    "Logistics and supply chain operating smoothly; inventory levels are adequate."
]

# 3. Create a function to generate the text based on the numerical 'time_deviation'
def generate_remark(time_dev):
    # If time_dev is greater than 0, it means the project took longer than expected (Delayed)
    if time_dev > 0:
        return random.choice(delayed_remarks)
    else:
        return random.choice(on_time_remarks)

# 4. Apply the function to create your new unstructured text column
df['Vendor_Remarks'] = df['time_deviation'].apply(generate_remark)

# 5. Save your final Hybrid AI dataset!
df.to_csv('GridGuard_Hybrid_Dataset.csv', index=False)
print("Dataset successfully generated!")