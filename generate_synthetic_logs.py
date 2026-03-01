import pandas as pd
import random

print("Loading original dataset...")
# Load the raw numerical dataset
df = pd.read_csv('./dataset/construction_project_dataset.csv')

# --- EXPANDED DELAYED REMARKS (Risk Hotspots) ---
delayed_remarks = [
    "Vendor reported a 3-week delay in high-voltage transformer delivery due to supply chain issues.",
    "Severe cement shortage from the primary supplier, holding up the foundation pouring.",
    "Imported steel cables stuck at customs; awaiting clearance documentation.",
    "Logistics partner vehicle broke down in transit, delaying the arrival of switchgear components.",
    "Global semiconductor shortage is delaying the delivery of the automated control panels.",
    "Pending environmental clearance from local municipality delaying site excavation.",
    "Permit renewal for heavy machinery transportation is currently stalled.",
    "Local protests regarding land acquisition have temporarily halted right-of-way clearing.",
    "Awaiting final approval from the State Electricity Board for the new routing plan.",
    "Subcontractor labor dispute causing a significant backlog in tower erection.",
    "High attrition rate among skilled welders is slowing down the structural assembly.",
    "Unexpected regional holiday resulted in a 2-day labor shortage on site.",
    "Key engineering supervisor on medical leave; temporary replacement is taking time to onboard.",
    "Quality inspection failed for the initial batch of insulators; waiting for replacements.",
    "Main heavy-lifting crane requires emergency maintenance; operations paused for 48 hours.",
    "Safety audit flagged the temporary scaffolding; rework required before proceeding.",
    "Concrete mixture failed the 7-day compression test; foundation phase needs to be redone.",
    "Unexpected heavy rainfall halted crane operations and delayed structural assembly.",
    "Severe heatwave conditions forced a reduction in afternoon working hours for safety.",
    "Waterlogging at the primary substation site is preventing heavy machinery access."
]

# --- EXPANDED ON-TIME REMARKS (Smooth Operations) ---
on_time_remarks = [
    "Routine site inspection completed. Concrete pouring proceeding as scheduled.",
    "Phase 2 milestone achieved without any resource or labor bottlenecks.",
    "Project is moving exactly according to the baseline Gantt chart.",
    "Weekly progress review shows all sub-teams are meeting their targets.",
    "All transmission line materials arrived on site; workers are proceeding on schedule.",
    "Vendor delivery confirmed on time. All quality checks passed successfully.",
    "Logistics and supply chain operating smoothly; inventory levels are adequate.",
    "Advance procurement strategy successful; no material shortages anticipated.",
    "No safety incidents reported. Equipment utilization is running at maximum efficiency.",
    "Third-party quality assurance audit passed with zero critical remarks.",
    "Preventative maintenance on heavy machinery completed without impacting the schedule.",
    "All sensor calibrations at the substation passed successfully.",
    "Optimal weather conditions allowing for uninterrupted grid stringing work.",
    "Local municipality expedited the road-closure permits for transformer transport.",
    "Excellent coordination with local authorities ensuring smooth site operations.",
    "Labor turnout is at 100%; productivity metrics are exceeding expectations."
]

# Function to generate the synthetic text WITH NOISE
def generate_remark_with_noise(time_dev):
    # 1. Determine the actual truth
    actual_is_delayed = time_dev > 0
    
    # 2. Introduce 15% Noise (Real-world chaos)
    # There is a 15% chance the text log CONTRADICTS the actual numerical outcome
    if random.random() < 0.15: 
        apparent_is_delayed = not actual_is_delayed # Flip the reality!
    else:
        apparent_is_delayed = actual_is_delayed
        
    # 3. Assign the remark based on the (potentially noisy) apparent status
    if apparent_is_delayed:
        return random.choice(delayed_remarks)
    else:
        return random.choice(on_time_remarks)

# Apply the function to create the new text column
print("Generating realistic synthetic text logs with 15% noise...")
df['Vendor_Remarks'] = df['time_deviation'].apply(generate_remark_with_noise)

# Save the final Hybrid AI dataset, overwriting the old perfect one
df.to_csv('./dataset/GridGuard_Hybrid_Dataset.csv', index=False)
print("Realistic dataset successfully generated and saved!")