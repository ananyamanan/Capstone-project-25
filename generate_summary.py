import pandas as pd

# Load alert results (already cleaned & standardized)
df = pd.read_csv("alert_summary.csv")

# Function to create natural language summary
def generate_summary(row):
    if pd.isna(row["activity_label"]):
        return "⚠️ No confident activity detected."
    elif row["is_alert"]:
        return f"🔴 Alert: '{row['activity_label']}' detected in this video. Needs review."
    else:
        return f"🟢 Normal behavior: '{row['activity_label']}' detected."

# Apply summary generation
df["summary"] = df.apply(generate_summary, axis=1)

# Save the enriched summary file (overwriting previous one)
df.to_csv("alert_summary.csv", index=False)

print("✅ alert_summary.csv updated with natural-language summaries.")