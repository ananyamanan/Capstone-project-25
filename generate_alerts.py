import pandas as pd
import os

# Load predicted results from correct path
df = pd.read_csv("activity_predictions/fine_tuned_results.csv")

# Remove any rows where the predicted_label is missing
df = df.dropna(subset=["predicted_label"])

# Rename the column to match dashboard logic
df["activity_label"] = df["predicted_label"].str.lower().str.strip()

# Define which activities are considered alerts
alert_activities = ["stealing", "vandalizing", "carcrash", "kicking"]

# Mark alert videos
df["is_alert"] = df["activity_label"].isin(alert_activities)

# Normalize video names (just filename and .mp4 extension)
df["video"] = df["video"].apply(lambda x: os.path.basename(x).replace(".avi", ".mp4"))

# Save for Streamlit dashboard
df.to_csv("alert_summary.csv", index=False)
print("✅ alert_summary.csv generated successfully from fine_tuned_results.csv")