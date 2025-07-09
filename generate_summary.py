import pandas as pd

df = pd.read_csv("alert_results.csv")

def generate_summary(row):
    if pd.isna(row["activity_label"]):
        return "⚠️ No confident activity detected."
    elif row["is_alert"]:
        return f"🔴 Alert: '{row['activity_label']}' detected in this video. Needs review."
    else:
        return f"🟢 Normal behavior: '{row['activity_label']}' detected."

df["summary"] = df.apply(generate_summary, axis=1)
df.to_csv("alert_summary.csv", index=False)
print("✅ alert_summary.csv generated with natural-language descriptions")