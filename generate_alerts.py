import pandas as pd

df = pd.read_csv("activity_predictions.csv")
df = df.dropna(subset=["predicted_class_id"])
df["predicted_class_id"] = df["predicted_class_id"].astype(int)

label_map = {
    5: "stealing", 8: "sitting", 10: "running", 60: "vandalizing",
    62: "carcrash", 87: "stealing", 92: "carcrash", 103: "carcrash",
    220: "stealing", 235: "carcrash", 254: "vandalizing", 257: "carcrash",
    313: "vandalizing", 326: "carcrash", 327: "vandalizing", 328: "carcrash",
    358: "vandalizing", 391: "vandalizing"
}

df["activity_label"] = df["predicted_class_id"].map(label_map)
df["is_alert"] = df["activity_label"].isin(["stealing", "vandalizing", "carcrash", "kicking"])
df.to_csv("alert_results.csv", index=False)
print("✅ alert_results.csv generated")