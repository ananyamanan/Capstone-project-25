from py2neo import Graph, Node, Relationship
import pandas as pd

graph = Graph("bolt://localhost:7687", auth=("neo4j", "12345678"))

graph.run("MATCH (n) DETACH DELETE n")
alert_df = pd.read_csv("alert_summary.csv")

video_count = 0

for _, row in alert_df.iterrows():
    video_name = row["video"]
    if pd.isna(video_name):
        print("⚠️ Skipping row with no video name.")
        continue

    try:
        video_node = Node("Video", name=video_name)
        graph.merge(video_node, "Video", "name")
        video_count += 1

        if pd.notna(row["activity_label"]):
            activity_node = Node("Activity", name=row["activity_label"])
            graph.merge(activity_node, "Activity", "name")
            graph.merge(Relationship(video_node, "CONTAINS", activity_node))

        if pd.notna(row["true_label"]):
            true_label_node = Node("TrueLabel", value=row["true_label"])
            graph.merge(true_label_node, "TrueLabel", "value")
            graph.merge(Relationship(video_node, "HAS_TRUE_LABEL", true_label_node))

        if pd.notna(row["predicted_label"]):
            predicted_label_node = Node("PredictedLabel", value=row["predicted_label"])
            graph.merge(predicted_label_node, "PredictedLabel", "value")
            graph.merge(Relationship(video_node, "HAS_PREDICTED_LABEL", predicted_label_node))

        if pd.notna(row["is_alert"]):
            is_alert_node = Node("IsAlert", value=str(row["is_alert"]))
            graph.merge(is_alert_node, "IsAlert", "value")
            graph.merge(Relationship(video_node, "HAS_IS_ALERT", is_alert_node))

        if pd.notna(row["summary"]):
            summary_node = Node("Summary", text=row["summary"])
            graph.merge(summary_node, "Summary", "text")
            graph.merge(Relationship(video_node, "HAS_SUMMARY", summary_node))

        print(f"✅ Processed: {video_name}")

    except Exception as e:
        print(f"❌ Error processing video {video_name}: {e}")

print(f"✅ Total videos processed: {video_count}")