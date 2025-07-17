import pandas as pd
from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

load_dotenv()

# Neo4j connection details
NEO4J_URI = os.getenv("url", "bolt://localhost:7687")
NEO4J_USER = os.getenv("username", "neo4j")
NEO4J_PASS = os.getenv("password", "12345678")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

def clear_existing_graph(tx):
    tx.run("MATCH (n) DETACH DELETE n")

def create_activity_node(tx, row):
    tx.run("""
        MERGE (a:Activity {video: $video})
        SET a.group = $group,
            a.folder_label = $folder_label,
            a.predicted_class_id = $predicted_class_id
    """, **row)

def create_alert_node(tx, row):
    tx.run("""
        MERGE (al:Alert {video: $video})
        SET al.activity_label = $activity_label,
            al.is_alert = $is_alert
    """, **row)

def create_summary_node(tx, row):
    tx.run("""
        MERGE (s:Summary {video: $video})
        SET s.summary = $summary
    """, **row)

def create_relationships(tx):
    tx.run("""
        MATCH (a:Activity), (al:Alert)
        WHERE a.video = al.video
        MERGE (a)-[:TRIGGERED]->(al)
    """)
    tx.run("""
        MATCH (al:Alert), (s:Summary)
        WHERE al.video = s.video
        MERGE (al)-[:SUMMARIZED_BY]->(s)
    """)

def ingest_to_neo4j():
    with driver.session() as session:
        # Optional: clear old data
        session.execute_write(clear_existing_graph)

        # ✅ Load CSVs using exact file names from your directory
        activity_df = pd.read_csv("activity_predictions.csv")
        alerts_df = pd.read_csv("alert_results.csv")
        summary_df = pd.read_csv("alert_summary.csv")

        # Insert nodes
        for _, row in activity_df.iterrows():
            session.execute_write(create_activity_node, row.to_dict())
        for _, row in alerts_df.iterrows():
            session.execute_write(create_alert_node, row.to_dict())
        for _, row in summary_df.iterrows():
            session.execute_write(create_summary_node, row.to_dict())

        # Link nodes
        session.execute_write(create_relationships)

    print("✅ Data inserted and connected in Neo4j.")

if __name__ == "__main__":
    ingest_to_neo4j()