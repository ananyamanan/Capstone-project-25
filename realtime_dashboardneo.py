# Streamlit Deloitte Dashboard (Real-Time Surveillance)
import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from groq import Groq
from datetime import datetime
from pathlib import Path
from neo4j import GraphDatabase
import atexit

# === Streamlit Page Config ===
st.set_page_config(page_title="Real-Time Surveillance", page_icon="📹", layout="wide")

# === Load Environment Variables ===
load_dotenv()

# === Groq Client ===
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# === Neo4j Credentials ===
NEO4J_URI = os.getenv("url", "bolt://localhost:7687")
NEO4J_USER = os.getenv("username", "neo4j")
NEO4J_PASS = os.getenv("password", "12345678")

# === Initialize Neo4j Driver ===
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
atexit.register(driver.close)  # Close on app exit

# === Theme Switch ===
theme_mode = st.sidebar.radio("Theme", ["Dark", "Light"], index=0)

LIGHT_THEME = {
    'background': '#f5f5f5',
    'panel': '#ffffff',
    'accent': '#86BC25',
    'secondary': '#0076A8',
    'text_primary': '#1e1e1e',
    'text_secondary': '#666666'
}

DARK_THEME = {
    'background': '#1e1e1e',
    'panel': '#2a2a2a',
    'accent': '#86BC25',
    'secondary': '#00b894',
    'text_primary': '#ffffff',
    'text_secondary': '#aaaaaa'
}

THEME = DARK_THEME if theme_mode == "Dark" else LIGHT_THEME

# === Custom CSS ===
st.markdown(f"""
<link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
<style>
    html, body, .main, .block-container {{
        background-color: {THEME['background']};
        color: {THEME['text_primary']};
    }}
    .main-title {{ font-size: 3rem; color: {THEME['accent']}; font-weight: bold; text-align: center; }}
    .video-card {{ background-color: {THEME['panel']}; padding: 1.2rem; border-radius: 12px; margin-bottom: 1.5rem; }}
    .alert-badge {{ padding: 0.5rem 1rem; border-radius: 5px; font-weight: 600; color: white; }}
    .alert-yes {{ background-color: #d63031; }}
    .alert-no {{ background-color: {THEME['accent']}; }}
    .section-title {{ font-size: 1.6rem; font-weight: 700; color: {THEME['accent']}; margin: 2rem 0 1rem; }}
</style>
""", unsafe_allow_html=True)

# === Canonical Label Map ===
LABEL_MAP = {
    'carcrash': 0,
    'kicking': 1,
    'stealing': 2,
    'vandalizing': 3,
    'neutral': 4,
    'running': 5,
    'sitting': 6
}
INVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}
ANOMALY_LABELS = ["stealing", "carcrash", "vandalizing", "kicking"]

# === Load Data from Neo4j ===
@st.cache_data
def load_from_neo4j():
    with driver.session() as session:
        result = session.run("""
            MATCH (v:Video)
            OPTIONAL MATCH (v)-[:CONTAINS]->(a:Activity)
            OPTIONAL MATCH (v)-[:HAS_IS_ALERT]->(al:IsAlert)
            OPTIONAL MATCH (v)-[:HAS_SUMMARY]->(s:Summary)
            RETURN v.name AS video,
                   a.name AS predicted_label,
                   al.value AS is_alert,
                   s.text AS summary
        """)
        records = result.data()
        df = pd.DataFrame(records)
        df["is_alert"] = df["is_alert"].astype(str).str.strip().map({"True": True, "False": False})
        return df

df = load_from_neo4j()

# === Groq Summary Generator ===
@st.cache_data(show_spinner=False)
def generate_groq_report(label, is_alert, date="N/A", time="N/A", location="N/A"):
    prompt = f"""
You are a security analyst summarizing surveillance footage.

Information provided:
- Activity: '{label}'
- Alert Triggered: {'Yes' if is_alert else 'No'}"""

    if date != "N/A":
        prompt += f"\n- Date: {date}"
    if time != "N/A":
        prompt += f"\n- Time: {time}"
    if location != "N/A":
        prompt += f"\n- Location: {location}"

    prompt += """

🔹 Write a 3–4 line **factual** summary of what is happening in the video.
🔹 Use ONLY the given information. Do NOT make up extra context.
🔹 End your answer with: '✅ Recommendation: <action to take>'
"""
    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt.strip()}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Groq error:\n{str(e)}"

# === Header ===
st.markdown(f"<div class='main-title'><i class='fas fa-satellite'></i> Real-Time Surveillance Dashboard</div>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Live Activity Detection & AI Summaries Powered by Groq</p>", unsafe_allow_html=True)

# === Video Path Resolver ===
def find_video_recursive(video_name):
    search_dirs = [
        Path("tracked_videos/test"),
        Path("tracked_videos/train"),
        Path("tracked_videos/val"),
        Path("tracked_videos")
    ]
    for base_dir in search_dirs:
        if base_dir.exists():
            for video_path in base_dir.rglob("*.mp4"):
                if video_path.name == video_name or video_path.stem == video_name:
                    return video_path
    return None

# === Display Each Video Card ===
for index, row in df.iterrows():
    video_name = row["video"]
    label = row.get("predicted_label", "unknown")
    label = label.lower() if isinstance(label, str) else "unknown"
    is_alert = label in ANOMALY_LABELS
    # Generate Groq Summary Dynamically
    summary = generate_groq_report(
    label=label,
    is_alert=is_alert,
    date="N/A",  # Replace with real date if available
    time="N/A",  # Replace with real time if available
    location="N/A"  # Replace with real location if available
)


    video_path = find_video_recursive(video_name)
    if not video_path or not video_path.exists():
        continue

    with st.container():
        cols = st.columns([1, 2])
        with cols[0]:
            st.video(str(video_path))
        with cols[1]:
            st.markdown(f"**Video Name:** `{Path(video_path).name}`")
            st.markdown(f"**Predicted Label:** `{label}`")
            st.markdown(f"**ALERT:** {'⚠️ Yes' if is_alert else '✅ No'}")
            st.markdown(f"**Groq Summary:**\n{summary}")

# === Footer ===
st.markdown(f"""
<div style="text-align: center; padding: 1rem; color: {THEME['text_secondary']}; font-size: 0.85rem;">
    Smart Surveillance © {datetime.now().year} | Real-Time AI Monitoring
</div>
""", unsafe_allow_html=True)