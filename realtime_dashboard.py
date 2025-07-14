import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from groq import Groq

# === Load API Key from .env ===
load_dotenv()

os.environ["GROQ_API_KEY"] = "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"

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

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

# === Page Setup ===
st.set_page_config(page_title="Real-Time Surveillance", page_icon="📹", layout="wide")

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

# === Load Data ===
df = pd.read_csv("activity_predictions.csv")

label_map = {
    5: "stealing", 8: "sitting", 10: "running", 60: "vandalizing",
    62: "carcrash", 87: "stealing", 92: "carcrash", 103: "carcrash",
    220: "stealing", 235: "carcrash", 254: "vandalizing", 257: "carcrash",
    313: "vandalizing", 326: "carcrash", 327: "vandalizing", 328: "carcrash",
    358: "vandalizing", 391: "vandalizing"
}
anomaly_labels = ["stealing", "carcrash", "vandalizing", "kicking"]

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

# === Display Cards ===
for idx, row in df.iterrows():
    video_path = row["video"].replace(".avi", ".mp4")
    class_id = row["predicted_class_id"]

    if pd.isna(class_id):
        continue

    class_id = int(class_id)
    label = label_map.get(class_id, "unknown")
    is_alert = label in anomaly_labels

    # Extract metadata
    filename = os.path.basename(video_path)
    parts = filename.split(".")
    date = parts[1] if len(parts) > 2 else "N/A"
    time = parts[2] if len(parts) > 3 else "N/A"
    location = parts[3] if len(parts) > 4 else "N/A"

    st.markdown("<div class='video-card'>", unsafe_allow_html=True)
    col1, col2 = st.columns([2, 3])

    with col1:
        st.video(video_path)
        st.markdown(f"**Detected Activity:** {label}")
        badge = "alert-yes" if is_alert else "alert-no"
        status = "🚨 Anomaly Detected" if is_alert else "✅ Normal Activity"
        st.markdown(f"<span class='alert-badge {badge}'>{status}</span>", unsafe_allow_html=True)

    with col2:
        st.subheader("🧠 AI Summary & Recommendation")
        with st.spinner("Generating AI summary..."):
            summary_text = generate_groq_report(label, is_alert, date, time, location)
            summary_lines = summary_text.strip().split("\n")

            st.markdown("**🗓️ Event Info:**")
            st.markdown(f"- Date: {date}")
            st.markdown(f"- Time: {time}")
            st.markdown(f"- Location: {location}")

            st.markdown("**📽️ Summary:**")
            for line in summary_lines:
                clean_line = line.strip()
                if "✅ Recommendation" not in clean_line:
                    st.markdown(f"- {clean_line}")

            rec_line = [line for line in summary_lines if "✅ Recommendation" in line]
            if rec_line:
                st.markdown("**✅ Recommendation:**")
                st.markdown(f"- {rec_line[0].split('✅ Recommendation:')[-1].strip()}")

    st.markdown("</div>", unsafe_allow_html=True)

# === Footer ===
st.markdown(f"""
<div style="text-align: center; padding: 1rem; color: {THEME['text_secondary']}; font-size: 0.85rem;">
    Smart Surveillance © {datetime.now().year} | Real-Time AI Monitoring
</div>
""", unsafe_allow_html=True)