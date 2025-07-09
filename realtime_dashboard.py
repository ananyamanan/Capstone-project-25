import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from groq import Groq

# === Load API Key from .env ===
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# === Streamlit App Setup ===
st.set_page_config(page_title="Smart Surveillance", layout="wide")
st.title("🎥 Real-Time Surveillance Dashboard")
st.markdown("This dashboard simulates real-time alerts, summaries, and recommendations using Groq API.")

# === Load Predictions CSV ===
df = pd.read_csv("activity_predictions.csv")

# === Map Class IDs to Activity Labels ===
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

# === Display Dashboard ===
for idx, row in df.iterrows():
    video_path = row["video"].replace(".avi", ".mp4")
    class_id = row["predicted_class_id"]

    if pd.isna(class_id):
        continue

    class_id = int(class_id)
    label = label_map.get(class_id, "unknown")
    is_alert = label in anomaly_labels

    # Extract metadata from filename
    filename = os.path.basename(video_path)
    parts = filename.split(".")
    date = parts[1] if len(parts) > 2 else "N/A"
    time = parts[2] if len(parts) > 3 else "N/A"
    location = parts[3] if len(parts) > 4 else "N/A"

    col1, col2 = st.columns([2, 3])

    # === Left: Video + Alert Tag ===
    with col1:
        st.video(video_path)
        st.caption(f"📌 Detected Activity: {label}")
        if is_alert:
            st.markdown("<span style='color:red; font-size:20px;'>🚨 Anomaly Detected</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span style='color:green; font-size:16px;'>✅ Normal Activity</span>", unsafe_allow_html=True)

    # === Right: AI Summary Section ===
    with col2:
        st.subheader("🧠 AI Summary & Recommendation")

        with st.spinner("Generating Groq response..."):
            summary_text = generate_groq_report(label, is_alert, date, time, location)
            summary_lines = summary_text.strip().split("\n")

            # === Event Info ===
            st.markdown("**🗓️ Event Info:**")
            st.markdown(f"Date- {date}, Time- {time}, Location- {location}")
            st.markdown("")

            # === Summary ===
            st.markdown("**📽️ Summary:**")
            for line in summary_lines:
                clean_line = line.strip()

                if line.strip().lower().startswith("✅ recommendation"):
                    continue  # skip recommendation lines
                if "✅ recommendation" in line.lower():
                    line = line.split("✅ recommendation")[0].strip()
                if clean_line:
                    st.markdown(f"- {clean_line}")

            # === Recommendation ===
            st.markdown("")
            st.markdown("**✅ Recommendation:**")
            st.markdown("- Review footage for identification and notify authorities.")

    st.markdown("---")