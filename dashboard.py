import streamlit as st
import pandas as pd
import os

st.set_page_config(layout="wide")
st.title("🎥 Smart Surveillance Dashboard")
df = pd.read_csv("alert_summary.csv")

st.metric("Total Videos Processed", len(df))
st.metric("Total Alerts", df["is_alert"].sum())

st.markdown("---")

for idx, row in df.iterrows():
    col1, col2 = st.columns([2, 3])
    
    with col1:
        print(row["video"])  # Debug path
        video_path = row["video"].replace(".avi", ".mp4")
        print(video_path)  # For debugging
        st.video(video_path)

    
    with col2:
        st.subheader("Summary")
        st.write(f"**Video:** {os.path.basename(row['video'])}")
        st.write(f"**Activity Label:** {row['activity_label']}")
        st.write(f"**Alert Triggered:** {'Yes' if row['is_alert'] else 'No'}")
        st.write(f"📝 {row['summary']}")
    
    st.markdown("---")