# Streamlit Deloitte Dashboard (IoT-Inspired with Theme Switch)
import streamlit as st
st.set_page_config(page_title="Smart Surveillance Dashboard", page_icon="📹", layout="wide")

import pandas as pd
import os
import plotly.express as px
from datetime import datetime
from pathlib import Path
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Neo4j Setup ---
NEO4J_URI = os.getenv("url", "bolt://localhost:7687")
NEO4J_USER = os.getenv("username", "neo4j")
NEO4J_PASS = os.getenv("password", "12345678")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

# --- Load Data from Neo4j ---
@st.cache_data
def load_data():
    with driver.session() as session:
        result = session.run("""
            MATCH (v:Video)
            OPTIONAL MATCH (v)-[:CONTAINS]->(a:Activity)
            OPTIONAL MATCH (v)-[:HAS_IS_ALERT]->(al:IsAlert)
            OPTIONAL MATCH (v)-[:HAS_SUMMARY]->(s:Summary)
            RETURN v.name AS video,
                   a.name AS activity_label,
                   al.value AS is_alert,
                   s.text AS summary
        """)
        records = result.data()
        df = pd.DataFrame(records)
        df["is_alert"] = df["is_alert"].astype(str).str.strip().map({"True": True, "False": False})
        return df

df = load_data()

# --- Theme Switch ---
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

# --- Custom CSS ---
st.markdown(f"""
<link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
<style>
    html, body, .main, .block-container {{
        background-color: {THEME['background']};
        color: {THEME['text_primary']};
    }}
    .main-title {{ font-size: 3rem; color: {THEME['accent']}; font-weight: bold; text-align: center; }}
    .subtitle {{ font-size: 1.2rem; color: {THEME['text_secondary']}; text-align: center; }}
    .credit {{ font-size: 0.9rem; color: {THEME['text_secondary']}; text-align: center; margin-top: -0.5rem; }}
    .metric-card {{ background-color: {THEME['panel']}; border-radius: 10px; padding: 1rem 1.5rem; height: 100px; display: flex; flex-direction: column; justify-content: center; }}
    .metric-value {{ font-size: 2rem; color: {THEME['accent']}; font-weight: bold; }}
    .metric-label {{ font-size: 0.8rem; color: {THEME['text_secondary']}; text-transform: uppercase; margin-top: 0.5rem; }}
    .section-title {{ font-size: 1.6rem; font-weight: 700; color: {THEME['accent']}; margin: 2rem 0 1rem; }}
    .video-card {{ background-color: {THEME['panel']}; padding: 1rem; border-radius: 10px; margin-bottom: 1rem; }}
    .alert-badge {{ padding: 0.5rem 1rem; border-radius: 5px; font-weight: 600; color: white; }}
    .alert-yes {{ background-color: #d63031; }}
    .alert-no {{ background-color: {THEME['accent']}; }}
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown(f"""
<div class="main-title"><i class="fas fa-video"></i> Smart Surveillance Dashboard</div>
<p class="subtitle">AI-Powered Video Analytics & Threat Detection</p>
<p class="credit">By Team Data Mavericks</p>
""", unsafe_allow_html=True)

# --- Sidebar Filters ---
with st.sidebar:
    st.markdown(f"<div class='section-title'><i class='fas fa-filter'></i> Filters</div>", unsafe_allow_html=True)
    selected_activity = st.selectbox("Select Activity", ['All'] + list(df['activity_label'].unique()), key="activity")
    alert_filter = st.selectbox("Alert Status", ['All', 'Alert Only', 'No Alert'], key="alert")

    filtered_df = df.copy()
    if selected_activity != 'All':
        filtered_df = filtered_df[filtered_df['activity_label'] == selected_activity]
    if alert_filter == 'Alert Only':
        filtered_df = filtered_df[filtered_df['is_alert'] == True]
    elif alert_filter == 'No Alert':
        filtered_df = filtered_df[filtered_df['is_alert'] == False]

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{len(filtered_df)}</div>
        <div class="metric-label">Filtered Videos</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">{filtered_df['is_alert'].sum()}</div>
        <div class="metric-label">Filtered Alerts</div>
    </div>
    """, unsafe_allow_html=True)

# --- System Metrics ---
st.markdown("<div class='section-title'><i class='fas fa-gauge'></i> System Metrics</div>", unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"<div class='metric-card'><div class='metric-value'>{len(df)}</div><div class='metric-label'>Total Videos</div></div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<div class='metric-card'><div class='metric-value'>{df['is_alert'].sum()}</div><div class='metric-label'>Total Alerts</div></div>", unsafe_allow_html=True)
with col3:
    alert_rate = (df['is_alert'].sum() / len(df) * 100) if len(df) > 0 else 0
    st.markdown(f"<div class='metric-card'><div class='metric-value'>{alert_rate:.1f}%</div><div class='metric-label'>Alert Rate</div></div>", unsafe_allow_html=True)
with col4:
    unique_activities = df['activity_label'].nunique()
    st.markdown(f"<div class='metric-card'><div class='metric-value'>{unique_activities}</div><div class='metric-label'>Activity Types</div></div>", unsafe_allow_html=True)

# --- Charts ---
col1, col2 = st.columns(2)
with col1:
    st.markdown("<div class='section-title'><i class='fas fa-chart-pie'></i> Activity Distribution</div>", unsafe_allow_html=True)
    fig_pie = px.pie(
        df,
        names='activity_label',
        color_discrete_sequence=[THEME['accent'], THEME['secondary'], '#27ae60'],
        hole=0.4
    )
    fig_pie.update_layout(
        paper_bgcolor=THEME['panel'],
        font_color=THEME['text_primary'],
        margin=dict(t=10, b=10, l=10, r=10)
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    st.markdown("<div class='section-title'><i class='fas fa-chart-bar'></i> Alert Counts</div>", unsafe_allow_html=True)
    alert_by_activity = df.groupby('activity_label')['is_alert'].sum().reset_index()
    fig_bar = px.bar(
        alert_by_activity,
        x='activity_label',
        y='is_alert',
        color='activity_label',
        color_discrete_sequence=[THEME['accent'], THEME['secondary'], '#27ae60']
    )
    fig_bar.update_layout(
        paper_bgcolor=THEME['panel'],
        font_color=THEME['text_primary'],
        margin=dict(t=10, b=10, l=10, r=10),
        coloraxis_showscale=False
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# --- Enhanced Helper to find video path recursively ---
@st.cache_data
def find_video_recursive(video_name):
    """
    Recursively search for video files in the entire tracked_videos directory structure
    """
    # Search in all possible directories (train, val, test)
    search_dirs = [
        Path("tracked_videos/test"),
        Path("tracked_videos/train"), 
        Path("tracked_videos/val"),
        Path("tracked_videos")  # Also search root in case structure is different
    ]
    
    # Remove file extension for comparison if present
    video_name_clean = Path(video_name).stem
    
    for base_dir in search_dirs:
        if base_dir.exists():
            # Use rglob to search recursively through all subdirectories
            for video_path in base_dir.rglob("*.mp4"):
                # Compare both with and without extension
                if (video_path.name == video_name or 
                    video_path.stem == video_name_clean or
                    video_path.name == video_name_clean + ".mp4"):
                    return video_path
    
    return None

# --- Alternative function to get video name variants ---
def get_video_name_variants(video_name):
    """
    Generate different possible variants of the video name to search for
    """
    variants = []
    
    # Original name
    variants.append(video_name)
    
    # With .mp4 extension if not present
    if not video_name.endswith('.mp4'):
        variants.append(video_name + '.mp4')
    
    # Without extension
    variants.append(Path(video_name).stem)
    
    # Handle common naming patterns
    base_name = Path(video_name).stem
    variants.extend([
        base_name + '_tracked.mp4',
        base_name + '_x264_tracked.mp4',
        base_name.replace('_tracked', '') + '.mp4'
    ])
    
    return list(set(variants))  # Remove duplicates

# --- Video Analysis Section ---
st.markdown(f"<div class='section-title'><i class='fas fa-clapperboard'></i> Video Analysis</div>", unsafe_allow_html=True)

# Debug: Show what videos we're looking for
if st.checkbox("Show Debug Info", key="debug"):
    st.write("**Video names from database:**")
    for video_name in df['video'].unique():
        st.write(f"- {video_name}")

for _, row in filtered_df.iterrows():
    st.markdown('<div class="video-card">', unsafe_allow_html=True)
    col1, col2 = st.columns([2, 3])
    
    with col1:
        try:
            video_name = row['video']
            
            # Try to find the video using recursive search
            video_path = find_video_recursive(video_name)
            
            if video_path and video_path.exists():
                st.video(str(video_path.resolve()))
                if st.checkbox("Show Video Path", key=f"path_{video_name}"):
                    st.info(f"Found at: {video_path}")
            else:
                # If not found, show all variants we tried
                st.error(f"⚠️ Video not found: {video_name}")
                
                if st.checkbox("Show Search Details", key=f"search_{video_name}"):
                    variants = get_video_name_variants(video_name)
                    st.write("**Searched for these variants:**")
                    for variant in variants:
                        st.write(f"- {variant}")
                    
                    # Show what files actually exist in the directories
                    st.write("**Available .mp4 files:**")
                    all_videos = []
                    for base_dir in [Path("tracked_videos/test"), Path("tracked_videos/train"), Path("tracked_videos/val")]:
                        if base_dir.exists():
                            for video_file in base_dir.rglob("*.mp4"):
                                all_videos.append(str(video_file))
                    
                    for video_file in sorted(all_videos)[:10]:  # Show first 10 to avoid clutter
                        st.write(f"- {video_file}")
                    if len(all_videos) > 10:
                        st.write(f"... and {len(all_videos) - 10} more files")
                        
        except Exception as e:
            st.error(f"⚠️ Error loading video: {e}")

    with col2:
        st.markdown(f"**File:** {os.path.basename(row['video'])}")
        st.markdown(f"**Activity:** {row['activity_label']}")
        status = "ALERT TRIGGERED" if row['is_alert'] else "SECURE"
        badge = "alert-yes" if row['is_alert'] else "alert-no"
        st.markdown(f"<span class='alert-badge {badge}'>{status}</span>", unsafe_allow_html=True)
        st.markdown(f"<div class='modern-text'><strong>AI Summary:</strong> {row['summary']}</div>", unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# --- Footer ---
st.markdown(f"""
<div style="text-align: center; padding: 1rem; color: {THEME['text_secondary']}; font-size: 0.85rem;">
    Deloitte Smart Surveillance Dashboard © {datetime.now().year} | Team Data Mavericks
</div>
""",unsafe_allow_html=True)