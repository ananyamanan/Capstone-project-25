# Streamlit Deloitte Dashboard (IoT-Inspired with Theme Switch)
import streamlit as st
import pandas as pd
import os
import plotly.express as px
from datetime import datetime

# --- Theme Switch ---
theme_mode = st.sidebar.radio("Theme", ["Dark", "Light"], index=0)

# --- Theme Colors ---
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

# Page setup
st.set_page_config(page_title="Smart Surveillance Dashboard", page_icon="📹", layout="wide")

# Custom CSS
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

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("alert_summary.csv")

df = load_data()

# Header
st.markdown(f"""
<div class="main-title"><i class="fas fa-video"></i> Smart Surveillance Dashboard</div>
<p class="subtitle">AI-Powered Video Analytics & Threat Detection</p>
<p class="credit">By Team Data Mavericks</p>
""", unsafe_allow_html=True)

# Sidebar Filters
with st.sidebar:
    st.markdown(f"<div class='section-title'><i class='fas fa-filter'></i> Filters</div>", unsafe_allow_html=True)
    selected_activity = st.selectbox("Select Activity", ['All'] + list(df['activity_label'].unique()), key="activity")
    alert_filter = st.selectbox("Alert Status", ['All', 'Alert Only', 'No Alert'], key="alert")

    # Show stats based on filters
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

# System Metrics
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

# Charts
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

# Videos
st.markdown(f"<div class='section-title'><i class='fas fa-clapperboard'></i> Video Analysis</div>", unsafe_allow_html=True)
for _, row in filtered_df.iterrows():
    st.markdown('<div class="video-card">', unsafe_allow_html=True)
    col1, col2 = st.columns([2, 3])
    with col1:
        try:
            st.video(row['video'].replace(".avi", ".mp4"))
        except:
            st.error("Video not found.")
    with col2:
        st.markdown(f"**File:** {os.path.basename(row['video'])}")
        st.markdown(f"**Activity:** {row['activity_label']}")
        status = "ALERT TRIGGERED" if row['is_alert'] else "SECURE"
        badge = "alert-yes" if row['is_alert'] else "alert-no"
        st.markdown(f"<span class='alert-badge {badge}'>{status}</span>", unsafe_allow_html=True)
        st.markdown(f"<div class='modern-text'><strong>AI Summary:</strong> {row['summary']}</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown(f"""
<div style="text-align: center; padding: 1rem; color: {THEME['text_secondary']}; font-size: 0.85rem;">
    Deloitte Smart Surveillance Dashboard © {datetime.now().year} | Team Data Mavericks
</div>
""", unsafe_allow_html=True)
