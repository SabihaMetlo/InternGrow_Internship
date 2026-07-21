# app.py - Clinical Disease Diagnostics Engine (Streamlit Dashboard)
# A polished dashboard for predicting diabetes risk using 3 trained ML models

import streamlit as st
import joblib
import numpy as np
import plotly.graph_objects as go
from xgboost import XGBClassifier

# ---------------------------------------------------------
# Page configuration - must be the first Streamlit command
# ---------------------------------------------------------
st.set_page_config(
    page_title="Clinical Diagnostics Engine",
    page_icon="🩺",
    layout="wide"
)

# ---------------------------------------------------------
# Custom CSS for a professional dark dashboard look
# ---------------------------------------------------------
st.markdown("""
<style>
    .block-container {
        # padding-top: 1rem !important;
    }
    section[data-testid="stSidebar"] > div {
        padding-top: 1rem !important;
    }
    section[data-testid="stSidebar"] {
        background: #0f1420;
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    .sidebar-hero {
        padding: 0.5rem 0 1rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 1.5rem;
    }
    .icon-badge-sm {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: rgba(0,212,255,0.12);
        border-radius: 12px;
        width: 2.6rem;
        height: 2.6rem;
        margin-bottom: 0.6rem;
    }
    .sidebar-title {
        font-size: 1.3rem;
        font-weight: 800;
        color: #e5e7eb;
        margin: 0;
    }
    .sidebar-sub {
        color: #6b7280;
        font-size: 0.85rem;
        margin-top: 0.3rem;
    }
    .sidebar-section {
        color: #00d4ff;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin: 1.3rem 0 0.3rem 0;
    }
    .hero {
        padding: 0.5rem 0 1rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 2rem;
    }
        .icon-badge-lg {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: rgba(0,212,255,0.12);
        border-radius: 16px;
        width: 4rem;
        height: 4rem;
        overflow: visible;
        margin-bottom: 0.8rem;
    }
        .icon-badge-sm {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: rgba(0,212,255,0.12);
        border-radius: 12px;
        width: 3rem;
        height: 3rem;
        overflow: visible;
        margin-bottom: 0.6rem;
    }
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00d4ff, #ff4b6e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        line-height: 1.2;
    }
    .subtitle {
        color: #9ca3af;
        font-size: 1.15rem;
        margin-top: 0.4rem;
    }
    .risk-card {
        padding: 1.2rem;
        border-radius: 14px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.08);
        background: rgba(255,255,255,0.03);
    }
    .risk-label {
        font-size: 0.95rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .risk-value {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0.2rem 0;
    }
    .risk-tag {
        display: inline-block;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
    }
    section[data-testid="stSidebar"] label {
        font-weight: 600;
        color: #d1d5db !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Reusable SVG icons (verified icon paths, render identically everywhere)
# ---------------------------------------------------------
STETHOSCOPE_SVG = """
<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none"
     stroke="#00d4ff" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
    <path d="M4.5 4v5a4.5 4.5 0 0 0 9 0V4"/>
    <path d="M4.5 4h-1.5"/>
    <path d="M13.5 4h1.5"/>
    <path d="M7 13v2a6.5 6.5 0 0 0 13 0v-1"/>
    <circle cx="20" cy="14" r="2.2"/>
</svg>
"""

CLIPBOARD_SVG = """
<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none"
     stroke="#00d4ff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <rect width="8" height="4" x="8" y="2" rx="1" ry="1"/>
    <path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/>
    <path d="M9 12h6"/>
    <path d="M9 16h6"/>
    <path d="M9 8h1"/>
</svg>
"""

# ---------------------------------------------------------
# Load trained models and scaler (cached so it loads only once)
# ---------------------------------------------------------
@st.cache_resource
def load_models():
    svm_model = joblib.load('svm_model.pkl')
    rf_model = joblib.load('rf_model.pkl')

    xgb_model = XGBClassifier()
    xgb_model.load_model('xgb_model.json')

    scaler = joblib.load('scaler.pkl')
    return svm_model, rf_model, xgb_model, scaler

svm_model, rf_model, xgb_model, scaler = load_models()

# ---------------------------------------------------------
# Sidebar - hero + grouped patient input sliders
# ---------------------------------------------------------
st.sidebar.markdown(f"""
<div class="sidebar-hero">
    <div class="icon-badge-sm">{CLIPBOARD_SVG.format(size=22)}</div>
    <div class="sidebar-title">Patient Metrics</div>
    <div class="sidebar-sub">Adjust the sliders to match the patient's data</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown('<div class="sidebar-section">Basic Info</div>', unsafe_allow_html=True)
pregnancies = st.sidebar.slider("Pregnancies", 0, 17, 1)
age = st.sidebar.slider("Age", 21, 81, 33)

st.sidebar.markdown('<div class="sidebar-section">Vitals</div>', unsafe_allow_html=True)
glucose = st.sidebar.slider("Glucose Level (mg/dL)", 0, 200, 120)
blood_pressure = st.sidebar.slider("Blood Pressure (mm Hg)", 0, 122, 70)
bmi = st.sidebar.slider("BMI", 0.0, 67.1, 32.0)

st.sidebar.markdown('<div class="sidebar-section">Lab Results</div>', unsafe_allow_html=True)
skin_thickness = st.sidebar.slider("Skin Thickness (mm)", 0, 99, 20)
insulin = st.sidebar.slider("Insulin (mu U/ml)", 0, 846, 79)
diabetes_pedigree = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)

# ---------------------------------------------------------
# Main hero header
# ---------------------------------------------------------
st.markdown(f"""
<div class="hero">
    <div class="icon-badge-lg">{STETHOSCOPE_SVG.format(size=30)}</div>
    <div class="main-title">Clinical Disease Diagnostics Engine</div>
    <div class="subtitle">AI-powered diabetes risk assessment using SVM, Random Forest & XGBoost</div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Build input array and get predictions from all 3 models
# ---------------------------------------------------------
input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                         insulin, bmi, diabetes_pedigree, age]])
input_scaled = scaler.transform(input_data)

svm_prob = float(svm_model.predict_proba(input_scaled)[0][1])
rf_prob = float(rf_model.predict_proba(input_data)[0][1])
xgb_prob = float(xgb_model.predict_proba(input_data)[0][1])

consensus_prob = (svm_prob + rf_prob + xgb_prob) / 3

# ---------------------------------------------------------
# Helper: convert a probability into a color + risk label
# ---------------------------------------------------------
def risk_style(prob):
    if prob < 0.33:
        return "#22c55e", "LOW RISK"
    elif prob < 0.66:
        return "#f59e0b", "MODERATE RISK"
    else:
        return "#ef4444", "HIGH RISK"

# ---------------------------------------------------------
# Consensus score banner
# ---------------------------------------------------------
color, label = risk_style(consensus_prob)
st.markdown(f"""
<div class="risk-card" style="margin-bottom: 1.5rem;">
    <div class="risk-label">Overall Consensus Risk (Average of 3 Models)</div>
    <div class="risk-value" style="color:{color};">{consensus_prob*100:.1f}%</div>
    <span class="risk-tag" style="background:{color}22; color:{color};">{label}</span>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Individual model cards
# ---------------------------------------------------------
st.subheader("Model-by-Model Breakdown")
col1, col2, col3 = st.columns(3)

for col, name, prob in zip([col1, col2, col3],
                            ["SVM", "Random Forest", "XGBoost"],
                            [svm_prob, rf_prob, xgb_prob]):
    color, label = risk_style(prob)
    with col:
        st.markdown(f"""
        <div class="risk-card">
            <div class="risk-label">{name}</div>
            <div class="risk-value" style="color:{color};">{prob*100:.1f}%</div>
            <span class="risk-tag" style="background:{color}22; color:{color};">{label}</span>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------
# Gauge charts for a more visual, clinical-dashboard feel
# ---------------------------------------------------------
st.subheader("Risk Gauges")
gauge_cols = st.columns(3)

for col, name, prob in zip(gauge_cols,
                            ["SVM", "Random Forest", "XGBoost"],
                            [svm_prob, rf_prob, xgb_prob]):
    color, _ = risk_style(prob)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={'suffix': "%", 'font': {'size': 28}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': "#9ca3af"},
            'bar': {'color': color},
            'bgcolor': "rgba(0,0,0,0)",
            'steps': [
                {'range': [0, 33], 'color': "rgba(34,197,94,0.15)"},
                {'range': [33, 66], 'color': "rgba(245,158,11,0.15)"},
                {'range': [66, 100], 'color': "rgba(239,68,68,0.15)"},
            ],
        },
        title={'text': name, 'font': {'size': 16}}
    ))
    fig.update_layout(height=220, margin=dict(l=20, r=20, t=40, b=10),
                       paper_bgcolor="rgba(0,0,0,0)", font={'color': "#e5e7eb"})
    col.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# Footer / model info
# ---------------------------------------------------------
with st.expander("ℹ️ About these models"):
    st.markdown("""
    - **SVM**: Finds the optimal boundary that separates diabetic vs non-diabetic patients with maximum margin.
    - **Random Forest**: An ensemble of decision trees voting together for a robust prediction.
    - **XGBoost**: Sequentially built trees where each one corrects the previous one's errors — typically the strongest performer on tabular medical data.

    Trained and evaluated on the Pima Indians Diabetes Dataset. This tool is for educational/demo purposes only and is not a substitute for professional medical diagnosis.
    """)