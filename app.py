"""
Contact Center WFM Tool
Complete workforce management with AI assistance
All features built with free resources
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv
import os

# Import custom modules
from src.forecasting import VolumeForecaster
from src.erlang_engine import ErlangEngine
from src.simulation import SimulationEngine
from src.llm_interface import get_llm_interface

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="WFM Pro - Contact Center Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-top: -10px;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin: 5px;
    }
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border-radius: 5px;
    }
    .stButton > button:hover {
        background-color: #145a8d;
        color: white;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'forecaster' not in st.session_state:
    st.session_state.forecaster = None
if 'simulation_engine' not in st.session_state:
    st.session_state.simulation_engine = SimulationEngine()
if 'llm_interface' not in st.session_state:
    try:
        st.session_state.llm_interface = get_llm_interface()
    except:
        st.session_state.llm_interface = None
if 'forecast_results' not in st.session_state:
    st.session_state.forecast_results = None

# Sidebar Navigation
st.sidebar.title("📊 WFM Pro")
st.sidebar.markdown("---")

pages = [
    "🏠 Dashboard",
    "📈 Volume Forecast",
    "📋 Staffing Calculator",
    "🔄 Shift Simulation",
    "📊 SLA Scenarios",
    "🤖 AI Assistant",
    "📚 Documentation"
]

selected_page = st.sidebar.radio("Navigation", pages)

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("Built with ❤️ using free resources")
st.sidebar.caption("Powered by OpenRouter AI (free tier)")

# Initialize engines
erlang = ErlangEngine()
sim_engine = st.session_state.simulation_engine

# =============== PAGE ROUTING ===============

if selected_page == "🏠 Dashboard":
    st.markdown('<p class="main-header">🎯 WFM Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Complete Workforce Management Solution - Powered by AI</p>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📞 Forecast Calls", "2,847", "+12% vs last week")
    with col2:
        st.metric("👥 Agents Needed", "34", "at peak")
    with col3:
        st.metric("🎯 Current SLA", "78.5%", "-1.5%")
    with col4:
        st.metric("⏱️ ASA", "28s", "+3s")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Quick Actions")
        if st.button("📈 Generate New Forecast", use_container_width=True):
            st.success("Ready to generate forecast!")
        if st.button("📋 Calculate Staffing", use_container_width=True):
            st.success("Staffing calculator ready!")
        if st.button("🔄 Run Simulation", use_container_width=True):
            st.success("Simulation engine ready!")
    
    with col2:
        st.subheader("System Status")
        st.info("✅ Erlang Engine: Active")
        st.info("✅ Forecast Engine: Ready")
        if st.session_state.llm_interface:
            st.info("🤖 AI Assistant: Connected (Free Model)")
        else:
            st.warning("⚠️ AI Assistant: Configure API Key")
    
    st.markdown("---")
    st.subheader("📋 Getting Started")
    st.markdown("""
    1. **Upload Historical Data** → Navigate to Volume Forecast
    2. **Generate Forecast** → Predict future call volumes
    3. **Calculate Staffing** → Determine agent requirements
    4. **Run Simulations** → Test different scenarios
    5. **Get AI Insights** → Ask questions about your data