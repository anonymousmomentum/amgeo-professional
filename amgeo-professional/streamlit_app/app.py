# streamlit_app/app.py
"""
Enhanced AMgeo Professional Streamlit Application
Entry point with modern design and professional workflow
"""

import streamlit as st
import sys
import os
from pathlib import Path
import logging
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to Python path for imports
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir / "src"))

try:
    from amgeo.config.settings import get_settings
    from amgeo.core.validation import VESDataValidator
    from amgeo.core.inversion import VESInversionEngine
    from amgeo.ml.classifier import AquiferClassificationEnsemble
except ImportError as e:
    st.error(f"Failed to import AMgeo modules: {e}")
    st.error("Please ensure the AMgeo package is properly installed.")
    st.stop()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="AMgeo Professional - Groundwater Exploration Suite",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/amgeo/amgeo-professional',
        'Report a bug': 'https://github.com/amgeo/amgeo-professional/issues',
        'About': 'AMgeo Professional v2.0 - Industry-standard groundwater exploration'
    }
)

# Load settings
@st.cache_resource
def load_settings():
    try:
        return get_settings()
    except Exception as e:
        logger.warning(f"Failed to load settings: {e}")
        return None

settings = load_settings()

# Initialize session state
def init_session_state():
    """Initialize session state variables"""
    defaults = {
        'uploaded_df': None,
        'validation_result': None,
        'inversion_result': None,
        'ml_model': None,
        'ml_prediction': None,
        'current_step': 1,
        'processing_history': [],
        'site_metadata': {}
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# Custom CSS for professional styling
st.markdown("""
<style>
.main-header {
    text-align: center;
    padding: 2rem 0;
    background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
    color: white;
    margin: -1rem -1rem 2rem -1rem;
    border-radius: 10px;
}

.metric-card {
    background: white;
    padding: 1rem;
    border-radius: 10px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    border-left: 4px solid #3b82f6;
}

.status-success {
    background: #dcfce7;
    border: 1px solid #16a34a;
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
}

.status-warning {
    background: #fef3c7;
    border: 1px solid #d97706;
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
}

.status-error {
    background: #fee2e2;
    border: 1px solid #dc2626;
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
}

.workflow-step {
    display: flex;
    align-items: center;
    padding: 1rem;
    margin: 0.5rem 0;
    border-radius: 8px;
    border: 2px solid #e5e7eb;
}

.workflow-step.active {
    border-color: #3b82f6;
    background: #eff6ff;
}

.workflow-step.completed {
    border-color: #16a34a;
    background: #f0fdf4;
}

.feature-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 1rem;
    margin: 2rem 0;
}

.feature-card {
    background: white;
    padding: 1.5rem;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    text-align: center;
}

.feature-icon {
    font-size: 2rem;
    margin-bottom: 1rem;
}

.sidebar-info {
    background: #f8fafc;
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
    border-left: 4px solid #3b82f6;
}

.progress-indicator {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin: 1rem 0;
    padding: 1rem;
    background: #f1f5f9;
    border-radius: 8px;
}

.step-circle {
    width: 30px;
    height: 30px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: bold;
    margin-right: 0.5rem;
}

.step-circle.completed {
    background: #16a34a;
    color: white;
}

.step-circle.active {
    background: #3b82f6;
    color: white;
}

.step-circle.pending {
    background: #e5e7eb;
    color: #6b7280;
}
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown("""
<div class="main-header">
    <h1>🌊 AMgeo Professional</h1>
    <h3>Advanced Groundwater Exploration Suite</h3>
    <p>Industry-standard VES inversion and ML-powered aquifer classification</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🧭 Navigation")
    
    # Progress indicator
    current_step = st.session_state.get('current_step', 1)
    
    workflow_steps = [
        {"id": 1, "name": "Data Upload", "icon": "📊"},
        {"id": 2, "name": "Inversion", "icon": "🔬"},
        {"id": 3, "name": "ML Analysis", "icon": "🤖"},
        {"id": 4, "name": "Results", "icon": "📋"}
    ]
    
    st.markdown("### 📈 Workflow Progress")
    
    for step in workflow_steps:
        if step["id"] < current_step:
            status = "completed"
            icon_color = "🟢"
        elif step["id"] == current_step:
            status = "active"
            icon_color = "🔵"
        else:
            status = "pending"
            icon_color = "⚪"
        
        st.markdown(f"{icon_color} **{step['icon']} {step['name']}**")
    
    st.markdown("---")
    
    # Session status
    st.markdown("### 📊 Session Status")
    
    data_status = "✅ Loaded" if st.session_state.uploaded_df is not None else "❌ Not loaded"
    st.markdown(f"**Data:** {data_status}")
    
    inversion_status = "✅ Complete" if st.session_state.inversion_result is not None else "❌ Pending"
    st.markdown(f"**Inversion:** {inversion_status}")
    
    ml_status = "✅ Complete" if st.session_state.ml_prediction is not None else "❌ Pending"
    st.markdown(f"**ML Analysis:** {ml_status}")
    
    # System status
    st.markdown("---")
    st.markdown("### ⚙️ System Status")
    
    # Check component availability
    components_status = []
    
    try:
        engine = VESInversionEngine()
        components_status.append(("Inversion Engine", "✅"))
    except Exception:
        components_status.append(("Inversion Engine", "❌"))
    
    try:
        classifier = AquiferClassificationEnsemble()
        components_status.append(("ML Classifier", "✅"))
    except Exception:
        components_status.append(("ML Classifier", "❌"))
    
    try:
        validator = VESDataValidator()
        components_status.append(("Data Validator", "✅"))
    except Exception:
        components_status.append(("Data Validator", "❌"))
    
    for component, status in components_status:
        st.markdown(f"**{component}:** {status}")

# Main content area
st.markdown("## 🏠 Welcome to AMgeo Professional")

# Quick start section
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ### 🚀 Getting Started
    
    AMgeo Professional is a comprehensive groundwater exploration suite that combines:
    - **Professional VES Inversion** with multiple algorithms
    - **Machine Learning Classification** for aquifer assessment
    - **Uncertainty Quantification** for reliable results
    - **Export & Reporting** capabilities for field work
    
    **Current Workflow Status:**
    """)
    
    # Workflow status cards
    if st.session_state.uploaded_df is None:
        st.markdown("""
        <div class="status-warning">
        <strong>📊 Step 1: Data Upload Required</strong><br>
        Please navigate to the "Data Upload & Validation" page to begin analysis.
        </div>
        """, unsafe_allow_html=True)
    
    elif st.session_state.inversion_result is None:
        st.markdown("""
        <div class="status-warning">
        <strong>🔬 Step 2: Inversion Required</strong><br>
        Data loaded successfully. Navigate to "Geophysical Inversion" to analyze your VES data.
        </div>
        """, unsafe_allow_html=True)
    
    elif st.session_state.ml_prediction is None:
        st.markdown("""
        <div class="status-warning">
        <strong>🤖 Step 3: ML Analysis Available</strong><br>
        Inversion complete. Navigate to "ML Aquifer Classification" for advanced analysis.
        </div>
        """, unsafe_allow_html=True)
    
    else:
        st.markdown("""
        <div class="status-success">
        <strong>✅ Analysis Complete</strong><br>
        All analysis steps completed. Navigate to "Results & Export" for comprehensive reporting.
        </div>
        """, unsafe_allow_html=True)

with col2:
    st.markdown("### 📊 Quick Stats")
    
    # Display quick statistics if data is available
    if st.session_state.uploaded_df is not None:
        df = st.session_state.uploaded_df
        
        st.metric("Data Points", len(df))
        st.metric("AB2 Range (m)", f"{df['AB2'].min():.1f} - {df['AB2'].max():.1f}")
        
        if st.session_state.inversion_result is not None:
            result = st.session_state.inversion_result
            st.metric("RMS Error (%)", f"{result.rms_error:.2f}")
            st.metric("Model Layers", len(result.resistivities))
        
        if st.session_state.ml_prediction is not None:
            pred = st.session_state.ml_prediction
            prob = pred['aquifer_probability'][0] if isinstance(pred['aquifer_probability'], list) else pred['aquifer_probability']
            st.metric("Aquifer Probability", f"{prob:.1%}")
    else:
        st.info("Upload data to see statistics")

# Feature showcase
st.markdown("---")
st.markdown("## 🌟 Key Features")

st.markdown("""
<div class="feature-grid">
    <div class="feature-card">
        <div class="feature-icon">🔬</div>
        <h4>Professional Inversion</h4>
        <p>Multiple algorithms including PyGIMLi, damped least squares, and ensemble methods for robust analysis</p>
    </div>
    
    <div class="feature-card">
        <div class="feature-icon">🤖</div>
        <h4>Machine Learning</h4>
        <p>Advanced ensemble classification with uncertainty quantification for aquifer potential assessment</p>
    </div>
    
    <div class="feature-card">
        <div class="feature-icon">📊</div>
        <h4>Data Validation</h4>
        <p>Comprehensive quality checks, outlier detection, and automated data preprocessing</p>
    </div>
    
    <div class="feature-card">
        <div class="feature-icon">📈</div>
        <h4>Visualization</h4>
        <p>Publication-quality plots, interactive analysis, and comprehensive result visualization</p>
    </div>
    
    <div class="feature-card">
        <div class="feature-icon">📋</div>
        <h4>Professional Reports</h4>
        <p>Automated report generation with geological interpretation and drilling recommendations</p>
    </div>
    
    <div class="feature-card">
        <div class="feature-icon">⚡</div>
        <h4>Fast & Reliable</h4>
        <p>Optimized algorithms with robust error handling and fallback methods for production use</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Recent activity
if st.session_state.processing_history:
    st.markdown("---")
    st.markdown("## 📋 Recent Activity")
    
    # Show last 5 processing steps
    recent_history = st.session_state.processing_history[-5:]
    
    for item in reversed(recent_history):
        timestamp = datetime.fromisoformat(item['timestamp']).strftime("%H:%M:%S")
        step_name = item['step'].replace('_', ' ').title()
        st.markdown(f"**{timestamp}** - {step_name} completed")

# Tips and help
st.markdown("---")
st.markdown("## 💡 Tips for Best Results")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 📊 Data Quality
    - Ensure AB2 spans at least 1.5-2 decades
    - Include at least 10-15 measurement points
    - Check for outliers and measurement errors
    - Verify electrode spacing ratios (AB >> MN)
    """)
    
    st.markdown("""
    ### 🔬 Inversion Tips
    - Start with 3-4 layers for most cases
    - Use regularization λ = 10-50 for balanced results
    - Apply resistivity bounds when geology is known
    - Try ensemble method for uncertainty analysis
    """)

with col2:
    st.markdown("""
    ### 🤖 ML Classification
    - Provide site metadata for better accuracy
    - Use uncertainty metrics for decision making
    - Consider geological context in interpretation
    - Validate results with local hydrogeology
    """)
    
    st.markdown("""
    ### 📋 Interpretation
    - Cross-reference with geological maps
    - Consider regional hydrogeological setting
    - Use multiple VES points for validation
    - Integrate with other geophysical methods
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6b7280; padding: 2rem;">
    <p><strong>AMgeo Professional v2.0</strong> | Industry-standard groundwater exploration</p>
    <p>Powered by advanced geophysical inversion and machine learning technologies</p>
</div>
""", unsafe_allow_html=True)