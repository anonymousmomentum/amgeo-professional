# streamlit_app/pages/3_ML_Aquifer_Classification.py
"""
Machine Learning Aquifer Classification Page
Enhanced with ensemble methods and uncertainty quantification
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add src to path
current_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(current_dir / "src"))

from amgeo.core.features import FeatureExtractor
from amgeo.ml.classifier import AquiferClassificationEnsemble

st.set_page_config(
    page_title="ML Aquifer Classification - AMgeo Professional",
    page_icon="🤖", 
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.ml-header {
    background: linear-gradient(135deg, #7c3aed 0%, #ec4899 100%);
    color: white;
    padding: 2rem;
    border-radius: 10px;
    margin-bottom: 2rem;
}

.prediction-excellent { border-left: 5px solid #10b""", unsafe_allow_html=True)

# Main application header
st.markdown("""
<div class="main-header">
    <h1>🌊 AMgeo Professional</h1>
    <h3>Industry-Standard Groundwater Exploration Suite</h3>
    <p>Advanced VES Analysis • Machine Learning • Professional Reporting</p>
</div>
""", unsafe_allow_html=True)

# Sidebar navigation and status
with st.sidebar:
    st.markdown("## 📋 Workflow Navigation")
    
    # Workflow steps with status indicators
    steps = [
        ("1. Data Upload & Validation", st.session_state.uploaded_df is not None),
        ("2. Geophysical Inversion", st.session_state.inversion_result is not None),
        ("3. ML Aquifer Classification", st.session_state.ml_prediction is not None),
        ("4. Results & Export", False),  # Always available
        ("5. Synthetic Demo", False)     # Always available
    ]
    
    for i, (step_name, completed) in enumerate(steps, 1):
        status_class = "completed" if completed else ("active" if i == st.session_state.current_step else "")
        status_icon = "✅" if completed else ("🔄" if i == st.session_state.current_step else "⭕")
        
        st.markdown(f"""
        <div class="workflow-step {status_class}">
            <span style="margin-right: 10px;">{status_icon}</span>
            <span>{step_name}</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # System status
    st.markdown("## 🔧 System Status")
    
    # Check dependencies
    dependencies = {
        "PyGIMLi": False,
        "PostgreSQL": False,
        "Redis": False
    }
    
    try:
        import pygimli
        dependencies["PyGIMLi"] = True
    except ImportError:
        pass
    
    for dep, available in dependencies.items():
        status_icon = "🟢" if available else "🔴"
        st.markdown(f"{status_icon} {dep}")
    
    st.markdown("---")
    
    # Quick actions
    st.markdown("## ⚡ Quick Actions")
    
    if st.button("🧹 Clear Session", type="secondary"):
        for key in list(st.session_state.keys()):
            if key not in ['current_step']:
                del st.session_state[key]
        init_session_state()
        st.rerun()
    
    if st.button("📊 System Info", type="secondary"):
        st.session_state.show_system_info = not st.session_state.get('show_system_info', False)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # Application overview
    st.markdown("## 🎯 Professional Groundwater Exploration")
    
    st.markdown("""
    AMgeo Professional provides industry-standard VES (Vertical Electrical Sounding) analysis 
    with advanced machine learning for groundwater exploration and aquifer assessment.
    """)
    
    # Feature highlights
    st.markdown("""
    <div class="feature-grid">
        <div class="feature-card">
            <div class="feature-icon">🔬</div>
            <h4>Multi-Method Inversion</h4>
            <p>PyGIMLi integration with robust fallbacks</p>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🤖</div>
            <h4>Advanced ML Pipeline</h4>
            <p>Ensemble learning with uncertainty quantification</p>
        </div>
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <h4>Professional Reporting</h4>
            <p>ASTM D6431 compliant analysis and documentation</p>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🗺️</div>
            <h4>Spatial Analysis</h4>
            <p>Multi-site groundwater mapping</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    # Current project status
    st.markdown("## 📈 Project Status")
    
    # Data status
    if st.session_state.uploaded_df is not None:
        n_points = len(st.session_state.uploaded_df)
        st.markdown(f"""
        <div class="status-success">
            <strong>✅ Data Loaded</strong><br>
            {n_points} measurement points
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.validation_result:
            quality_score = st.session_state.validation_result.quality_score
            quality_level = st.session_state.validation_result.quality_level.value
            st.metric("Data Quality", f"{quality_score:.0f}/100", f"{quality_level.title()}")
    else:
        st.markdown("""
        <div class="status-warning">
            <strong>⏳ No Data</strong><br>
            Upload VES data to begin
        </div>
        """, unsafe_allow_html=True)
    
    # Inversion status
    if st.session_state.inversion_result is not None:
        result = st.session_state.inversion_result
        st.markdown(f"""
        <div class="status-success">
            <strong>✅ Inversion Complete</strong><br>
            {len(result.resistivities)} layers, RMS: {result.rms_error:.1f}%
        </div>
        """, unsafe_allow_html=True)
        
        st.metric("Inversion Method", result.method.upper())
        st.metric("Model Fit", f"{result.rms_error:.1f}%", f"χ² = {result.chi2:.3f}")
    else:
        st.markdown("""
        <div class="status-warning">
            <strong>⏳ No Inversion</strong><br>
            Run geophysical inversion
        </div>
        """, unsafe_allow_html=True)
    
    # ML prediction status
    if st.session_state.ml_prediction is not None:
        prediction = st.session_state.ml_prediction
        prob = prediction.get('aquifer_probability', 0)
        uncertainty = prediction.get('total_uncertainty', 0)
        
        st.markdown(f"""
        <div class="status-success">
            <strong>✅ ML Prediction</strong><br>
            Aquifer probability: {prob:.1%} ± {uncertainty:.1%}
        </div>
        """, unsafe_allow_html=True)
        
        # Recommendation
        if prob >= 0.7:
            recommendation = "🎯 Drilling Recommended"
            rec_color = "success"
        elif prob >= 0.4:
            recommendation = "🤔 Further Investigation"
            rec_color = "warning"
        else:
            recommendation = "❌ Poor Aquifer Potential"
            rec_color = "error"
        
        st.markdown(f"""
        <div class="status-{rec_color}">
            <strong>{recommendation}</strong>
        </div>
        """, unsafe_allow_html=True)

# Quick start guide
st.markdown("---")
st.markdown("## 🚀 Quick Start Guide")

tab1, tab2, tab3, tab4 = st.tabs(["📁 Upload Data", "🔄 Run Inversion", "🤖 ML Analysis", "📊 Export Results"])

with tab1:
    st.markdown("""
    **Step 1: Upload VES Data**
    
    1. Prepare CSV file with columns: `AB2`, `MN2`, `Rhoa`
    2. Navigate to **Data Upload & Validation** page
    3. Upload file and review quality assessment
    4. Address any validation issues before proceeding
    
    **Data Requirements:**
    - Minimum 5 measurement points
    - AB2 values in ascending order
    - All values positive and finite
    - AB2/MN2 ratio >= 2
    """)

with tab2:
    st.markdown("""
    **Step 2: Geophysical Inversion**
    
    1. Configure inversion parameters (layers, regularization)
    2. Select inversion method (PyGIMLi recommended)
    3. Run inversion and review results
    4. Check model fit quality (RMS < 10% ideal)
    
    **Parameter Guidelines:**
    - Start with 3-4 layers
    - Higher λ = smoother models
    - Lower λ = more detailed models
    """)

with tab3:
    st.markdown("""
    **Step 3: Machine Learning Analysis**
    
    1. Features automatically extracted from inversion
    2. Run ensemble classifier for aquifer prediction
    3. Review prediction confidence and uncertainty
    4. Examine feature importance analysis
    
    **Interpretation:**
    - Probability > 70%: High aquifer potential
    - Probability 40-70%: Moderate potential
    - Probability < 40%: Poor potential
    """)

with tab4:
    st.markdown("""
    **Step 4: Export Results**
    
    1. Generate professional technical report
    2. Export data in multiple formats (CSV, PDF)
    3. Download high-resolution plots
    4. Save project for future reference
    
    **Export Options:**
    - Technical reports (PDF)
    - Data tables (CSV, Excel)
    - Plots (PNG, SVG)
    - Project files (JSON)
    """)

# System information (if requested)
if st.session_state.get('show_system_info', False):
    st.markdown("---")
    st.markdown("## 🔧 System Information")
    
    with st.expander("Detailed System Status", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Environment:**")
            st.code(f"""
Environment: {settings.environment}
Debug Mode: {settings.debug}
Python Path: {sys.executable}
Working Directory: {os.getcwd()}
            """)
            
            st.markdown("**Configuration:**")
            st.code(f"""
Database: {settings.database.url}
Inversion Method: {settings.inversion.default_method}
ML Models: {settings.ml.model_cache_dir}
Max Layers: {settings.ml.max_layers}
            """)
        
        with col2:
            st.markdown("**Session State:**")
            st.json({
                key: str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                for key, value in st.session_state.items()
                if not key.startswith('_')
            })
            
            st.markdown("**Available Methods:**")
            try:
                from amgeo.core.inversion import VESInversionEngine
                engine = VESInversionEngine(settings)
                available_methods = [method.value for method, available in engine.available_methods.items() if available]
                st.code("\n".join(available_methods))
            except Exception as e:
                st.error(f"Error checking methods: {e}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 2rem 0;'>
    <p><strong>AMgeo Professional v2.0</strong></p>
    <p>Industry-standard groundwater exploration suite</p>
    <p>Powered by PyGIMLi • scikit-learn • Streamlit</p>
    <p><em>For professional groundwater assessment and aquifer exploration</em></p>
</div>
""", unsafe_allow_html=True)