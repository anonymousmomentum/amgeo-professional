# streamlit_app/pages/2_Geophysical_Inversion.py
"""
Enhanced Geophysical Inversion Page with Multiple Methods
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import logging
from datetime import datetime
from io import BytesIO

# Add src to path
current_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(current_dir / "src"))

try:
    from amgeo.core.inversion import VESInversionEngine, InversionMethod
    from amgeo.core.validation import VESDataValidator, QualityLevel
except ImportError as e:
    st.error(f"Failed to import AMgeo modules: {e}")
    st.stop()

st.set_page_config(
    page_title="Geophysical Inversion - AMgeo Professional", 
    page_icon="🔬",
    layout="wide"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom CSS for inversion page
st.markdown("""
<style>
.inversion-header {
    background: linear-gradient(135deg, #1e40af 0%, #7c3aed 100%);
    color: white;
    padding: 2rem;
    border-radius: 10px;
    margin-bottom: 2rem;
    text-align: center;
}

.method-card {
    border: 2px solid #e5e7eb;
    border-radius: 10px;
    padding: 1rem;
    margin: 0.5rem 0;
    transition: all 0.3s ease;
}

.method-card.selected {
    border-color: #3b82f6;
    background-color: #eff6ff;
}

.validation-card {
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
}

.results-excellent { 
    border-left: 5px solid #10b981; 
    background: #ecfdf5; 
}

.results-good { 
    border-left: 5px solid #3b82f6; 
    background: #eff6ff; 
}

.results-acceptable { 
    border-left: 5px solid #f59e0b; 
    background: #fffbeb; 
}

.results-poor { 
    border-left: 5px solid #ef4444; 
    background: #fef2f2; 
}

.parameter-help {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
}

.metric-container {
    display: flex;
    justify-content: space-around;
    margin: 1rem 0;
}

.progress-container {
    width: 100%;
    background-color: #f0f0f0;
    border-radius: 10px;
    margin: 1rem 0;
}

.progress-bar {
    height: 20px;
    background-color: #4CAF50;
    border-radius: 10px;
    text-align: center;
    line-height: 20px;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# Page header
st.markdown("""
<div class="inversion-header">
    <h1>🔬 Geophysical Inversion</h1>
    <p>Professional VES inversion with multiple methods and uncertainty quantification</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    """Initialize session state variables"""
    defaults = {
        'uploaded_df': None,
        'validation_result': None,
        'inversion_result': None,
        'current_step': 1,
        'processing_history': []
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# Check if data is available
if st.session_state.uploaded_df is None:
    st.error("No VES data available. Please upload data in the Data Upload & Validation page first.")
    st.info("Navigate to 'Data Upload & Validation' using the sidebar to upload your VES data.")
    st.stop()

# Get data from session state
df = st.session_state.uploaded_df
validation_result = st.session_state.get('validation_result')

st.success(f"Data loaded: {len(df)} measurement points")

# Initialize inversion engine
@st.cache_resource
def get_inversion_engine():
    try:
        return VESInversionEngine()
    except Exception as e:
        st.error(f"Failed to initialize inversion engine: {e}")
        st.stop()

engine = get_inversion_engine()

# Display data quality summary
if validation_result and hasattr(validation_result, 'quality_score'):
    quality_score = validation_result.quality_score
    quality_level = validation_result.quality_level.value if hasattr(validation_result.quality_level, 'value') else str(validation_result.quality_level)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Data Points", len(df))
    with col2:
        st.metric("Quality Score", f"{quality_score:.0f}/100")
    with col3:
        st.metric("AB2 Range", f"{df['AB2'].min():.1f} - {df['AB2'].max():.1f} m")
    with col4:
        st.metric("Data Span", f"{np.log10(df['AB2'].max()/df['AB2'].min()):.1f} decades")
else:
    # Basic data summary if validation result not available
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Data Points", len(df))
    with col2:
        st.metric("AB2 Min", f"{df['AB2'].min():.1f} m")
    with col3:
        st.metric("AB2 Max", f"{df['AB2'].max():.1f} m")
    with col4:
        st.metric("Rhoa Range", f"{df['Rhoa'].min():.1f} - {df['Rhoa'].max():.1f}")

st.markdown("---")

# Inversion configuration
st.markdown("## ⚙️ Inversion Configuration")

# Method selection
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 🔧 Method Selection")
    
    # Get available methods from engine
    available_methods = engine.available_methods if hasattr(engine, 'available_methods') else {
        InversionMethod.DAMPED_LSQ: True,
        InversionMethod.ENSEMBLE: True,
        InversionMethod.PYGIMLI: False  # Default to False unless available
    }
    
    method_options = []
    method_descriptions = {
        InversionMethod.PYGIMLI.value: "Professional-grade inversion using PyGIMLi (Recommended)",
        InversionMethod.DAMPED_LSQ.value: "Damped least squares with robust fallback",
        InversionMethod.ENSEMBLE.value: "Ensemble of multiple methods for uncertainty quantification"
    }
    
    # Filter available methods
    for method, available in available_methods.items():
        if available:
            method_val = method.value if hasattr(method, 'value') else str(method)
            desc = method_descriptions.get(method_val, f"{method_val} inversion method")
            method_options.append((method_val, desc))
    
    # Ensure at least one method is available
    if not method_options:
        method_options = [("damped_lsq", "Damped least squares (fallback)")]
    
    selected_method_str = st.selectbox(
        "Inversion Method",
        options=[opt[0] for opt in method_options],
        format_func=lambda x: next(desc for val, desc in method_options if val == x),
        help="Select the inversion algorithm to use"
    )
    
    try:
        selected_method = InversionMethod(selected_method_str)
    except ValueError:
        selected_method = InversionMethod.DAMPED_LSQ  # Default fallback

with col2:
    st.markdown("### 📊 Method Status")
    for method, available in available_methods.items():
        status_icon = "✅" if available else "❌"
        method_name = method.value.upper() if hasattr(method, 'value') else str(method).upper()
        st.markdown(f"{status_icon} {method_name}")

# Parameter configuration
st.markdown("### 🎛️ Inversion Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    n_layers = st.number_input(
        "Number of layers",
        min_value=2, max_value=10, value=4, step=1,
        help="Number of layers in the resistivity model"
    )
    
    lambda_reg = st.slider(
        "Regularization (λ)",
        min_value=1, max_value=200, value=20, step=1,
        help="Higher λ = smoother model, Lower λ = more detailed model"
    )

with col2:
    max_iterations = st.number_input(
        "Max iterations",
        min_value=10, max_value=200, value=50, step=10,
        help="Maximum number of inversion iterations"
    )
    
    error_level = st.slider(
        "Data error level",
        min_value=0.01, max_value=0.20, value=0.03, step=0.01,
        format="%.2f",
        help="Assumed relative error in the data"
    )

with col3:
    verbose = st.checkbox("Verbose output", value=False, 
                         help="Show detailed inversion progress")
    
    enable_uncertainty = st.checkbox("Uncertainty analysis", value=False,
                                   help="Calculate model uncertainty (slower)")

# Advanced options
with st.expander("🔬 Advanced Options"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Model Constraints:**")
        use_bounds = st.checkbox("Apply resistivity bounds")
        if use_bounds:
            res_min = st.number_input("Min resistivity (Ω·m)", value=1.0, min_value=0.1)
            res_max = st.number_input("Max resistivity (Ω·m)", value=10000.0, min_value=10.0)
    
    with col2:
        st.markdown("**Site Information:**")
        site_name = st.text_input("Site name", value="VES Site", help="Site identification")
        location = st.text_input("Location", value="Unknown Location", help="Geographic location")
        survey_date = st.date_input("Survey date", help="Date of VES survey")

# Parameter guidance
with st.expander("📖 Parameter Guidance"):
    st.markdown("""
    <div class="parameter-help">
    <h4>🎯 Parameter Selection Guidelines</h4>
    
    <strong>Number of Layers:</strong>
    <ul>
    <li>Start with 3-4 layers for most cases</li>
    <li>Increase if data span > 2.5 decades</li>
    <li>Avoid over-parameterization (too many layers)</li>
    </ul>
    
    <strong>Regularization (λ):</strong>
    <ul>
    <li>λ = 1-10: Detailed models, may overfit</li>
    <li>λ = 10-50: Balanced approach (recommended)</li>
    <li>λ = 50-200: Smooth models, good for noisy data</li>
    </ul>
    
    <strong>Data Error Level:</strong>
    <ul>
    <li>0.01-0.03: High-quality measurements</li>
    <li>0.03-0.05: Standard field measurements</li>
    <li>0.05-0.10: Noisy or challenging conditions</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# Run inversion
st.markdown("---")
st.markdown("## 🚀 Run Inversion")

if st.button("🔬 Start VES Inversion", type="primary", use_container_width=True):
    
    # Prepare data
    ab2 = df['AB2'].values
    mn2 = df['MN2'].values if 'MN2' in df.columns else np.ones_like(ab2)  # Default MN2 if not present
    rhoa = df['Rhoa'].values
    
    # Prepare kwargs
    kwargs = {}
    if use_bounds:
        kwargs['resistivity_bounds'] = (res_min, res_max)
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Run inversion with progress
    status_text.text(f"Running {selected_method.value} inversion...")
    progress_bar.progress(20)
    
    try:
        result = engine.run_inversion(
            ab2=ab2,
            mn2=mn2,
            rhoa=rhoa,
            method=selected_method,
            n_layers=n_layers,
            lambda_reg=lambda_reg,
            max_iterations=max_iterations,
            error_level=error_level,
            verbose=verbose,
            **kwargs
        )
        
        progress_bar.progress(70)
        status_text.text("Processing results...")
        
        # Add site information
        result.site_info = {
            'site_name': site_name,
            'location': location,
            'survey_date': str(survey_date)
        }
        
        progress_bar.progress(90)
        
        # Run uncertainty analysis if requested
        if enable_uncertainty:
            status_text.text("Calculating uncertainties...")
            try:
                result = engine.estimate_uncertainty(result, n_bootstrap=50)
            except Exception as e:
                st.warning(f"Uncertainty analysis failed: {e}")
        
        progress_bar.progress(100)
        status_text.text("Inversion completed successfully!")
        
        # Store results
        st.session_state.inversion_result = result
        st.session_state.current_step = 3
        
        # Add to processing history
        st.session_state.processing_history.append({
            'step': 'inversion',
            'timestamp': datetime.now().isoformat(),
            'method': selected_method.value,
            'parameters': {
                'n_layers': n_layers,
                'lambda_reg': lambda_reg,
                'max_iterations': max_iterations,
                'error_level': error_level
            }
        })
        
        st.success("Inversion completed successfully!")
        
    except Exception as e:
        progress_bar.progress(0)
        status_text.text("")
        st.error(f"Inversion failed: {e}")
        st.error("Try adjusting parameters or using a different method.")
        logger.error(f"Inversion error: {e}")
        st.stop()

# Display results if available
if st.session_state.inversion_result is not None:
    
    result = st.session_state.inversion_result
    
    st.markdown("---")
    st.markdown("## 📊 Inversion Results")
    
    # Results summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("RMS Error", f"{result.rms_error:.2f}%")
        
    with col2:
        st.metric("Chi-squared", f"{result.chi2:.4f}")
        
    with col3:
        st.metric("Iterations", f"{result.n_iterations}")
        
    with col4:
        st.metric("Method", result.method.upper())
    
    # Quality assessment
    if result.rms_error < 3:
        quality_class = "results-excellent"
        quality_text = "Excellent fit"
    elif result.rms_error < 5:
        quality_class = "results-good" 
        quality_text = "Good fit"
    elif result.rms_error < 10:
        quality_class = "results-acceptable"
        quality_text = "Acceptable fit"
    else:
        quality_class = "results-poor"
        quality_text = "Poor fit - review parameters"
    
    st.markdown(f"""
    <div class="validation-card {quality_class}">
        <h4>Model Quality: {quality_text}</h4>
        <p>RMS Error: {result.rms_error:.2f}% | Chi-squared: {result.chi2:.4f}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Layer model results
    st.markdown("### 🗻 Layer Model")
    
    resistivities = result.resistivities
    thicknesses = result.thicknesses
    depths = result.depths
    
    # Create layer table
    layer_data = []
    for i, (res, depth) in enumerate(zip(resistivities, depths)):
        if i < len(thicknesses):
            thickness = thicknesses[i]
            depth_range = f"{depth:.1f} - {depths[i+1]:.1f} m"
        else:
            thickness = "∞"
            depth_range = f"{depth:.1f} - ∞ m"
        
        # Geological interpretation
        if res < 20:
            lithology = "Clay/Silt"
            aquifer_potential = "Poor"
        elif 20 <= res < 100:
            lithology = "Sandy Clay"
            aquifer_potential = "Moderate"
        elif 100 <= res < 500:
            lithology = "Sand/Gravel"
            aquifer_potential = "Good"
        else:
            lithology = "Bedrock/Consolidated"
            aquifer_potential = "Poor"
        
        layer_data.append({
            'Layer': i + 1,
            'Resistivity (Ω·m)': f"{res:.1f}",
            'Thickness (m)': f"{thickness}" if isinstance(thickness, str) else f"{thickness:.1f}",
            'Depth Range (m)': depth_range,
            'Likely Lithology': lithology,
            'Aquifer Potential': aquifer_potential
        })
    
    layer_df = pd.DataFrame(layer_data)
    st.dataframe(layer_df, use_container_width=True)
    
    # Visualization
    st.markdown("### 📈 Results Visualization")
    
    # Create comprehensive plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # VES curve fit
    ax1 = axes[0, 0]
    ax1.loglog(result.ab2, result.rhoa, 'bo', markersize=8, label='Observed', alpha=0.7)
    ax1.loglog(result.ab2, result.fitted_rhoa, 'r-', linewidth=3, label='Calculated')
    ax1.set_xlabel('AB/2 (m)', fontweight='bold')
    ax1.set_ylabel('Apparent Resistivity (Ω·m)', fontweight='bold')
    ax1.set_title(f'VES Data Fit (RMS: {result.rms_error:.1f}%)', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Resistivity model
    ax2 = axes[0, 1]
    
    # Create step plot for resistivity model
    depths_plot = []
    res_plot = []
    
    for i, (depth, res) in enumerate(zip(depths[:-1], resistivities[:-1])):
        depths_plot.extend([depth, depths[i+1]])
        res_plot.extend([res, res])
    
    # Add final layer
    max_depth = depths[-1] * 2 if depths[-1] > 0 else 200
    depths_plot.extend([depths[-1], max_depth])
    res_plot.extend([resistivities[-1], resistivities[-1]])
    
    ax2.semilogx(res_plot, depths_plot, 'g-', linewidth=4, label='Resistivity Model')
    
    # Shade aquifer zones
    aquifer_labeled = False
    for i, res in enumerate(resistivities):
        if 50 <= res <= 300:  # Potential aquifer range
            top = depths[i] if i < len(depths) else 0
            bottom = depths[i+1] if i+1 < len(depths) else max_depth
            label = 'Potential Aquifer' if not aquifer_labeled else ""
            ax2.axhspan(top, bottom, alpha=0.3, color='blue', label=label)
            aquifer_labeled = True
    
    ax2.set_xlabel('Resistivity (Ω·m)', fontweight='bold')
    ax2.set_ylabel('Depth (m)', fontweight='bold')
    ax2.set_title('Resistivity vs Depth', fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Residuals analysis
    ax3 = axes[1, 0]
    residuals = (result.rhoa - result.fitted_rhoa) / result.rhoa * 100
    ax3.semilogx(result.ab2, residuals, 'mo-', markersize=6, alpha=0.7)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax3.axhline(y=5, color='red', linestyle='--', alpha=0.7, label='±5% threshold')
    ax3.axhline(y=-5, color='red', linestyle='--', alpha=0.7)
    ax3.set_xlabel('AB/2 (m)', fontweight='bold')
    ax3.set_ylabel('Residual (%)', fontweight='bold')
    ax3.set_title('Data Fit Residuals', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Layer properties
    ax4 = axes[1, 1]
    
    # Bar chart of layer resistivities
    layer_names = [f'L{i+1}' for i in range(len(resistivities))]
    colors = ['red' if r < 20 else 'orange' if r < 100 else 'green' if r < 500 else 'blue' 
              for r in resistivities]
    bars = ax4.bar(layer_names, resistivities, alpha=0.7, color=colors)
    
    ax4.set_ylabel('Resistivity (Ω·m)', fontweight='bold')
    ax4.set_xlabel('Layer', fontweight='bold')
    ax4.set_title('Layer Resistivities', fontweight='bold')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, res in zip(bars, resistivities):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{res:.0f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Uncertainty analysis
    if (hasattr(result, 'resistivity_uncertainty') and result.resistivity_uncertainty is not None 
        and not np.all(np.isnan(result.resistivity_uncertainty))):
        st.markdown("### 🎯 Uncertainty Analysis")
        
        uncertainty_data = []
        for i, (res, unc) in enumerate(zip(resistivities, result.resistivity_uncertainty)):
            if not np.isnan(unc) and unc > 0:
                rel_unc = (unc / res) * 100
                uncertainty_data.append({
                    'Layer': i + 1,
                    'Resistivity (Ω·m)': f"{res:.1f}",
                    'Uncertainty (Ω·m)': f"±{unc:.1f}",
                    'Relative Uncertainty (%)': f"±{rel_unc:.1f}%"
                })
        
        if uncertainty_data:
            unc_df = pd.DataFrame(uncertainty_data)
            st.dataframe(unc_df, use_container_width=True)
        else:
            st.info("Uncertainty analysis was requested but no valid uncertainties were calculated.")
    
    # Geological interpretation
    st.markdown("### 🌍 Geological Interpretation")
    
    interpretation_text = f"""
    **Site:** {result.site_info.get('site_name', 'Unknown')}  
    **Location:** {result.site_info.get('location', 'Unknown')}  
    **Survey Date:** {result.site_info.get('survey_date', 'Unknown')}
    
    **Model Summary:**
    - **Total Layers:** {len(resistivities)}
    - **Investigation Depth:** ~{depths[-1]:.0f} m
    - **Model Quality:** {quality_text}
    
    **Aquifer Assessment:**
    """
    
    # Identify potential aquifers
    aquifer_layers = []
    for i, res in enumerate(resistivities):
        if 50 <= res <= 300:
            depth_top = depths[i] if i < len(depths) else 0
            depth_bottom = depths[i+1] if i+1 < len(depths) else "∞"
            thickness = thicknesses[i] if i < len(thicknesses) else "∞"
            aquifer_layers.append(f"Layer {i+1}: {depth_top:.1f}-{depth_bottom} m, thickness: {thickness} m")
    
    if aquifer_layers:
        interpretation_text += "\n\n**Potential Aquifer Zones:**\n"
        for aq in aquifer_layers:
            interpretation_text += f"- {aq}\n"
        interpretation_text += "\n✅ **Recommendation:** Drilling potential identified"
    else:
        interpretation_text += "\n\n❌ **No clear aquifer zones identified**"
        interpretation_text += "\n⚠️ **Recommendation:** Additional investigation recommended"
    
    st.markdown(interpretation_text)
    
    # Validation
    if hasattr(engine, 'validate_result'):
        validation = engine.validate_result(result)
        
        st.markdown("### ✅ Model Validation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Quality Score", f"{validation['quality_score']:.0f}/100")
            st.metric("Data Fit Quality", validation['metrics']['data_fit_quality'].title())
        
        with col2:
            valid_icon = "✅" if validation['is_valid'] else "❌"
            st.metric("Model Validity", f"{valid_icon} {'Valid' if validation['is_valid'] else 'Invalid'}")
            
            if validation['warnings']:
                st.markdown("**Warnings:**")
                for warning in validation['warnings']:
                    st.warning(warning)
    
    # Export options
    st.markdown("### 💾 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Download CSV Results"):
            # Create comprehensive results CSV
            results_data = {
                'AB2_m': result.ab2,
                'Observed_Rhoa_Ohm_m': result.rhoa,
                'Calculated_Rhoa_Ohm_m': result.fitted_rhoa,
                'Residual_percent': residuals
            }
            
            results_df = pd.DataFrame(results_data)
            csv = results_df.to_csv(index=False)
            
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"ves_inversion_{site_name.replace(' ', '_')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📈 Download Plot"):
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
            buf.seek(0)
            
            st.download_button(
                label="📥 Download PNG",
                data=buf.getvalue(),
                file_name=f"ves_inversion_plot_{site_name.replace(' ', '_')}.png",
                mime="image/png"
            )
    
    with col3:
        if st.button("📋 Layer Model CSV"):
            csv = layer_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Layer Model",
                data=csv,
                file_name=f"layer_model_{site_name.replace(' ', '_')}.csv",
                mime="text/csv"
            )
    
    # Next steps
    st.markdown("---")
    st.success("✅ Inversion completed successfully!")
    
    # Check if ML modules are available
    try:
        sys.path.insert(0, str(current_dir / "src"))
        from amgeo.ml.classifier import AquiferClassificationEnsemble
        from amgeo.ml.features import FeatureExtractor
        st.info("**Next Step:** Navigate to 'ML Aquifer Classification' to analyze aquifer potential using machine learning.")
    except ImportError:
        st.info("**Next Step:** Navigate to 'Results & Export' for comprehensive reporting and data export options.")

    else:
    	# Show inversion guidance
    	st.markdown("## 📖 Inversion Guidance")
    	st.info("Configure parameters above and click 'Start VES Inversion' to begin analysis.")
    
    with st.expander("🎓 Understanding VES Inversion"):
        st.markdown("""
        **What is VES Inversion?**
        
        VES (Vertical Electrical Sounding) inversion is the process of determining the subsurface 
        resistivity structure from apparent resistivity measurements at different electrode spacings.
        
        **The Process:**
        1. **Forward Modeling:** Calculate theoretical response for a layered earth model
        2. **Misfit Analysis:** Compare calculated vs observed data
        3. **Parameter Update:** Adjust model parameters to reduce misfit """)
        
        

