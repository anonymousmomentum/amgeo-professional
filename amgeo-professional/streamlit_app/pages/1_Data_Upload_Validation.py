# streamlit_app/pages/1_Data_Upload_Validation.py
"""
Professional Data Upload and Validation Page
Enhanced with comprehensive quality assessment
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

from amgeo.core.validation import VESDataValidator, DataQualityLevel
from amgeo.visualization.plotting import create_ves_sounding_plot, create_comprehensive_plot

st.set_page_config(
    page_title="Data Upload & Validation - AMgeo Professional",
    page_icon="📁",
    layout="wide"
)

# Custom CSS for validation page
st.markdown("""
<style>
.validation-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 2rem;
    border-radius: 10px;
    margin-bottom: 2rem;
}

.quality-excellent { border-left: 5px solid #10b981; background: #ecfdf5; }
.quality-good { border-left: 5px solid #3b82f6; background: #eff6ff; }
.quality-acceptable { border-left: 5px solid #f59e0b; background: #fffbeb; }
.quality-poor { border-left: 5px solid #ef4444; background: #fef2f2; }
.quality-unacceptable { border-left: 5px solid #991b1b; background: #fef2f2; }

.validation-card {
    padding: 1.5rem;
    border-radius: 10px;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# Page header
st.markdown("""
<div class="validation-header">
    <h1>📁 Data Upload & Validation</h1>
    <p>Professional VES data validation following ASTM D6431 standards</p>
</div>
""", unsafe_allow_html=True)

# Initialize validator
@st.cache_resource
def get_validator():
    return VESDataValidator()
    validator = get_validator()

# File upload section
st.markdown("## 📤 Upload VES Data")

col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Choose VES CSV file",
        type=['csv'],
        help="Upload CSV file with AB2, MN2, Rhoa columns",
        accept_multiple_files=False
    )

with col2:
    st.markdown("**Required Format:**")
    st.code("""
AB2,MN2,Rhoa
0.5,0.15,45.2
1.0,0.3,52.1
2.0,0.6,48.9
5.0,1.5,67.2
...
    """)

# Data processing options
if uploaded_file is not None:
    st.markdown("## ⚙️ Processing Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        auto_clean = st.checkbox("Auto-clean data", value=True, 
                                help="Remove invalid values and duplicates")
        sort_by_ab2 = st.checkbox("Sort by AB2", value=True,
                                 help="Ensure AB2 values are in ascending order")
    
    with col2:
        filter_outliers = st.checkbox("Filter outliers", value=False,
                                     help="Remove statistical outliers using IQR method")
        validate_config = st.checkbox("Validate electrode configuration", value=True,
                                     help="Check AB2/MN2 ratios and spacing")
    
    with col3:
        quality_threshold = st.slider("Minimum quality score", 0, 100, 40,
                                     help="Minimum acceptable quality score")

# Process uploaded file
if uploaded_file is not None:
    try:
        # Read CSV
        df = pd.read_csv(uploaded_file)
        
        # Check required columns
        required_cols = ['AB2', 'MN2', 'Rhoa']
        missing_cols = set(required_cols) - set(df.columns)
        
        if missing_cols:
            st.error(f"❌ Missing required columns: {missing_cols}")
            st.error("Please ensure your CSV contains exactly these columns: AB2, MN2, Rhoa")
            st.info("**Available columns:** " + ", ".join(df.columns.tolist()))
            st.stop()
        
        # Show original data info
        st.info(f"📊 Original data: {len(df)} rows, {len(df.columns)} columns")
        
        # Data preprocessing
        original_length = len(df)
        
        if auto_clean:
            # Remove rows with NaN values
            df = df.dropna(subset=required_cols)
            
            # Remove non-positive values
            df = df[(df['AB2'] > 0) & (df['MN2'] > 0) & (df['Rhoa'] > 0)]
            
            # Remove duplicates
            df = df.drop_duplicates()
            
            # Remove infinite values
            df = df[np.isfinite(df[required_cols]).all(axis=1)]
            
            if len(df) < original_length:
                st.warning(f"⚠️ Cleaned data: removed {original_length - len(df)} invalid rows")
        
        if sort_by_ab2:
            df = df.sort_values('AB2').reset_index(drop=True)
        
        if filter_outliers and len(df) > 5:
            # Remove outliers using IQR method
            Q1 = df['Rhoa'].quantile(0.25)
            Q3 = df['Rhoa'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            df_filtered = df[(df['Rhoa'] >= lower_bound) & (df['Rhoa'] <= upper_bound)]
            
            if len(df_filtered) < len(df):
                st.warning(f"⚠️ Outlier filtering: removed {len(df) - len(df_filtered)} outliers")
                df = df_filtered
        
        # Final data check
        if len(df) == 0:
            st.error("❌ No valid data remaining after preprocessing")
            st.stop()
        
        # Run comprehensive validation
        with st.spinner("🔍 Running comprehensive validation..."):
            validation_result = validator.validate_ves_data(df)
        
        # Display validation results
        st.markdown("## 📋 Validation Results")
        
        # Overall quality assessment
        quality_level = validation_result.quality_level
        quality_score = validation_result.quality_score
        
        quality_class = f"quality-{quality_level.value}"
        quality_emoji = {
            DataQualityLevel.EXCELLENT: "🟢",
            DataQualityLevel.GOOD: "🔵", 
            DataQualityLevel.ACCEPTABLE: "🟡",
            DataQualityLevel.POOR: "🟠",
            DataQualityLevel.UNACCEPTABLE: "🔴"
        }
        
        st.markdown(f"""
        <div class="validation-card {quality_class}">
            <h3>{quality_emoji[quality_level]} Data Quality: {quality_level.value.title()}</h3>
            <h2>Quality Score: {quality_score:.0f}/100</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Detailed validation report
        col1, col2 = st.columns(2)
        
        with col1:
            # Issues and warnings
            if validation_result.issues:
                st.markdown("### 🚨 Critical Issues")
                for issue in validation_result.issues:
                    st.error(f"• {issue}")
            
            if validation_result.warnings:
                st.markdown("### ⚠️ Warnings")
                for warning in validation_result.warnings:
                    st.warning(f"• {warning}")
        
        with col2:
            # Recommendations
            if validation_result.recommendations:
                st.markdown("### 💡 Recommendations")
                for rec in validation_result.recommendations:
                    st.info(f"• {rec}")
        
        # Data quality metrics
        st.markdown("## 📊 Quality Metrics")
        
        metadata = validation_result.metadata
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Data Points", metadata['n_points'])
            ab2_range = metadata['ab2_range']
            st.metric("AB2 Range (m)", f"{ab2_range[0]:.1f} - {ab2_range[1]:.1f}")
        
        with col2:
            rhoa_range = metadata['rhoa_range']
            st.metric("Resistivity Range", f"{rhoa_range[0]:.0f} - {rhoa_range[1]:.0f} Ω·m")
            st.metric("Data Span", f"{metadata['data_span_decades']:.1f} decades")
        
        with col3:
            st.metric("Measurement Density", f"{metadata['measurement_density']:.1f} pts/decade")
            st.metric("Noise Level", f"{metadata['noise_level']:.3f}")
        
        with col4:
            st.metric("Smoothness Score", f"{metadata['smoothness_score']:.2f}")
            is_valid_icon = "✅" if validation_result.is_valid else "❌"
            st.metric("Valid for Inversion", f"{is_valid_icon}")
        
        # Data preview
        st.markdown("## 👁️ Data Preview")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📊 Statistical Summary")
            st.dataframe(df[required_cols].describe())
        
        with col2:
            st.markdown("### 📋 Raw Data Sample")
            st.dataframe(df.head(10))
        
        # Visualization
        st.markdown("## 📈 Data Visualization")
        
        # Create VES sounding curve
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # VES curve
        ax1.loglog(df['AB2'], df['Rhoa'], 'bo-', markersize=6, linewidth=2, alpha=0.8)
        ax1.set_xlabel('AB/2 (m)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Apparent Resistivity (Ω·m)', fontsize=12, fontweight='bold')
        ax1.set_title('VES Sounding Curve', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add data quality annotations
        ax1.text(0.02, 0.98, f'Quality: {quality_level.value.title()}\nScore: {quality_score:.0f}/100', 
                transform=ax1.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Data distribution
        ax2.hist(np.log10(df['Rhoa']), bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_xlabel('log₁₀(Apparent Resistivity)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax2.set_title('Resistivity Distribution', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Additional quality plots
        if quality_score >= quality_threshold:
            with st.expander("🔬 Advanced Quality Analysis", expanded=False):
                
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
                
                # AB2/MN2 ratios
                ratios = df['AB2'] / df['MN2']
                ax1.semilogx(df['AB2'], ratios, 'go-', alpha=0.7)
                ax1.axhline(y=2, color='red', linestyle='--', alpha=0.7, label='Minimum (2)')
                ax1.set_xlabel('AB/2 (m)')
                ax1.set_ylabel('AB2/MN2 Ratio')
                ax1.set_title('Electrode Configuration')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Data spacing analysis
                if len(df) > 1:
                    spacing_ratios = df['AB2'].values[1:] / df['AB2'].values[:-1]
                    ax2.plot(df['AB2'].values[1:], spacing_ratios, 'mo-', alpha=0.7)
                    ax2.axhline(y=1.5, color='orange', linestyle='--', alpha=0.7, label='Ideal minimum')
                    ax2.axhline(y=3, color='red', linestyle='--', alpha=0.7, label='Maximum recommended')
                    ax2.set_xlabel('AB/2 (m)')
                    ax2.set_ylabel('Spacing Ratio')
                    ax2.set_title('Measurement Spacing Analysis')
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                
                # Residual analysis (simple smoothness check)
                if len(df) >= 5:
                    window = min(5, len(df) // 3)
                    smoothed = df['Rhoa'].rolling(window=window, center=True).mean()
                    residuals = (df['Rhoa'] - smoothed) / df['Rhoa'] * 100
                    
                    ax3.semilogx(df['AB2'], residuals, 'co-', alpha=0.7)
                    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
                    ax3.axhline(y=10, color='red', linestyle='--', alpha=0.7, label='±10% noise threshold')
                    ax3.axhline(y=-10, color='red', linestyle='--', alpha=0.7)
                    ax3.set_xlabel('AB/2 (m)')
                    ax3.set_ylabel('Residual (%)')
                    ax3.set_title('Data Smoothness Analysis')
                    ax3.legend()
                    ax3.grid(True, alpha=0.3)
                
                # Quality evolution
                quality_metrics = []
                window_sizes = range(5, min(len(df)+1, 21))
                
                for w in window_sizes:
                    subset = df.iloc[:w]
                    if len(subset) >= 5:
                        temp_result = validator.validate_ves_data(subset)
                        quality_metrics.append(temp_result.quality_score)
                    else:
                        quality_metrics.append(0)
                
                if quality_metrics:
                    ax4.plot(window_sizes, quality_metrics, 'bo-', alpha=0.7)
                    ax4.axhline(y=quality_threshold, color='red', linestyle='--', alpha=0.7, 
                               label=f'Threshold ({quality_threshold})')
                    ax4.set_xlabel('Number of Data Points')
                    ax4.set_ylabel('Quality Score')
                    ax4.set_title('Quality vs Data Size')
                    ax4.legend()
                    ax4.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
        
        # Decision point
        st.markdown("## ✅ Validation Complete")
        
        if validation_result.is_valid and quality_score >= quality_threshold:
            st.success(f"🎉 Data validation passed! Quality score: {quality_score:.0f}/100")
            st.success("✅ Data is ready for geophysical inversion.")
            
            # Save to session state
            st.session_state.uploaded_df = df
            st.session_state.validation_result = validation_result
            st.session_state.current_step = 2
            
            # Provide next step guidance
            st.info("📍 **Next Step:** Navigate to 'Geophysical Inversion' to analyze your validated data.")
            
            # Optional: Show processing history
            processing_info = {
                'timestamp': pd.Timestamp.now(),
                'action': 'data_validation',
                'original_rows': original_length,
                'final_rows': len(df),
                'quality_score': quality_score,
                'quality_level': quality_level.value
            }
            
            if 'processing_history' not in st.session_state:
                st.session_state.processing_history = []
            st.session_state.processing_history.append(processing_info)
            
        else:
            st.error(f"❌ Data validation failed. Quality score: {quality_score:.0f}/100")
            
            if quality_score < quality_threshold:
                st.error(f"Quality score ({quality_score:.0f}) is below threshold ({quality_threshold})")
            
            st.error("🚫 Data quality is insufficient for reliable inversion results.")
            
            # Provide improvement suggestions
            st.markdown("### 🔧 Improvement Suggestions")
            improvement_suggestions = [
                "📏 **Increase AB2 range** - Extend measurements to larger electrode spacings",
                "📍 **Add more measurement points** - Improve resolution with additional data",
                "🔍 **Review measurement quality** - Check for systematic errors or noise",
                "⚖️ **Verify electrode configuration** - Ensure proper AB2/MN2 ratios",
                "🧹 **Clean raw data** - Remove obvious outliers and measurement errors"
            ]
            
            for suggestion in improvement_suggestions:
                st.info(suggestion)
    
    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
        st.info("**Troubleshooting tips:**")
        st.info("• Ensure CSV file has proper comma-separated format")
        st.info("• Check that column names are exactly: AB2, MN2, Rhoa")
        st.info("• Verify all data values are numeric")
        st.info("• Remove any header rows or comments from CSV")

else:
    # Show example data format and guidance
    st.markdown("## 📖 Getting Started")
    
    st.info("👆 **Upload a VES CSV file to begin professional data validation**")
    
    # Example data format
    with st.expander("📋 Example Data Format", expanded=True):
        
        example_data = {
            'AB2': [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0],
            'MN2': [0.15, 0.3, 0.6, 1.5, 3.0, 6.0, 15.0, 30.0],
            'Rhoa': [120.5, 85.3, 67.8, 45.2, 78.9, 156.7, 234.5, 189.3]
        }
        
        example_df = pd.DataFrame(example_data)
        st.dataframe(example_df)
        
        st.markdown("**Column Descriptions:**")
        st.markdown("- **AB2**: Half current electrode spacing (meters)")
        st.markdown("- **MN2**: Half potential electrode spacing (meters)")  
        st.markdown("- **Rhoa**: Apparent resistivity (ohm-meters)")
    
    # Data requirements
    with st.expander("📏 Data Requirements & Standards"):
        st.markdown("""
        **Minimum Requirements:**
        - At least 5 measurement points
        - AB2 values in ascending order
        - All values positive and finite
        - AB2/MN2 ratio ≥ 2.0
        
        **Recommended for High Quality:**
        - 15-25 measurement points
        - AB2 range spanning 2+ log decades
        - Measurement density > 3 points per log decade
        - AB2/MN2 ratio between 3-10
        - Noise level < 10%
        
        **Industry Standards (ASTM D6431):**
        - Systematic electrode configuration
        - Proper grounding and contact resistance
        - Multiple readings at each spacing
        - Quality control measurements
        """)
    
    # Sample data generator
    with st.expander("🎲 Generate Sample Data for Testing"):
        st.markdown("Create synthetic VES data to test the application workflow:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            n_points = st.slider("Number of points", 8, 30, 20)
            noise_level = st.slider("Noise level (%)", 0, 10, 3)
        
        with col2:
            min_ab2 = st.number_input("Min AB2 (m)", 0.1, 2.0, 0.5)
            max_ab2 = st.number_input("Max AB2 (m)", 10, 200, 100)
        
        with col3:
            model_type = st.selectbox("Model type", 
                                    ["3-layer (H-type)", "4-layer (K-type)", "2-layer", "Complex"])
        
        if st.button("Generate Sample Data"):
            # Generate synthetic VES data
            ab2_synth = np.logspace(np.log10(min_ab2), np.log10(max_ab2), n_points)
            mn2_synth = ab2_synth / 3  # Standard ratio
            
            # Create synthetic model based on type
            if model_type == "3-layer (H-type)":
                # Surface - Conductor - Basement
                true_res = [100, 20, 500]
                true_thick = [5, 15]
            elif model_type == "4-layer (K-type)":
                # Surface - Conductor - Resistor - Basement  
                true_res = [80, 15, 200, 800]
                true_thick = [3, 8, 20]
            elif model_type == "2-layer":
                # Simple surface over basement
                true_res = [150, 800]
                true_thick = [12]
            else:  # Complex
                true_res = [200, 50, 10, 100, 1000]
                true_thick = [2, 6, 12, 25]
            
            # Simple forward modeling (approximation)
            rhoa_synth = np.zeros_like(ab2_synth)
            cum_depths = np.concatenate([[0], np.cumsum(true_thick), [1000]])
            
            for i, spacing in enumerate(ab2_synth):
                inv_depth = spacing / 3
                
                # Find dominant layer
                layer_weights = np.zeros(len(true_res))
                for j in range(len(true_res)):
                    if inv_depth >= cum_depths[j] and (j == len(true_res)-1 or inv_depth < cum_depths[j+1]):
                        layer_weights[j] = 1.0
                        break
                
                if np.sum(layer_weights) > 0:
                    rhoa_synth[i] = np.average(true_res, weights=layer_weights)
                else:
                    rhoa_synth[i] = true_res[0]
            
            # Add noise
            noise = np.random.normal(1, noise_level/100, len(rhoa_synth))
            rhoa_synth = rhoa_synth * noise
            
            # Create DataFrame
            sample_df = pd.DataFrame({
                'AB2': ab2_synth,
                'MN2': mn2_synth,
                'Rhoa': rhoa_synth
            })
            
            # Store in session state
            st.session_state.uploaded_df = sample_df
            st.session_state.sample_data_info = {
                'model_type': model_type,
                'true_resistivities': true_res,
                'true_thicknesses': true_thick,
                'noise_level': noise_level
            }
            
            st.success(f"Generated {n_points} sample data points!")
            st.info("Sample data is now loaded. The page will refresh to show validation results.")
            st.rerun()
