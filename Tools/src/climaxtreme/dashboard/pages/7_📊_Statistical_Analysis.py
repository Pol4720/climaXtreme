"""
Statistical Analysis Page - Correlations, Chi-square tests, and Distribution Analysis
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import sys
from pathlib import Path

try:
    from climaxtreme.dashboard.utils import DataSource, configure_sidebar, show_data_info
except ImportError:
    _src_dir = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(_src_dir))
    from climaxtreme.dashboard.utils import DataSource, configure_sidebar, show_data_info

st.set_page_config(page_title="Statistical Analysis", page_icon="📊", layout="wide")
configure_sidebar()

st.title("📊 Statistical Analysis")
st.markdown("Deep dive into correlations, distributions, and statistical tests")

data_source = DataSource()

tab1, tab2, tab3 = st.tabs(["📈 Descriptive Statistics", "🔗 Correlations", "🧪 Chi-Square Tests"])

# TAB 1: Descriptive Statistics
with tab1:
    st.subheader("Descriptive Statistics")
    
    stats_df = data_source.load_parquet('descriptive_stats.parquet')
    
    if stats_df is not None and not stats_df.empty:
        show_data_info(stats_df, "Descriptive Statistics Dataset")
        
        # Pivot stats to wide format for easier processing
        stats_wide = stats_df.pivot(index='variable', columns='statistic', values='value').reset_index()
        
        # Display statistics nicely
        for idx, row in stats_wide.iterrows():
            variable = row['variable']
            
            st.markdown(f"### 📊 {variable.replace('_', ' ').title()}")
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Count", f"{int(row.get('count', 0)):,}")
            with col2:
                st.metric("Mean", f"{row.get('mean', 0):.2f}")
            with col3:
                st.metric("Std Dev", f"{row.get('std_dev', 0):.2f}")
            with col4:
                st.metric("Range", f"{row.get('max', 0) - row.get('min', 0):.2f}")
            
            # Distribution info
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Quartiles:**")
                st.write(f"- Min: {row.get('min', 0):.2f}")
                st.write(f"- Q1: {row.get('q1', 0):.2f}")
                st.write(f"- Median: {row.get('median', 0):.2f}")
                st.write(f"- Q3: {row.get('q3', 0):.2f}")
                st.write(f"- Max: {row.get('max', 0):.2f}")
            
            with col2:
                st.markdown("**Distribution Shape:**")
                skew = row.get('skewness', 0)
                kurt = row.get('kurtosis', 0)
                
                st.write(f"- Skewness: {skew:.3f}")
                if skew < -0.5:
                    st.caption("  ← Left-skewed")
                elif skew > 0.5:
                    st.caption("  → Right-skewed")
                else:
                    st.caption("  ≈ Symmetric")
                
                st.write(f"- Kurtosis: {kurt:.3f}")
                if kurt > 0:
                    st.caption("  ↑ Heavy-tailed")
                else:
                    st.caption("  ↓ Light-tailed")
            
            with col3:
                st.markdown("**Other Stats:**")
                iqr = row.get('iqr', 0)
                st.write(f"- IQR: {iqr:.2f}")
                
                # Outlier detection using IQR method
                q1 = row.get('q1', 0)
                q3 = row.get('q3', 0)
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                st.write(f"- Outlier bounds:")
                st.caption(f"  [{lower_bound:.2f}, {upper_bound:.2f}]")
            
            # Visualization: Box plot and Violin plot side by side
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                # Box plot
                fig_box = go.Figure()
                
                fig_box.add_trace(go.Box(
                    q1=[row.get('q1', 0)],
                    median=[row.get('median', 0)],
                    q3=[row.get('q3', 0)],
                    lowerfence=[row.get('min', 0)],
                    upperfence=[row.get('max', 0)],
                    mean=[row.get('mean', 0)],
                    name=variable,
                    boxmean='sd',
                    orientation='h',
                    marker=dict(color='lightblue')
                ))
                
                fig_box.update_layout(
                    title=f"Box Plot - {variable.replace('_', ' ').title()}",
                    xaxis_title="Value",
                    height=250,
                    showlegend=False
                )
                
                st.plotly_chart(fig_box, use_container_width=True)
            
            with viz_col2:
                # Violin plot (simulated from statistics)
                # Generate synthetic data points based on quartiles for visualization
                n_points = 200
                
                # Generate points using normal distribution approximation
                mean = row.get('mean', 0)
                std = row.get('std_dev', 1)
                
                # Generate data with proper distribution shape
                synthetic_data = np.random.normal(mean, std, n_points)
                
                # Clip to min/max bounds
                synthetic_data = np.clip(synthetic_data, row.get('min', mean-3*std), row.get('max', mean+3*std))
                
                fig_violin = go.Figure()
                
                fig_violin.add_trace(go.Violin(
                    y=synthetic_data,
                    name=variable,
                    box_visible=True,
                    meanline_visible=True,
                    fillcolor='lightgreen',
                    opacity=0.6,
                    x0=variable
                ))
                
                fig_violin.update_layout(
                    title=f"Violin Plot - {variable.replace('_', ' ').title()}",
                    yaxis_title="Value",
                    height=250,
                    showlegend=False
                )
                
                st.plotly_chart(fig_violin, use_container_width=True)
            
            st.markdown("---")
    
    else:
        st.error("❌ Failed to load descriptive_stats.parquet")

# TAB 2: Correlations
with tab2:
    st.subheader("Correlation Matrix")
    
    corr_df = data_source.load_parquet('correlation_matrix.parquet')
    
    if corr_df is not None and not corr_df.empty:
        show_data_info(corr_df, "Correlation Matrix Dataset")
        
        # Create pivot table for heatmap
        variables = sorted(corr_df['variable_1'].unique())
        
        corr_matrix = pd.DataFrame(index=variables, columns=variables)
        
        for _, row in corr_df.iterrows():
            var1 = row['variable_1']
            var2 = row['variable_2']
            corr = row['correlation']
            
            corr_matrix.loc[var1, var2] = corr
            corr_matrix.loc[var2, var1] = corr
        
        corr_matrix = corr_matrix.astype(float)
        
        # Correlation heatmap
        st.markdown("#### Correlation Heatmap")
        
        fig = px.imshow(
            corr_matrix,
            labels=dict(color="Correlation"),
            x=variables,
            y=variables,
            color_continuous_scale='RdBu',
            zmin=-1,
            zmax=1,
            title="Pearson Correlation Matrix",
            aspect="auto"
        )
        
        fig.update_layout(height=600)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Interpretation guide
        st.markdown("#### 📖 Correlation Interpretation")
        st.info("""
        **Correlation Coefficient (r) ranges from -1 to +1:**
        - **r = +1**: Perfect positive correlation
        - **r = 0**: No linear correlation
        - **r = -1**: Perfect negative correlation
        
        **Strength:**
        - |r| > 0.7: Strong correlation
        - |r| > 0.4: Moderate correlation
        - |r| < 0.3: Weak correlation
        """)
        
        # Top correlations
        st.markdown("#### Top Correlations")
        
        # Filter out diagonal
        top_corr = corr_df[corr_df['variable_1'] != corr_df['variable_2']].copy()
        
        # Sort by absolute correlation
        top_corr = top_corr.sort_values('abs_correlation', ascending=False).head(10)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Strongest Positive:**")
            positive = top_corr[top_corr['correlation'] > 0].head(5)
            for _, row in positive.iterrows():
                st.write(f"• {row['variable_1']} ↔ {row['variable_2']}: **{row['correlation']:.3f}**")
        
        with col2:
            st.markdown("**Strongest Negative:**")
            negative = top_corr[top_corr['correlation'] < 0].sort_values('correlation').head(5)
            for _, row in negative.iterrows():
                st.write(f"• {row['variable_1']} ↔ {row['variable_2']}: **{row['correlation']:.3f}**")
        
        # Full correlation table
        st.markdown("#### Complete Correlation Table")
        
        display_df = corr_df.sort_values('abs_correlation', ascending=False).copy()
        display_df = display_df[['variable_1', 'variable_2', 'correlation', 'abs_correlation']]
        display_df.columns = ['Variable 1', 'Variable 2', 'Correlation', 'Abs Correlation']
        display_df = display_df.round(3)
        
        st.dataframe(display_df, hide_index=True, use_container_width=True, height=400)
    
    else:
        st.error("❌ Failed to load correlation_matrix.parquet")

# TAB 3: Chi-Square Tests
with tab3:
    st.subheader("Chi-Square Independence Tests")
    
    chi_df = data_source.load_parquet('chi_square_tests.parquet')
    
    if chi_df is not None and not chi_df.empty:
        show_data_info(chi_df, "Chi-Square Tests Dataset")
        
        st.markdown("#### 📖 Chi-Square Test Interpretation")
        st.info("""
        **Chi-square tests assess independence between categorical variables:**
        - **H₀ (Null Hypothesis)**: Variables are independent
        - **H₁ (Alternative)**: Variables are dependent
        
        **Decision Rule:**
        - **p-value < 0.05**: Reject H₀ → Variables are dependent (✅ Significant)
        - **p-value ≥ 0.05**: Don't reject H₀ → Variables are independent
        
        **About Temperature Category:**
        - `temperature_category` is a derived categorical variable
        - Categorizes temperatures as: **Cold** (< 10°C), **Moderate** (10-20°C), **Hot** (> 20°C)
        - Used to test relationships between geographic/temporal factors and temperature ranges
        """)
        
        # Display tests
        for idx, row in chi_df.iterrows():
            test_name = row['test']
            var1 = row['variable_1']
            var2 = row['variable_2']
            
            with st.expander(f"📊 {test_name}", expanded=True):
                # Test info
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**Variables:** `{var1}` × `{var2}`")
                    
                    # Add explanation for temperature_category
                    if var2 == 'temperature_category':
                        st.caption("💡 Temperature category groups temperatures into Cold/Moderate/Hot ranges")
                    
                    # Metrics
                    mcol1, mcol2, mcol3 = st.columns(3)
                    with mcol1:
                        st.metric("χ² Statistic", f"{row['chi_square_statistic']:.2f}")
                    with mcol2:
                        st.metric("p-value", f"{row['p_value']:.4f}")
                    with mcol3:
                        st.metric("df", int(row['degrees_of_freedom']))
                
                with col2:
                    # Result
                    is_sig = row['is_significant']
                    if is_sig:
                        st.success("✅ **SIGNIFICANT**")
                        st.write("Variables are dependent")
                    else:
                        st.info("❌ **NOT SIGNIFICANT**")
                        st.write("Variables are independent")
                
                # Interpretation
                st.markdown("**Interpretation:**")
                
                # Get readable variable names
                var1_name = var1.replace('_', ' ').title()
                var2_name = "Temperature Category (Cold/Moderate/Hot)" if var2 == 'temperature_category' else var2.replace('_', ' ').title()
                
                if is_sig:
                    st.write(f"There is a **statistically significant relationship** between "
                            f"**{var1_name}** and **{var2_name}** (p < 0.05). "
                            f"This means that the distribution of {var2_name.lower()} varies significantly "
                            f"across different categories of {var1_name.lower()}.")
                    
                    if var2 == 'temperature_category':
                        st.caption("🔍 Example: Different continents/seasons have different proportions of cold/moderate/hot temperatures")
                else:
                    st.write(f"There is **no statistically significant relationship** between "
                            f"**{var1_name}** and **{var2_name}** (p ≥ 0.05). "
                            f"The variables appear to be independent.")
                    
                    if var2 == 'temperature_category':
                        st.caption("🔍 Example: Temperature categories are distributed similarly across all groups")
        
        # Summary table
        st.markdown("#### Test Results Summary")
        
        display_df = chi_df[['test', 'variable_1', 'variable_2', 'chi_square_statistic', 
                             'p_value', 'degrees_of_freedom', 'is_significant']].copy()
        display_df.columns = ['Test', 'Variable 1', 'Variable 2', 'χ² Statistic', 
                             'p-value', 'df', 'Significant']
        display_df['Significant'] = display_df['Significant'].map({True: '✅ Yes', False: '❌ No'})
        display_df = display_df.round({'χ² Statistic': 2, 'p-value': 4})
        
        st.dataframe(display_df, hide_index=True, use_container_width=True)
    
    else:
        st.error("❌ Failed to load chi_square_tests.parquet")
