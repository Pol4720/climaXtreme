"""
📜 Historical Comparison Page
Compare current climate data with historical records and trends.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Optional, List, Dict, Any

try:
    from climaxtreme.dashboard.utils import configure_sidebar, DataSource, show_data_info
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from climaxtreme.dashboard.utils import configure_sidebar, DataSource, show_data_info


def load_data(data_source: DataSource) -> Dict[str, Optional[pd.DataFrame]]:
    """Load historical and synthetic data."""
    data = {}
    
    # Try loading original historical data
    try:
        data['historical'] = data_source.load_csv('GlobalLandTemperaturesByCity.csv')
    except Exception:
        data['historical'] = None
    
    # Try loading processed data
    try:
        data['processed'] = data_source.load_parquet('processed/temperature_data.parquet')
    except Exception:
        data['processed'] = None
    
    # Try loading synthetic data
    try:
        data['synthetic'] = data_source.load_parquet('synthetic/synthetic_hourly.parquet')
        if data['synthetic'] is None:
            data['synthetic'] = data_source.load_parquet('synthetic_hourly.parquet')
    except Exception:
        data['synthetic'] = None
    
    return data


def create_historical_trend_chart(
    df: pd.DataFrame, 
    temperature_col: str,
    date_col: str = 'dt'
) -> go.Figure:
    """Create historical temperature trend with moving averages."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
    
    # Calculate annual averages
    df['year'] = df[date_col].dt.year
    annual = df.groupby('year')[temperature_col].agg(['mean', 'std', 'min', 'max']).reset_index()
    
    fig = go.Figure()
    
    # Confidence band (min-max range)
    fig.add_trace(go.Scatter(
        x=list(annual['year']) + list(annual['year'][::-1]),
        y=list(annual['max']) + list(annual['min'][::-1]),
        fill='toself',
        fillcolor='rgba(52, 152, 219, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Temperature Range'
    ))
    
    # Standard deviation band
    fig.add_trace(go.Scatter(
        x=list(annual['year']) + list(annual['year'][::-1]),
        y=list(annual['mean'] + annual['std']) + list((annual['mean'] - annual['std'])[::-1]),
        fill='toself',
        fillcolor='rgba(231, 76, 60, 0.3)',
        line=dict(color='rgba(255,255,255,0)'),
        name='±1 Std Dev'
    ))
    
    # Mean line
    fig.add_trace(go.Scatter(
        x=annual['year'],
        y=annual['mean'],
        mode='lines+markers',
        line=dict(color='#E74C3C', width=2),
        marker=dict(size=4),
        name='Annual Mean'
    ))
    
    # Add trend line
    z = np.polyfit(annual['year'], annual['mean'], 1)
    p = np.poly1d(z)
    fig.add_trace(go.Scatter(
        x=annual['year'],
        y=p(annual['year']),
        mode='lines',
        line=dict(color='#2C3E50', width=2, dash='dash'),
        name=f'Linear Trend ({z[0]:.3f}°C/year)'
    ))
    
    fig.update_layout(
        title='Historical Temperature Trend',
        xaxis_title='Year',
        yaxis_title='Temperature (°C)',
        height=500,
        hovermode='x unified'
    )
    
    return fig, annual


def create_decade_comparison_chart(df: pd.DataFrame, temperature_col: str) -> go.Figure:
    """Compare temperature distributions by decade."""
    df = df.copy()
    df['decade'] = (df['year'] // 10) * 10
    
    fig = px.box(
        df,
        x='decade',
        y=temperature_col,
        color='decade',
        title='Temperature Distribution by Decade',
        labels={temperature_col: 'Temperature (°C)', 'decade': 'Decade'},
        color_continuous_scale='Reds'
    )
    
    fig.update_layout(height=500)
    fig.update_xaxes(type='category')
    
    return fig


def create_period_comparison_chart(
    df: pd.DataFrame, 
    temperature_col: str,
    baseline_start: int,
    baseline_end: int,
    comparison_start: int,
    comparison_end: int
) -> go.Figure:
    """Compare two time periods."""
    baseline = df[(df['year'] >= baseline_start) & (df['year'] <= baseline_end)]
    comparison = df[(df['year'] >= comparison_start) & (df['year'] <= comparison_end)]
    
    # Monthly averages
    baseline_monthly = baseline.groupby('month')[temperature_col].mean().reset_index()
    baseline_monthly['Period'] = f'Baseline ({baseline_start}-{baseline_end})'
    
    comparison_monthly = comparison.groupby('month')[temperature_col].mean().reset_index()
    comparison_monthly['Period'] = f'Comparison ({comparison_start}-{comparison_end})'
    
    combined = pd.concat([baseline_monthly, comparison_monthly])
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    combined['month_name'] = combined['month'].apply(lambda x: month_names[int(x)-1] if 1 <= x <= 12 else str(x))
    
    fig = px.line(
        combined,
        x='month_name',
        y=temperature_col,
        color='Period',
        markers=True,
        title=f'Monthly Temperature Comparison',
        labels={temperature_col: 'Temperature (°C)', 'month_name': 'Month'}
    )
    
    fig.update_layout(height=450)
    
    # Add difference annotation
    baseline_mean = baseline[temperature_col].mean()
    comparison_mean = comparison[temperature_col].mean()
    diff = comparison_mean - baseline_mean
    
    fig.add_annotation(
        x=0.5, y=1.1,
        xref='paper', yref='paper',
        text=f'Mean Difference: {diff:+.2f}°C',
        showarrow=False,
        font=dict(size=14, color='#E74C3C' if diff > 0 else '#3498DB')
    )
    
    return fig


def create_anomaly_timeline(df: pd.DataFrame, temperature_col: str) -> go.Figure:
    """Create temperature anomaly timeline relative to baseline."""
    df = df.copy()
    
    # Calculate baseline (e.g., 1900-1950)
    baseline = df[(df['year'] >= 1900) & (df['year'] <= 1950)]
    if len(baseline) == 0:
        baseline = df[df['year'] <= df['year'].median()]
    
    baseline_mean = baseline[temperature_col].mean()
    
    # Calculate annual anomalies
    annual = df.groupby('year')[temperature_col].mean().reset_index()
    annual['anomaly'] = annual[temperature_col] - baseline_mean
    
    fig = go.Figure()
    
    # Color by anomaly (red for positive, blue for negative)
    colors = ['#E74C3C' if a > 0 else '#3498DB' for a in annual['anomaly']]
    
    fig.add_trace(go.Bar(
        x=annual['year'],
        y=annual['anomaly'],
        marker_color=colors,
        name='Temperature Anomaly'
    ))
    
    # Zero line
    fig.add_hline(y=0, line_dash='dash', line_color='black')
    
    # Add trend
    z = np.polyfit(annual['year'], annual['anomaly'], 1)
    p = np.poly1d(z)
    fig.add_trace(go.Scatter(
        x=annual['year'],
        y=p(annual['year']),
        mode='lines',
        line=dict(color='#2C3E50', width=2),
        name=f'Trend ({z[0]*10:.2f}°C/decade)'
    ))
    
    fig.update_layout(
        title='Temperature Anomaly Timeline (Relative to Historical Baseline)',
        xaxis_title='Year',
        yaxis_title='Temperature Anomaly (°C)',
        height=500
    )
    
    return fig


def create_warming_stripes(df: pd.DataFrame, temperature_col: str) -> go.Figure:
    """Create climate warming stripes visualization."""
    annual = df.groupby('year')[temperature_col].mean().reset_index()
    
    # Normalize to colormap
    vmin = annual[temperature_col].min()
    vmax = annual[temperature_col].max()
    
    fig = go.Figure()
    
    for i, (_, row) in enumerate(annual.iterrows()):
        fig.add_shape(
            type='rect',
            x0=i - 0.5, x1=i + 0.5,
            y0=0, y1=1,
            fillcolor=px.colors.sample_colorscale(
                'RdBu_r',
                [(row[temperature_col] - vmin) / (vmax - vmin)]
            )[0],
            line=dict(width=0)
        )
    
    fig.update_layout(
        title='Climate Warming Stripes',
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(0, len(annual), max(1, len(annual)//10))),
            ticktext=[str(annual.iloc[i]['year']) for i in range(0, len(annual), max(1, len(annual)//10))],
            title=None
        ),
        yaxis=dict(visible=False),
        height=200,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig


def create_location_comparison(df: pd.DataFrame, temperature_col: str) -> go.Figure:
    """Compare temperature trends across locations."""
    if 'City' not in df.columns:
        return None
    
    # Get top 5 cities by data count
    top_cities = df['City'].value_counts().head(5).index.tolist()
    df_filtered = df[df['City'].isin(top_cities)]
    
    # Annual averages by city
    annual = df_filtered.groupby(['year', 'City'])[temperature_col].mean().reset_index()
    
    fig = px.line(
        annual,
        x='year',
        y=temperature_col,
        color='City',
        title='Temperature Trends by Location',
        labels={temperature_col: 'Temperature (°C)', 'year': 'Year'}
    )
    
    fig.update_layout(height=500)
    
    return fig


def main():
    st.set_page_config(
        page_title="Historical Comparison - climaXtreme",
        page_icon="📜",
        layout="wide"
    )
    
    configure_sidebar()
    
    st.title("📜 Historical Climate Comparison")
    st.markdown("""
    Compare current climate conditions with historical records to understand 
    long-term trends and changes in temperature patterns.
    """)
    
    # Load data
    data_source = DataSource()
    
    with st.spinner("Loading data..."):
        data = load_data(data_source)
    
    # Determine which data to use
    df = None
    temperature_col = 'AverageTemperature'
    date_col = 'dt'
    
    if data['historical'] is not None:
        df = data['historical'].copy()
        st.success(f"✅ Loaded historical data: {len(df):,} records")
    elif data['processed'] is not None:
        df = data['processed'].copy()
        st.success(f"✅ Loaded processed data: {len(df):,} records")
    elif data['synthetic'] is not None:
        df = data['synthetic'].copy()
        if 'temperature_hourly' in df.columns:
            temperature_col = 'temperature_hourly'
        if 'timestamp' in df.columns:
            date_col = 'timestamp'
        st.info("📊 Using synthetic data for demonstration")
    
    if df is None or df.empty:
        st.warning("""
        ⚠️ **No climate data found!**
        
        Please ensure you have data available:
        - Original: `DATA/GlobalLandTemperaturesByCity.csv`
        - Or generate synthetic data with `climaxtreme generate-synthetic`
        """)
        
        # Demo mode
        st.markdown("---")
        st.subheader("📊 Demo Mode")
        
        np.random.seed(42)
        years = range(1900, 2024)
        n_cities = 10
        records = []
        
        for year in years:
            for month in range(1, 13):
                for city_id in range(n_cities):
                    # Add warming trend
                    warming = (year - 1900) * 0.01
                    seasonal = 15 * np.sin(2 * np.pi * (month - 1) / 12)
                    base_temp = 15 + city_id * 2
                    temp = base_temp + seasonal + warming + np.random.normal(0, 2)
                    
                    records.append({
                        'dt': f'{year}-{month:02d}-01',
                        'AverageTemperature': temp,
                        'City': f'City_{city_id}',
                        'Country': 'Demo Country'
                    })
        
        df = pd.DataFrame(records)
        st.info("Showing demo data for illustration")
    
    # Ensure date parsing
    df[date_col] = pd.to_datetime(df[date_col])
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    
    # Sidebar filters
    st.sidebar.markdown("### 🎛️ Analysis Filters")
    
    # Year range
    years = sorted(df['year'].unique())
    year_range = st.sidebar.slider(
        "Year Range",
        min_value=int(min(years)),
        max_value=int(max(years)),
        value=(int(min(years)), int(max(years)))
    )
    
    df = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]
    
    # Location filter
    if 'City' in df.columns:
        cities = ['All'] + sorted(df['City'].dropna().unique().tolist())[:50]
        selected_city = st.sidebar.selectbox("Filter by City", cities)
        if selected_city != 'All':
            df = df[df['City'] == selected_city]
    
    # Remove missing values
    df = df.dropna(subset=[temperature_col])
    
    st.info(f"📊 Analyzing {len(df):,} records from {year_range[0]} to {year_range[1]}")
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Long-term Trend",
        "📊 Decade Comparison",
        "🔄 Period Comparison",
        "🌡️ Anomaly Timeline",
        "🌍 Location Comparison"
    ])
    
    with tab1:
        st.subheader("Historical Temperature Trend")
        
        fig_trend, annual = create_historical_trend_chart(df, temperature_col, date_col)
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # Summary statistics
        st.markdown("### Trend Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            first_decade = annual[annual['year'] < annual['year'].min() + 10]['mean'].mean()
            last_decade = annual[annual['year'] > annual['year'].max() - 10]['mean'].mean()
            change = last_decade - first_decade
            st.metric(
                "Total Change",
                f"{change:+.2f}°C",
                delta=f"{change/len(annual.year.unique())*100:.1f}°C/century"
            )
        
        with col2:
            st.metric("Mean Temperature", f"{annual['mean'].mean():.2f}°C")
        
        with col3:
            st.metric("Min Annual Mean", f"{annual['mean'].min():.2f}°C")
        
        with col4:
            st.metric("Max Annual Mean", f"{annual['mean'].max():.2f}°C")
        
        # Warming stripes
        st.markdown("### Climate Stripes")
        fig_stripes = create_warming_stripes(df, temperature_col)
        st.plotly_chart(fig_stripes, use_container_width=True)
    
    with tab2:
        st.subheader("Temperature by Decade")
        
        fig_decades = create_decade_comparison_chart(df, temperature_col)
        st.plotly_chart(fig_decades, use_container_width=True)
        
        # Decade statistics
        st.markdown("### Decade Statistics")
        df_decade = df.copy()
        df_decade['decade'] = (df_decade['year'] // 10) * 10
        decade_stats = df_decade.groupby('decade')[temperature_col].agg(['mean', 'std', 'min', 'max']).round(2)
        decade_stats.columns = ['Mean', 'Std Dev', 'Minimum', 'Maximum']
        st.dataframe(decade_stats, use_container_width=True)
    
    with tab3:
        st.subheader("Period Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Baseline Period**")
            baseline_start = st.number_input("Start Year", value=int(min(years)), key='baseline_start')
            baseline_end = st.number_input("End Year", value=int(min(years)) + 30, key='baseline_end')
        
        with col2:
            st.markdown("**Comparison Period**")
            comparison_start = st.number_input("Start Year", value=int(max(years)) - 30, key='comp_start')
            comparison_end = st.number_input("End Year", value=int(max(years)), key='comp_end')
        
        if st.button("Compare Periods"):
            fig_periods = create_period_comparison_chart(
                df, temperature_col,
                int(baseline_start), int(baseline_end),
                int(comparison_start), int(comparison_end)
            )
            st.plotly_chart(fig_periods, use_container_width=True)
            
            # Detailed comparison
            baseline_data = df[(df['year'] >= baseline_start) & (df['year'] <= baseline_end)]
            comparison_data = df[(df['year'] >= comparison_start) & (df['year'] <= comparison_end)]
            
            col1, col2, col3 = st.columns(3)
            
            baseline_mean = baseline_data[temperature_col].mean()
            comparison_mean = comparison_data[temperature_col].mean()
            diff = comparison_mean - baseline_mean
            
            with col1:
                st.metric("Baseline Mean", f"{baseline_mean:.2f}°C")
            with col2:
                st.metric("Comparison Mean", f"{comparison_mean:.2f}°C")
            with col3:
                st.metric("Difference", f"{diff:+.2f}°C", delta_color="inverse" if diff > 0 else "normal")
    
    with tab4:
        st.subheader("Temperature Anomaly Timeline")
        
        fig_anomaly = create_anomaly_timeline(df, temperature_col)
        st.plotly_chart(fig_anomaly, use_container_width=True)
        
        st.markdown("""
        **Note:** Anomalies are calculated relative to the historical baseline period 
        (first half of available data). Positive anomalies (red) indicate warmer than 
        average years, while negative anomalies (blue) indicate cooler years.
        """)
    
    with tab5:
        st.subheader("Temperature Trends by Location")
        
        fig_location = create_location_comparison(df, temperature_col)
        if fig_location:
            st.plotly_chart(fig_location, use_container_width=True)
        else:
            st.info("Location comparison requires 'City' column in data")
    
    # Key findings
    st.markdown("---")
    st.subheader("📋 Key Findings")
    
    # Calculate warming rate
    annual = df.groupby('year')[temperature_col].mean().reset_index()
    z = np.polyfit(annual['year'], annual[temperature_col], 1)
    warming_rate = z[0] * 100  # Per century
    
    findings = []
    
    if warming_rate > 0:
        findings.append(f"📈 **Warming Trend:** Temperature is increasing at approximately {warming_rate:.2f}°C per century")
    else:
        findings.append(f"📉 **Cooling Trend:** Temperature is decreasing at approximately {abs(warming_rate):.2f}°C per century")
    
    # Recent vs historical
    if len(annual) > 20:
        early = annual.head(10)[temperature_col].mean()
        recent = annual.tail(10)[temperature_col].mean()
        findings.append(f"🌡️ **Historical Change:** Recent decade is {recent - early:+.2f}°C compared to earliest decade")
    
    # Warmest year
    warmest_year = annual.loc[annual[temperature_col].idxmax()]
    findings.append(f"🔥 **Warmest Year:** {int(warmest_year['year'])} with {warmest_year[temperature_col]:.2f}°C average")
    
    for finding in findings:
        st.markdown(finding)


if __name__ == "__main__":
    main()
