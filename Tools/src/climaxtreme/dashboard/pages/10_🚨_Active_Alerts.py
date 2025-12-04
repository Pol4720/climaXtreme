"""
🚨 Active Alerts Page
Real-time weather alerts dashboard with severity levels.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
from typing import Optional

try:
    from climaxtreme.dashboard.utils import configure_sidebar, DataSource, show_data_info
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from climaxtreme.dashboard.utils import configure_sidebar, DataSource, show_data_info


# Alert styling
ALERT_STYLES = {
    'EMERGENCY': {'color': '#C0392B', 'icon': '🔴', 'bg': '#FADBD8'},
    'WARNING': {'color': '#E67E22', 'icon': '🟠', 'bg': '#FDEBD0'},
    'WATCH': {'color': '#F1C40F', 'icon': '🟡', 'bg': '#FEF9E7'},
    'NONE': {'color': '#27AE60', 'icon': '🟢', 'bg': '#D5F5E3'}
}

ALERT_TYPE_ICONS = {
    'HEAT': '🔥',
    'COLD': '❄️',
    'STORM': '🌀',
    'FLOOD': '🌊',
    'WIND': '💨',
    'WEATHER': '⛈️',
    'NONE': '✅'
}


def load_alerts_data(data_source: DataSource) -> Optional[pd.DataFrame]:
    """Load alerts history data."""
    try:
        df = data_source.load_parquet('alerts_history.parquet')
        if df is None:
            df = data_source.load_parquet('synthetic/alerts_history.parquet')
        return df
    except Exception as e:
        return None


def load_synthetic_data(data_source: DataSource) -> Optional[pd.DataFrame]:
    """Load synthetic hourly data for alerts."""
    try:
        df = data_source.load_parquet('synthetic_hourly.parquet')
        if df is None:
            df = data_source.load_parquet('synthetic/synthetic_hourly.parquet')
        return df
    except Exception as e:
        return None


def create_alert_card(alert_row: pd.Series) -> str:
    """Create HTML for an alert card."""
    alert_level = alert_row.get('alert_level', 'WATCH')
    alert_type = alert_row.get('alert_type', 'WEATHER')
    
    style = ALERT_STYLES.get(alert_level, ALERT_STYLES['WATCH'])
    icon = ALERT_TYPE_ICONS.get(alert_type, '⚠️')
    
    city = alert_row.get('City', 'Unknown')
    country = alert_row.get('Country', '')
    temp = alert_row.get('temperature_hourly', 0)
    wind = alert_row.get('wind_speed_kmh', 0)
    rain = alert_row.get('rain_mm', 0)
    intensity = alert_row.get('event_intensity', 0)
    
    timestamp = alert_row.get('timestamp', alert_row.get('alert_issued_at', 'N/A'))
    if isinstance(timestamp, pd.Timestamp):
        timestamp = timestamp.strftime('%Y-%m-%d %H:%M')
    
    return f"""
    <div style='
        background-color:{style["bg"]}; 
        border-left: 4px solid {style["color"]}; 
        padding: 15px; 
        margin: 10px 0; 
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    '>
        <div style='display: flex; justify-content: space-between; align-items: center;'>
            <div>
                <span style='font-size: 24px;'>{icon} {style["icon"]}</span>
                <strong style='font-size: 18px; color: {style["color"]};'> {alert_level}</strong>
                <span style='color: #666;'> - {alert_type}</span>
            </div>
            <span style='color: #888; font-size: 12px;'>{timestamp}</span>
        </div>
        <div style='margin-top: 10px;'>
            <strong>{city}</strong>, {country}
        </div>
        <div style='margin-top: 5px; color: #555; font-size: 14px;'>
            🌡️ {temp:.1f}°C | 💨 {wind:.1f} km/h | 🌧️ {rain:.1f} mm | ⚡ Intensity: {intensity:.2f}
        </div>
    </div>
    """


def create_alerts_map(df: pd.DataFrame) -> go.Figure:
    """Create a map showing alert locations."""
    # Get unique alert colors
    color_map = {level: style['color'] for level, style in ALERT_STYLES.items()}
    
    fig = go.Figure()
    
    for level in ['EMERGENCY', 'WARNING', 'WATCH']:
        level_df = df[df['alert_level'] == level]
        if not level_df.empty:
            lat_col = 'lat_decimal' if 'lat_decimal' in level_df.columns else 'latitude'
            lon_col = 'lon_decimal' if 'lon_decimal' in level_df.columns else 'longitude'
            
            fig.add_trace(go.Scattergeo(
                lat=level_df[lat_col],
                lon=level_df[lon_col],
                mode='markers',
                marker=dict(
                    size=8 if level == 'WATCH' else (12 if level == 'WARNING' else 16),
                    color=color_map[level],
                    opacity=0.8,
                    line=dict(width=1, color='white')
                ),
                name=f"{ALERT_STYLES[level]['icon']} {level}",
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>" +
                    "Type: %{customdata[1]}<br>" +
                    "Level: " + level + "<br>" +
                    "<extra></extra>"
                ),
                customdata=level_df[['City', 'alert_type']].values if 'City' in level_df.columns else None
            ))
    
    fig.update_layout(
        title="🗺️ Active Alerts Map",
        geo=dict(
            showland=True,
            landcolor='rgb(243, 243, 243)',
            countrycolor='rgb(204, 204, 204)',
            showocean=True,
            oceancolor='rgb(230, 245, 255)',
            showcountries=True,
            projection_type='natural earth'
        ),
        height=500,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    return fig


def create_alerts_timeline(df: pd.DataFrame) -> go.Figure:
    """Create a timeline of alerts."""
    # Aggregate by time period
    df = df.copy()
    
    if 'timestamp' in df.columns:
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
    elif 'alert_issued_at' in df.columns:
        df['date'] = pd.to_datetime(df['alert_issued_at']).dt.date
    else:
        return go.Figure().add_annotation(text="No timestamp data", showarrow=False)
    
    daily_counts = df.groupby(['date', 'alert_level']).size().reset_index(name='count')
    
    fig = px.bar(
        daily_counts,
        x='date',
        y='count',
        color='alert_level',
        title='📅 Alerts Timeline',
        color_discrete_map={level: style['color'] for level, style in ALERT_STYLES.items()}
    )
    
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Number of Alerts',
        legend_title='Alert Level',
        height=400
    )
    
    return fig


def create_alert_types_chart(df: pd.DataFrame) -> go.Figure:
    """Create a chart showing distribution of alert types."""
    type_counts = df['alert_type'].value_counts().reset_index()
    type_counts.columns = ['Alert Type', 'Count']
    
    # Add icons to labels
    type_counts['Label'] = type_counts['Alert Type'].apply(
        lambda x: f"{ALERT_TYPE_ICONS.get(x, '⚠️')} {x}"
    )
    
    colors = ['#E74C3C', '#3498DB', '#9B59B6', '#1ABC9C', '#F39C12', '#95A5A6']
    
    fig = px.pie(
        type_counts,
        values='Count',
        names='Label',
        title='🎯 Alert Types Distribution',
        color_discrete_sequence=colors
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=400)
    
    return fig


def create_severity_gauge(emergency_pct: float, warning_pct: float, watch_pct: float) -> go.Figure:
    """Create a severity gauge indicator."""
    # Calculate weighted severity score (0-100)
    severity_score = emergency_pct * 100 + warning_pct * 50 + watch_pct * 20
    severity_score = min(100, severity_score)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=severity_score,
        title={'text': "Overall Severity Index"},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#2C3E50"},
            'steps': [
                {'range': [0, 33], 'color': "#D5F5E3"},
                {'range': [33, 66], 'color': "#FDEBD0"},
                {'range': [66, 100], 'color': "#FADBD8"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig


def main():
    st.set_page_config(
        page_title="Active Alerts - climaXtreme",
        page_icon="🚨",
        layout="wide"
    )
    
    configure_sidebar()
    
    st.title("🚨 Active Weather Alerts")
    st.markdown("""
    Real-time weather alert monitoring dashboard. View active warnings, their severity,
    and affected locations across the globe.
    """)
    
    # Load data
    data_source = DataSource()
    
    with st.spinner("Loading alert data..."):
        alerts_df = load_alerts_data(data_source)
        synthetic_df = load_synthetic_data(data_source)
    
    # Extract alerts from synthetic data if dedicated alerts file not found
    if alerts_df is None and synthetic_df is not None:
        if 'alert_active' in synthetic_df.columns:
            alerts_df = synthetic_df[synthetic_df['alert_active'] == True].copy()
    
    if alerts_df is None or alerts_df.empty:
        st.warning("""
        ⚠️ **No alert data found!**
        
        Please generate synthetic data first:
        ```bash
        climaxtreme generate-synthetic --input-path DATA/GlobalLandTemperaturesByCity.csv --output-path DATA/synthetic
        ```
        """)
        
        # Demo mode
        st.markdown("---")
        st.subheader("📊 Demo Mode")
        
        np.random.seed(42)
        demo_alerts = pd.DataFrame({
            'City': ['Miami', 'Tokyo', 'Sydney', 'London', 'Cairo', 'Mumbai'],
            'Country': ['USA', 'Japan', 'Australia', 'UK', 'Egypt', 'India'],
            'lat_decimal': [25.7, 35.7, -33.9, 51.5, 30.0, 19.1],
            'lon_decimal': [-80.2, 139.7, 151.2, -0.1, 31.2, 72.9],
            'alert_level': ['EMERGENCY', 'WARNING', 'WATCH', 'WARNING', 'EMERGENCY', 'WATCH'],
            'alert_type': ['STORM', 'HEAT', 'WIND', 'FLOOD', 'HEAT', 'STORM'],
            'temperature_hourly': [32, 38, 28, 18, 45, 35],
            'wind_speed_kmh': [150, 30, 80, 60, 25, 90],
            'rain_mm': [100, 0, 5, 50, 0, 80],
            'event_intensity': [0.85, 0.6, 0.4, 0.55, 0.9, 0.5],
            'timestamp': pd.date_range('2024-01-01', periods=6, freq='6h')
        })
        alerts_df = demo_alerts
        st.info("Showing demo data for illustration")
    
    # Alert summary metrics
    st.markdown("---")
    
    total_alerts = len(alerts_df)
    emergency_count = len(alerts_df[alerts_df['alert_level'] == 'EMERGENCY'])
    warning_count = len(alerts_df[alerts_df['alert_level'] == 'WARNING'])
    watch_count = len(alerts_df[alerts_df['alert_level'] == 'WATCH'])
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; text-align: center; color: white;'>
            <h2 style='margin: 0;'>{total_alerts:,}</h2>
            <p style='margin: 5px 0 0 0;'>Total Alerts</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style='background: {ALERT_STYLES["EMERGENCY"]["color"]}; padding: 20px; border-radius: 10px; text-align: center; color: white;'>
            <h2 style='margin: 0;'>🔴 {emergency_count}</h2>
            <p style='margin: 5px 0 0 0;'>Emergency</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style='background: {ALERT_STYLES["WARNING"]["color"]}; padding: 20px; border-radius: 10px; text-align: center; color: white;'>
            <h2 style='margin: 0;'>🟠 {warning_count}</h2>
            <p style='margin: 5px 0 0 0;'>Warning</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div style='background: #F1C40F; padding: 20px; border-radius: 10px; text-align: center; color: #333;'>
            <h2 style='margin: 0;'>🟡 {watch_count}</h2>
            <p style='margin: 5px 0 0 0;'>Watch</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Tabs for different views
    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs([
        "🗺️ Alerts Map", 
        "📋 Alert Feed", 
        "📊 Analytics",
        "🔍 Search"
    ])
    
    with tab1:
        st.subheader("Global Alerts Map")
        fig_map = create_alerts_map(alerts_df)
        st.plotly_chart(fig_map, use_container_width=True)
    
    with tab2:
        st.subheader("Real-Time Alert Feed")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            level_filter = st.multiselect(
                "Filter by Level",
                options=['EMERGENCY', 'WARNING', 'WATCH'],
                default=['EMERGENCY', 'WARNING', 'WATCH']
            )
        
        with col2:
            if 'alert_type' in alerts_df.columns:
                type_filter = st.multiselect(
                    "Filter by Type",
                    options=alerts_df['alert_type'].unique().tolist(),
                    default=alerts_df['alert_type'].unique().tolist()
                )
            else:
                type_filter = None
        
        with col3:
            max_alerts = st.slider("Max alerts to show", 5, 50, 20)
        
        # Apply filters
        filtered_df = alerts_df[alerts_df['alert_level'].isin(level_filter)]
        if type_filter:
            filtered_df = filtered_df[filtered_df['alert_type'].isin(type_filter)]
        
        # Sort by severity and recency
        severity_order = {'EMERGENCY': 0, 'WARNING': 1, 'WATCH': 2, 'NONE': 3}
        filtered_df = filtered_df.copy()
        filtered_df['severity_order'] = filtered_df['alert_level'].map(severity_order)
        filtered_df = filtered_df.sort_values('severity_order').head(max_alerts)
        
        # Display alert cards
        st.markdown(f"**Showing {len(filtered_df)} alerts**")
        
        for _, row in filtered_df.iterrows():
            st.markdown(create_alert_card(row), unsafe_allow_html=True)
    
    with tab3:
        st.subheader("Alert Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Alert types distribution
            fig_types = create_alert_types_chart(alerts_df)
            st.plotly_chart(fig_types, use_container_width=True)
        
        with col2:
            # Severity gauge
            emergency_pct = emergency_count / total_alerts if total_alerts > 0 else 0
            warning_pct = warning_count / total_alerts if total_alerts > 0 else 0
            watch_pct = watch_count / total_alerts if total_alerts > 0 else 0
            
            fig_gauge = create_severity_gauge(emergency_pct, warning_pct, watch_pct)
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Timeline
        fig_timeline = create_alerts_timeline(alerts_df)
        st.plotly_chart(fig_timeline, use_container_width=True)
        
        # By country
        if 'Country' in alerts_df.columns:
            st.subheader("Alerts by Country")
            country_counts = alerts_df['Country'].value_counts().head(15).reset_index()
            country_counts.columns = ['Country', 'Alerts']
            
            fig_country = px.bar(
                country_counts,
                x='Country',
                y='Alerts',
                title='Top 15 Countries by Alert Count',
                color='Alerts',
                color_continuous_scale='Reds'
            )
            fig_country.update_layout(height=400)
            st.plotly_chart(fig_country, use_container_width=True)
    
    with tab4:
        st.subheader("Search Alerts")
        
        col1, col2 = st.columns(2)
        
        with col1:
            search_city = st.text_input("Search by City", placeholder="Enter city name...")
        
        with col2:
            if 'Country' in alerts_df.columns:
                search_country = st.selectbox(
                    "Filter by Country",
                    options=['All'] + sorted(alerts_df['Country'].unique().tolist())
                )
            else:
                search_country = 'All'
        
        # Apply search
        search_df = alerts_df.copy()
        
        if search_city and 'City' in search_df.columns:
            search_df = search_df[search_df['City'].str.contains(search_city, case=False, na=False)]
        
        if search_country != 'All' and 'Country' in search_df.columns:
            search_df = search_df[search_df['Country'] == search_country]
        
        st.markdown(f"**Found {len(search_df)} alerts**")
        
        if not search_df.empty:
            # Display as table
            display_cols = ['City', 'Country', 'alert_level', 'alert_type', 'event_intensity']
            display_cols = [c for c in display_cols if c in search_df.columns]
            
            st.dataframe(
                search_df[display_cols].head(100),
                use_container_width=True,
                column_config={
                    'alert_level': st.column_config.TextColumn('Level'),
                    'alert_type': st.column_config.TextColumn('Type'),
                    'event_intensity': st.column_config.ProgressColumn('Intensity', min_value=0, max_value=1)
                }
            )


if __name__ == "__main__":
    main()
