"""
🌀 Storm Tracking Page
Real-time storm evolution visualization with trajectory tracking.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List

try:
    from climaxtreme.dashboard.utils import configure_sidebar, DataSource, show_data_info
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from climaxtreme.dashboard.utils import configure_sidebar, DataSource, show_data_info


# Saffir-Simpson Hurricane Scale
STORM_CATEGORIES = {
    0: {'name': 'Tropical Depression', 'color': '#5DADE2', 'wind': '<63 km/h'},
    1: {'name': 'Category 1', 'color': '#F7DC6F', 'wind': '63-118 km/h'},
    2: {'name': 'Category 2', 'color': '#F5B041', 'wind': '119-153 km/h'},
    3: {'name': 'Category 3', 'color': '#E74C3C', 'wind': '154-177 km/h'},
    4: {'name': 'Category 4', 'color': '#8E44AD', 'wind': '178-208 km/h'},
    5: {'name': 'Category 5', 'color': '#1B2631', 'wind': '>208 km/h'}
}


def load_storm_data(data_source: DataSource) -> Optional[pd.DataFrame]:
    """Load storm tracking data."""
    try:
        df = data_source.load_parquet('storm_tracks.parquet')
        if df is None:
            df = data_source.load_parquet('synthetic/storm_tracks.parquet')
        return df
    except Exception as e:
        st.error(f"Error loading storm data: {e}")
        return None


def load_synthetic_data(data_source: DataSource) -> Optional[pd.DataFrame]:
    """Load synthetic hourly data for storm events."""
    try:
        df = data_source.load_parquet('synthetic_hourly.parquet')
        if df is None:
            df = data_source.load_parquet('synthetic/synthetic_hourly.parquet')
        return df
    except Exception as e:
        return None


def create_storm_trajectory_map(storm_df: pd.DataFrame, storm_id: str) -> go.Figure:
    """
    Create an animated map showing storm trajectory.
    
    Args:
        storm_df: Storm tracking DataFrame
        storm_id: ID of the storm to visualize
    """
    storm_data = storm_df[storm_df['storm_id'] == storm_id].copy()
    
    if storm_data.empty:
        return go.Figure().add_annotation(text="No data for selected storm", showarrow=False)
    
    # Sort by timestamp
    storm_data = storm_data.sort_values('timestamp')
    
    # Get storm name
    storm_name = storm_data['storm_name'].iloc[0] if 'storm_name' in storm_data.columns else storm_id
    
    fig = go.Figure()
    
    # Add trajectory line
    fig.add_trace(go.Scattergeo(
        lat=storm_data['latitude'],
        lon=storm_data['longitude'],
        mode='lines',
        line=dict(width=3, color='#E74C3C'),
        name='Trajectory',
        hoverinfo='skip'
    ))
    
    # Add points colored by category
    for cat in range(6):
        cat_data = storm_data[storm_data['category'] == cat]
        if not cat_data.empty:
            fig.add_trace(go.Scattergeo(
                lat=cat_data['latitude'],
                lon=cat_data['longitude'],
                mode='markers',
                marker=dict(
                    size=10 + cat * 3,
                    color=STORM_CATEGORIES[cat]['color'],
                    line=dict(width=1, color='white')
                ),
                name=STORM_CATEGORIES[cat]['name'],
                hovertemplate=(
                    f"<b>{storm_name}</b><br>" +
                    "Category: %{text}<br>" +
                    "Lat: %{lat:.2f}<br>" +
                    "Lon: %{lon:.2f}<br>" +
                    "<extra></extra>"
                ),
                text=[STORM_CATEGORIES[cat]['name']] * len(cat_data)
            ))
    
    # Add start and end markers
    fig.add_trace(go.Scattergeo(
        lat=[storm_data['latitude'].iloc[0]],
        lon=[storm_data['longitude'].iloc[0]],
        mode='markers+text',
        marker=dict(size=15, color='green', symbol='star'),
        text=['START'],
        textposition='top center',
        name='Start',
        showlegend=True
    ))
    
    fig.add_trace(go.Scattergeo(
        lat=[storm_data['latitude'].iloc[-1]],
        lon=[storm_data['longitude'].iloc[-1]],
        mode='markers+text',
        marker=dict(size=15, color='red', symbol='x'),
        text=['END'],
        textposition='top center',
        name='End',
        showlegend=True
    ))
    
    # Calculate center
    center_lat = storm_data['latitude'].mean()
    center_lon = storm_data['longitude'].mean()
    
    fig.update_layout(
        title=f"🌀 Storm {storm_name} Trajectory",
        geo=dict(
            showland=True,
            landcolor='rgb(243, 243, 243)',
            countrycolor='rgb(204, 204, 204)',
            showocean=True,
            oceancolor='rgb(210, 235, 255)',
            showlakes=True,
            lakecolor='rgb(180, 215, 255)',
            showcountries=True,
            coastlinecolor='rgb(100, 100, 100)',
            projection_type='natural earth',
            center=dict(lat=center_lat, lon=center_lon),
            projection_scale=2
        ),
        height=600,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    return fig


def create_animated_storm_map(storm_df: pd.DataFrame, storm_id: str) -> go.Figure:
    """Create an animated visualization of storm progression."""
    storm_data = storm_df[storm_df['storm_id'] == storm_id].copy()
    
    if storm_data.empty:
        return go.Figure().add_annotation(text="No data", showarrow=False)
    
    storm_data = storm_data.sort_values('timestamp')
    storm_data['frame'] = range(len(storm_data))
    
    storm_name = storm_data['storm_name'].iloc[0] if 'storm_name' in storm_data.columns else storm_id
    
    # Create color mapping
    storm_data['color'] = storm_data['category'].apply(lambda x: STORM_CATEGORIES.get(x, STORM_CATEGORIES[0])['color'])
    
    fig = px.scatter_geo(
        storm_data,
        lat='latitude',
        lon='longitude',
        color='category',
        size='max_wind_kmh' if 'max_wind_kmh' in storm_data.columns else None,
        animation_frame='frame',
        hover_data=['category', 'max_wind_kmh', 'central_pressure_hpa'] if 'max_wind_kmh' in storm_data.columns else ['category'],
        title=f"🌀 Storm {storm_name} Evolution",
        projection='natural earth',
        color_continuous_scale='YlOrRd'
    )
    
    fig.update_layout(
        geo=dict(
            showland=True,
            landcolor='rgb(243, 243, 243)',
            showocean=True,
            oceancolor='rgb(210, 235, 255)'
        ),
        height=600
    )
    
    return fig


def create_storm_intensity_chart(storm_df: pd.DataFrame, storm_id: str) -> go.Figure:
    """Create a chart showing storm intensity over time."""
    storm_data = storm_df[storm_df['storm_id'] == storm_id].copy()
    
    if storm_data.empty:
        return go.Figure().add_annotation(text="No data", showarrow=False)
    
    storm_data = storm_data.sort_values('timestamp')
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Wind Speed', 'Central Pressure'),
        shared_xaxes=True,
        vertical_spacing=0.1
    )
    
    # Wind speed
    if 'max_wind_kmh' in storm_data.columns:
        fig.add_trace(
            go.Scatter(
                x=storm_data['timestamp'],
                y=storm_data['max_wind_kmh'],
                mode='lines+markers',
                name='Max Wind (km/h)',
                line=dict(color='#E74C3C', width=2),
                marker=dict(size=6)
            ),
            row=1, col=1
        )
        
        # Add category thresholds
        thresholds = [63, 118, 154, 178, 209]
        colors = ['#F7DC6F', '#F5B041', '#E74C3C', '#8E44AD', '#1B2631']
        for thresh, color in zip(thresholds, colors):
            fig.add_hline(y=thresh, line_dash="dash", line_color=color, 
                         opacity=0.5, row=1, col=1)
    
    # Pressure
    if 'central_pressure_hpa' in storm_data.columns:
        fig.add_trace(
            go.Scatter(
                x=storm_data['timestamp'],
                y=storm_data['central_pressure_hpa'],
                mode='lines+markers',
                name='Pressure (hPa)',
                line=dict(color='#3498DB', width=2),
                marker=dict(size=6)
            ),
            row=2, col=1
        )
    
    fig.update_layout(
        height=500,
        showlegend=True,
        title_text="Storm Intensity Metrics"
    )
    
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="Wind Speed (km/h)", row=1, col=1)
    fig.update_yaxes(title_text="Pressure (hPa)", row=2, col=1)
    
    return fig


def create_all_storms_map(storm_df: pd.DataFrame) -> go.Figure:
    """Create a map showing all storm trajectories."""
    fig = go.Figure()
    
    # Get unique storms
    storms = storm_df['storm_id'].unique()
    colors = px.colors.qualitative.Set3
    
    for i, storm_id in enumerate(storms[:15]):  # Limit to 15 storms for readability
        storm_data = storm_df[storm_df['storm_id'] == storm_id].sort_values('timestamp')
        storm_name = storm_data['storm_name'].iloc[0] if 'storm_name' in storm_data.columns else storm_id[:8]
        
        color = colors[i % len(colors)]
        
        fig.add_trace(go.Scattergeo(
            lat=storm_data['latitude'],
            lon=storm_data['longitude'],
            mode='lines+markers',
            line=dict(width=2, color=color),
            marker=dict(size=4, color=color),
            name=storm_name,
            hovertemplate=f"<b>{storm_name}</b><br>Lat: %{{lat:.2f}}<br>Lon: %{{lon:.2f}}<extra></extra>"
        ))
    
    fig.update_layout(
        title="🗺️ All Storm Trajectories",
        geo=dict(
            showland=True,
            landcolor='rgb(243, 243, 243)',
            showocean=True,
            oceancolor='rgb(210, 235, 255)',
            showcountries=True,
            projection_type='natural earth'
        ),
        height=600,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    return fig


def main():
    st.set_page_config(
        page_title="Storm Tracking - climaXtreme",
        page_icon="🌀",
        layout="wide"
    )
    
    configure_sidebar()
    
    st.title("🌀 Storm Tracking & Evolution")
    st.markdown("""
    Real-time visualization of storm trajectories, intensity evolution, and affected areas.
    Track synthetic storm events generated from climate patterns.
    """)
    
    # Legend for categories
    with st.expander("📖 Saffir-Simpson Hurricane Scale", expanded=False):
        cols = st.columns(6)
        for i, (cat, info) in enumerate(STORM_CATEGORIES.items()):
            with cols[i]:
                st.markdown(f"""
                <div style='background-color:{info["color"]}; padding:10px; border-radius:5px; text-align:center; color:white;'>
                    <strong>{info["name"]}</strong><br>
                    {info["wind"]}
                </div>
                """, unsafe_allow_html=True)
    
    # Load data
    data_source = DataSource()
    
    with st.spinner("Loading storm tracking data..."):
        storm_df = load_storm_data(data_source)
        synthetic_df = load_synthetic_data(data_source)
    
    # Check for storm data in synthetic if dedicated storm file not found
    if storm_df is None and synthetic_df is not None:
        if 'storm_id' in synthetic_df.columns:
            storm_df = synthetic_df[synthetic_df['storm_id'].notna()].copy()
            if 'lat_decimal' in storm_df.columns:
                storm_df = storm_df.rename(columns={'lat_decimal': 'latitude', 'lon_decimal': 'longitude'})
            if 'wind_speed_kmh' in storm_df.columns:
                storm_df['max_wind_kmh'] = storm_df['wind_speed_kmh']
            if 'pressure_hpa' in storm_df.columns:
                storm_df['central_pressure_hpa'] = storm_df['pressure_hpa']
            if 'storm_category' in storm_df.columns:
                storm_df = storm_df.rename(columns={'storm_category': 'category'})
    
    if storm_df is None or storm_df.empty:
        st.warning("""
        ⚠️ **No storm tracking data found!**
        
        Please generate synthetic data first:
        ```bash
        climaxtreme generate-synthetic --input-path DATA/GlobalLandTemperaturesByCity.csv --output-path DATA/synthetic
        ```
        """)
        
        # Demo mode
        st.markdown("---")
        st.subheader("📊 Demo Mode")
        
        # Generate demo storm
        np.random.seed(42)
        n_points = 50
        
        # Simulate storm trajectory (moving northwest from Caribbean)
        base_lat = 15 + np.cumsum(np.random.normal(0.3, 0.2, n_points))
        base_lon = -60 + np.cumsum(np.random.normal(-0.5, 0.3, n_points))
        
        demo_storm = pd.DataFrame({
            'storm_id': ['DEMO-STORM-001'] * n_points,
            'storm_name': ['Demo Hurricane'] * n_points,
            'timestamp': pd.date_range('2024-09-01', periods=n_points, freq='6h'),
            'latitude': base_lat,
            'longitude': base_lon,
            'category': np.clip(np.random.randint(0, 6, n_points), 0, 5),
            'max_wind_kmh': 50 + np.cumsum(np.random.normal(3, 2, n_points)),
            'central_pressure_hpa': 1010 - np.cumsum(np.random.normal(0.5, 0.3, n_points))
        })
        
        fig = create_storm_trajectory_map(demo_storm, 'DEMO-STORM-001')
        st.plotly_chart(fig, use_container_width=True)
        return
    
    # Show data info
    st.success(f"✅ Loaded {len(storm_df):,} storm track records")
    
    n_storms = storm_df['storm_id'].nunique()
    st.info(f"🌀 Total storms tracked: {n_storms}")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "🗺️ All Storms", 
        "🎯 Single Storm Tracker", 
        "📈 Intensity Analysis",
        "📊 Statistics"
    ])
    
    with tab1:
        st.subheader("All Storm Trajectories")
        fig_all = create_all_storms_map(storm_df)
        st.plotly_chart(fig_all, use_container_width=True)
    
    with tab2:
        st.subheader("Individual Storm Tracker")
        
        # Storm selection
        storms = storm_df['storm_id'].unique()
        storm_names = {}
        for sid in storms:
            name = storm_df[storm_df['storm_id'] == sid]['storm_name'].iloc[0] if 'storm_name' in storm_df.columns else sid[:8]
            storm_names[sid] = name
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            selected_storm = st.selectbox(
                "Select Storm",
                options=list(storm_names.keys()),
                format_func=lambda x: f"{storm_names[x]} ({x[:8]}...)",
                help="Choose a storm to track"
            )
        
        with col2:
            view_type = st.radio("View", ["Static", "Animated"], horizontal=True)
        
        if selected_storm:
            if view_type == "Static":
                fig = create_storm_trajectory_map(storm_df, selected_storm)
            else:
                with st.spinner("Creating animation..."):
                    fig = create_animated_storm_map(storm_df, selected_storm)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Storm details
            storm_data = storm_df[storm_df['storm_id'] == selected_storm]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                max_cat = storm_data['category'].max() if 'category' in storm_data.columns else 0
                st.metric("Max Category", f"Cat {max_cat}")
            
            with col2:
                if 'max_wind_kmh' in storm_data.columns:
                    max_wind = storm_data['max_wind_kmh'].max()
                    st.metric("Max Wind", f"{max_wind:.0f} km/h")
            
            with col3:
                if 'central_pressure_hpa' in storm_data.columns:
                    min_pressure = storm_data['central_pressure_hpa'].min()
                    st.metric("Min Pressure", f"{min_pressure:.0f} hPa")
            
            with col4:
                duration = len(storm_data)
                st.metric("Track Points", duration)
    
    with tab3:
        st.subheader("Storm Intensity Analysis")
        
        selected_storm_int = st.selectbox(
            "Select Storm for Analysis",
            options=list(storm_names.keys()),
            format_func=lambda x: f"{storm_names[x]} ({x[:8]}...)",
            key="intensity_storm"
        )
        
        if selected_storm_int:
            fig_intensity = create_storm_intensity_chart(storm_df, selected_storm_int)
            st.plotly_chart(fig_intensity, use_container_width=True)
    
    with tab4:
        st.subheader("Storm Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Category distribution
            if 'category' in storm_df.columns:
                cat_counts = storm_df.groupby('category').size().reset_index(name='count')
                cat_counts['category_name'] = cat_counts['category'].apply(
                    lambda x: STORM_CATEGORIES.get(x, STORM_CATEGORIES[0])['name']
                )
                
                fig_cat = px.pie(
                    cat_counts,
                    values='count',
                    names='category_name',
                    title='Distribution by Category',
                    color='category',
                    color_discrete_map={i: info['color'] for i, info in STORM_CATEGORIES.items()}
                )
                st.plotly_chart(fig_cat, use_container_width=True)
        
        with col2:
            # Wind speed distribution
            if 'max_wind_kmh' in storm_df.columns:
                fig_wind = px.histogram(
                    storm_df,
                    x='max_wind_kmh',
                    nbins=30,
                    title='Wind Speed Distribution',
                    color_discrete_sequence=['#E74C3C']
                )
                fig_wind.update_layout(xaxis_title='Max Wind (km/h)', yaxis_title='Count')
                st.plotly_chart(fig_wind, use_container_width=True)
        
        # Summary table
        st.markdown("### Storm Summary Table")
        
        summary = storm_df.groupby('storm_id').agg({
            'storm_name': 'first' if 'storm_name' in storm_df.columns else lambda x: 'Unknown',
            'category': 'max' if 'category' in storm_df.columns else lambda x: 0,
            'max_wind_kmh': ['max', 'mean'] if 'max_wind_kmh' in storm_df.columns else lambda x: (0, 0),
            'latitude': ['min', 'max'],
            'longitude': ['min', 'max']
        }).reset_index()
        
        summary.columns = ['Storm ID', 'Name', 'Max Category', 'Peak Wind', 'Avg Wind', 
                          'Min Lat', 'Max Lat', 'Min Lon', 'Max Lon']
        
        st.dataframe(summary, use_container_width=True)


if __name__ == "__main__":
    main()
