"""
🌀 Storm Tracking Page
Real-time storm evolution visualization with trajectory tracking.
Supports both static data and LIVE Kafka streaming.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List
import time

try:
    from climaxtreme.dashboard.utils import configure_sidebar, DataSource, show_data_info
    from climaxtreme.dashboard.components.data_checker import (
        check_synthetic_data_availability,
        UserAction,
        show_hdfs_connection_status
    )
    from climaxtreme.dashboard.components.kafka_realtime import (
        get_kafka_state,
        check_kafka_available,
        TOPICS
    )
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from climaxtreme.dashboard.utils import configure_sidebar, DataSource, show_data_info
    from climaxtreme.dashboard.components.data_checker import (
        check_synthetic_data_availability,
        UserAction,
        show_hdfs_connection_status
    )
    from climaxtreme.dashboard.components.kafka_realtime import (
        get_kafka_state,
        check_kafka_available,
        TOPICS
    )


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
        page_title="Seguimiento de Tormentas - climaXtreme",
        page_icon="🌀",
        layout="wide"
    )
    
    configure_sidebar()
    show_hdfs_connection_status()
    
    st.title("🌀 Seguimiento y Evolución de Tormentas")
    st.markdown("""
    Visualización en tiempo real de trayectorias de tormentas, evolución de intensidad y áreas afectadas.
    Sigue los eventos de tormenta sintéticos generados a partir de patrones climáticos.
    """)
    
    # Leyenda de categorías
    with st.expander("📖 Escala Saffir-Simpson de Huracanes", expanded=False):
        cols = st.columns(6)
        for i, (cat, info) in enumerate(STORM_CATEGORIES.items()):
            with cols[i]:
                st.markdown(f"""
                <div style='background-color:{info["color"]}; padding:10px; border-radius:5px; text-align:center; color:white;'>
                    <strong>{info["name"]}</strong><br>
                    {info["wind"]}
                </div>
                """, unsafe_allow_html=True)
    
    # Verificar disponibilidad de datos sintéticos
    storm_df, action = check_synthetic_data_availability(
        page_name="Seguimiento de Tormentas",
        required_dataset="synthetic_extreme_events.parquet",
        min_records=1000,
        min_cities=10
    )
    
    if action == UserAction.NONE or storm_df is None:
        st.stop()
    
    # Filtrar solo eventos de tormenta si el dataset tiene storm_id
    if 'storm_id' in storm_df.columns:
        storm_df = storm_df[storm_df['storm_id'].notna()].copy()
    
    # Normalizar nombres de columnas
    if 'lat_decimal' in storm_df.columns:
        storm_df = storm_df.rename(columns={'lat_decimal': 'latitude', 'lon_decimal': 'longitude'})
    if 'wind_speed_kmh' in storm_df.columns:
        storm_df['max_wind_kmh'] = storm_df['wind_speed_kmh']
    if 'pressure_hpa' in storm_df.columns:
        storm_df['central_pressure_hpa'] = storm_df['pressure_hpa']
    if 'storm_category' in storm_df.columns:
        storm_df = storm_df.rename(columns={'storm_category': 'category'})
    
    if storm_df.empty:
        st.warning("""
        ⚠️ **No se encontraron datos de tormentas en el dataset sintético.**
        
        El dataset cargado no contiene eventos de tormenta.
        Por favor, regenere los datos sintéticos con eventos extremos habilitados.
        """)
        st.stop()
    
    # Mostrar info de datos
    st.success(f"✅ Loaded {len(storm_df):,} storm track records")
    
    n_storms = storm_df['storm_id'].nunique()
    st.info(f"🌀 Total storms tracked: {n_storms}")
    
    # Tabs for different views - INCLUDING LIVE STREAMING
    tab_live, tab1, tab2, tab3, tab4 = st.tabs([
        "🔴 En Vivo (Kafka)",
        "🗺️ Todas las Tormentas",
        "📍 Tormenta Individual",
        "📊 Intensidad",
        "📈 Estadísticas"
    ])
    
    # TAB 0: LIVE STREAMING VIA KAFKA
    with tab_live:
        st.subheader("🔴 Tormentas en Tiempo Real (Kafka Streaming)")
        
        kafka_state = get_kafka_state()
        
        # Controles de conexión
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            if kafka_state.is_running():
                st.success("🟢 Conectado a Kafka - Recibiendo eventos de tormentas")
            else:
                st.warning("⚠️ No conectado a Kafka")
                if st.button("🔌 Conectar a Kafka", key="connect_kafka_storms"):
                    topics = [TOPICS['storms'], TOPICS['weather']]
                    if kafka_state.start(topics):
                        st.rerun()
                    else:
                        st.error("Error al conectar")
        
        with col2:
            auto_refresh_storm = st.selectbox(
                "Auto-refresh",
                options=[0, 2, 5, 10],
                format_func=lambda x: "Desactivado" if x == 0 else f"{x}s",
                key="storm_refresh"
            )
        
        with col3:
            if st.button("🔄 Actualizar", key="refresh_storms"):
                st.rerun()
        
        if kafka_state.is_running():
            # Obtener eventos de tormenta en tiempo real
            storm_events = kafka_state.get_storm_events(100)
            stats = kafka_state.get_stats()
            
            # Métricas en vivo
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("🌀 Tormentas Recibidas", stats['storms_buffered'])
            with m2:
                st.metric("📊 Total Eventos Clima", stats['total_weather'])
            with m3:
                if stats.get('last_event_time'):
                    st.metric("🕐 Último Evento", stats['last_event_time'][-8:])
            
            st.markdown("---")
            
            if storm_events:
                # Convertir a DataFrame
                live_storm_df = pd.DataFrame(storm_events)
                
                # Mapa de tormentas en vivo
                if 'latitude' in live_storm_df.columns and 'longitude' in live_storm_df.columns:
                    fig_live = go.Figure()
                    
                    for storm_id in live_storm_df['storm_id'].unique()[:5]:  # Max 5 tormentas
                        storm_data = live_storm_df[live_storm_df['storm_id'] == storm_id]
                        storm_name = storm_data['storm_name'].iloc[0] if 'storm_name' in storm_data.columns else storm_id[:8]
                        
                        fig_live.add_trace(go.Scattergeo(
                            lat=storm_data['latitude'],
                            lon=storm_data['longitude'],
                            mode='lines+markers',
                            name=storm_name,
                            marker=dict(size=8),
                            line=dict(width=2)
                        ))
                    
                    fig_live.update_layout(
                        title="🌀 Tormentas en Tiempo Real",
                        geo=dict(
                            showland=True,
                            landcolor='rgb(243, 243, 243)',
                            showocean=True,
                            oceancolor='rgb(210, 235, 255)',
                            projection_type='natural earth'
                        ),
                        height=500
                    )
                    st.plotly_chart(fig_live, use_container_width=True, key="live_storms_map")
                
                # Panel de alertas de tormenta
                st.markdown("### 🚨 Alertas de Tormenta Activas")
                
                for _, storm in live_storm_df.groupby('storm_id').last().iterrows():
                    cat = storm.get('category', 0)
                    cat_info = STORM_CATEGORIES.get(cat, STORM_CATEGORIES[0])
                    
                    st.markdown(f"""
                    <div style='background-color:{cat_info["color"]}; padding:10px; 
                                border-radius:5px; margin-bottom:5px; color:white;'>
                        <strong>🌀 {storm.get('storm_name', 'Unknown Storm')}</strong><br>
                        Categoría: {cat_info['name']} | Viento: {storm.get('max_wind_kmh', 'N/A')} km/h<br>
                        📍 Lat: {storm.get('latitude', 0):.2f}, Lon: {storm.get('longitude', 0):.2f}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("⏳ Esperando eventos de tormentas desde Kafka...")
                st.markdown("""
                **Para generar tormentas en tiempo real:**
                1. Asegúrate de que Kafka esté corriendo
                2. Inicia el productor con `include_storms=True`
                3. Los eventos aparecerán aquí automáticamente
                """)
            
            # Auto-refresh
            if auto_refresh_storm > 0:
                time.sleep(auto_refresh_storm)
                st.rerun()
        else:
            st.info("""
            👆 **Conecta a Kafka para ver tormentas en tiempo real**
            
            Mientras tanto, puedes explorar los datos estáticos en las otras pestañas.
            """)
    
    # Tab 1: All Storms Map (ORIGINAL)
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
