"""
🌊 Real-Time Streaming Forecast Dashboard Page

This page provides real-time synthetic data generation using PySpark:
- On-demand forecast generation for any number of cities
- Integration with HDFS Big Data architecture
- Visual display of streaming forecasts
- Alert monitoring and statistics
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import time

# Page config
st.set_page_config(
    page_title="Streaming Forecast - climaXtreme",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 Real-Time Streaming Forecast")
st.markdown("""
Generate **on-demand forecasts** using PySpark Structured Streaming.
Data is generated in real-time based on historical patterns stored in HDFS.
""")

# Sidebar configuration
st.sidebar.header("⚙️ Streaming Configuration")

n_cities = st.sidebar.slider(
    "Number of Cities",
    min_value=5,
    max_value=500,
    value=50,
    step=5,
    help="Number of cities to include in forecast"
)

forecast_hours = st.sidebar.slider(
    "Forecast Horizon (hours)",
    min_value=6,
    max_value=72,
    value=24,
    step=6,
    help="How many hours ahead to forecast"
)

# Data source selection
data_source = st.sidebar.selectbox(
    "Historical Data Source",
    options=[
        "monthly.parquet",
        "yearly.parquet",
        "climatology.parquet"
    ],
    index=0,
    help="Source data for learning patterns (processed historical data)"
)

st.sidebar.markdown("---")

# Generation button
generate_btn = st.sidebar.button(
    "🚀 Generate Forecast",
    type="primary",
    use_container_width=True
)

# Initialize session state
if 'forecast_df' not in st.session_state:
    st.session_state.forecast_df = None
if 'last_generation_time' not in st.session_state:
    st.session_state.last_generation_time = None
if 'generation_stats' not in st.session_state:
    st.session_state.generation_stats = {}


def generate_forecast_spark(n_cities: int, forecast_hours: int, data_source: str):
    """Generate forecast using PySpark in the processor container."""
    import subprocess
    import json
    
    # Build the Python script to run in Docker
    script = f'''
import json
from pyspark.sql import SparkSession
from climaxtreme.streaming import SparkStreamingGenerator, StreamingConfig

spark = SparkSession.builder \\
    .appName("Dashboard-StreamingForecast") \\
    .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \\
    .config("spark.sql.parquet.datetimeRebaseModeInRead", "CORRECTED") \\
    .getOrCreate()

config = StreamingConfig(
    hdfs_base="hdfs://climaxtreme-namenode:9000",
    historical_data_path="/data/climaxtreme/processed/{data_source}",
    forecast_horizon_hours={forecast_hours}
)

generator = SparkStreamingGenerator(spark, config)
forecast_df = generator.generate_streaming_forecast(n_cities={n_cities}, forecast_hours={forecast_hours})

# Convert to JSON for transfer
result = forecast_df.toPandas().to_json(orient="records", date_format="iso")
print("FORECAST_DATA_START")
print(result)
print("FORECAST_DATA_END")

spark.stop()
'''
    
    # Run in Docker container
    cmd = [
        "docker", "exec", "climaxtreme-processor",
        "python", "-c", script
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    
    if result.returncode != 0:
        raise Exception(f"Spark generation failed: {result.stderr}")
    
    # Parse output
    output = result.stdout
    start_marker = "FORECAST_DATA_START"
    end_marker = "FORECAST_DATA_END"
    
    start_idx = output.find(start_marker) + len(start_marker)
    end_idx = output.find(end_marker)
    
    json_data = output[start_idx:end_idx].strip()
    records = json.loads(json_data)
    
    return pd.DataFrame(records)


# Main content
if generate_btn:
    start_time = time.time()
    
    with st.spinner(f"🔄 Generating {forecast_hours}h forecast for {n_cities} cities using PySpark..."):
        try:
            forecast_df = generate_forecast_spark(n_cities, forecast_hours, data_source)
            
            elapsed = time.time() - start_time
            
            st.session_state.forecast_df = forecast_df
            st.session_state.last_generation_time = datetime.now()
            st.session_state.generation_stats = {
                'n_records': len(forecast_df),
                'n_cities': forecast_df['city'].nunique() if 'city' in forecast_df.columns else 0,
                'elapsed_seconds': elapsed,
                'data_source': data_source
            }
            
            st.success(f"✅ Generated {len(forecast_df):,} forecast records in {elapsed:.2f}s")
            
        except Exception as e:
            st.error(f"❌ Generation failed: {e}")
            st.exception(e)

# Display results
if st.session_state.forecast_df is not None:
    df = st.session_state.forecast_df
    stats = st.session_state.generation_stats
    
    st.markdown("---")
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Total Records", f"{stats.get('n_records', 0):,}")
    with col2:
        st.metric("🏙️ Cities", f"{stats.get('n_cities', 0):,}")
    with col3:
        st.metric("⏱️ Generation Time", f"{stats.get('elapsed_seconds', 0):.2f}s")
    with col4:
        if st.session_state.last_generation_time:
            st.metric("🕐 Last Update", 
                     st.session_state.last_generation_time.strftime("%H:%M:%S"))
    
    st.markdown("---")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Temperature Forecast",
        "🌧️ Weather Conditions", 
        "⚠️ Alert Analysis",
        "📋 Raw Data"
    ])
    
    with tab1:
        st.subheader("Temperature Forecast by City")
        
        # Convert timestamp if needed
        if 'forecast_timestamp' in df.columns:
            df['forecast_timestamp'] = pd.to_datetime(df['forecast_timestamp'])
        
        # Select cities to display
        cities = df['city'].unique()[:10] if 'city' in df.columns else []
        
        if len(cities) > 0:
            fig = px.line(
                df[df['city'].isin(cities)],
                x='hour_offset' if 'hour_offset' in df.columns else df.index,
                y='temperature',
                color='city',
                title="Temperature Forecast (Next Hours)",
                labels={'hour_offset': 'Hours Ahead', 'temperature': 'Temperature (°C)'}
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Temperature distribution
            col1, col2 = st.columns(2)
            
            with col1:
                fig_hist = px.histogram(
                    df, x='temperature', nbins=50,
                    title="Temperature Distribution",
                    color_discrete_sequence=['#FF6B6B']
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                if 'climate_zone' in df.columns:
                    fig_box = px.box(
                        df, x='climate_zone', y='temperature',
                        title="Temperature by Climate Zone",
                        color='climate_zone'
                    )
                    st.plotly_chart(fig_box, use_container_width=True)
    
    with tab2:
        st.subheader("Weather Conditions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Humidity distribution
            if 'humidity' in df.columns:
                fig_hum = px.histogram(
                    df, x='humidity', nbins=40,
                    title="Humidity Distribution (%)",
                    color_discrete_sequence=['#4ECDC4']
                )
                st.plotly_chart(fig_hum, use_container_width=True)
            
            # Wind speed
            if 'wind_speed' in df.columns:
                fig_wind = px.histogram(
                    df, x='wind_speed', nbins=40,
                    title="Wind Speed Distribution (km/h)",
                    color_discrete_sequence=['#45B7D1']
                )
                st.plotly_chart(fig_wind, use_container_width=True)
        
        with col2:
            # Rain state distribution
            if 'rain_state' in df.columns:
                rain_counts = df['rain_state'].value_counts()
                fig_rain = px.pie(
                    values=rain_counts.values,
                    names=rain_counts.index,
                    title="Rain State Distribution",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                st.plotly_chart(fig_rain, use_container_width=True)
            
            # Pressure distribution
            if 'pressure' in df.columns:
                fig_press = px.histogram(
                    df, x='pressure', nbins=40,
                    title="Atmospheric Pressure (hPa)",
                    color_discrete_sequence=['#96CEB4']
                )
                st.plotly_chart(fig_press, use_container_width=True)
    
    with tab3:
        st.subheader("Alert Analysis")
        
        if 'alert_level' in df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                # Alert level distribution
                alert_counts = df['alert_level'].value_counts()
                colors = {
                    'green': '#2ecc71',
                    'yellow': '#f1c40f', 
                    'orange': '#e67e22',
                    'red': '#e74c3c'
                }
                
                fig_alert = px.pie(
                    values=alert_counts.values,
                    names=alert_counts.index,
                    title="Alert Level Distribution",
                    color=alert_counts.index,
                    color_discrete_map=colors
                )
                st.plotly_chart(fig_alert, use_container_width=True)
            
            with col2:
                # Alert type breakdown
                if 'alert_type' in df.columns:
                    alert_type_counts = df[df['alert_level'] != 'green']['alert_type'].value_counts()
                    if len(alert_type_counts) > 0:
                        fig_type = px.bar(
                            x=alert_type_counts.index,
                            y=alert_type_counts.values,
                            title="Active Alert Types",
                            labels={'x': 'Alert Type', 'y': 'Count'},
                            color=alert_type_counts.index,
                            color_discrete_sequence=px.colors.qualitative.Bold
                        )
                        st.plotly_chart(fig_type, use_container_width=True)
                    else:
                        st.info("✅ No active alerts in current forecast")
            
            # Cities with alerts
            st.markdown("### 🏙️ Cities with Active Alerts")
            alert_cities = df[df['alert_level'].isin(['yellow', 'orange', 'red'])][
                ['city', 'country', 'temperature', 'wind_speed', 'alert_level', 'alert_type']
            ].drop_duplicates(subset=['city'])
            
            if len(alert_cities) > 0:
                st.dataframe(
                    alert_cities.sort_values('alert_level', ascending=False),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.success("✅ All cities have green (safe) status")
    
    with tab4:
        st.subheader("Raw Forecast Data")
        
        # Data summary
        st.markdown(f"**Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns")
        
        # Column selector
        all_cols = df.columns.tolist()
        selected_cols = st.multiselect(
            "Select columns to display",
            options=all_cols,
            default=all_cols[:8]
        )
        
        if selected_cols:
            st.dataframe(
                df[selected_cols].head(100),
                use_container_width=True,
                hide_index=True
            )
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Forecast CSV",
            data=csv,
            file_name=f"streaming_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

else:
    # No data yet - show instructions
    st.info("""
    👆 **Click "Generate Forecast" in the sidebar** to create real-time synthetic data.
    
    This page uses **PySpark Structured Streaming** to generate forecasts:
    1. Loads historical data from HDFS
    2. Learns statistical patterns per city
    3. Generates hourly forecasts with weather variables
    4. Produces alert levels based on extreme conditions
    """)
    
    # Show architecture diagram
    st.markdown("""
    ### 🏗️ Architecture
    ```
    HDFS Historical Data → Spark Session → Statistical Learning
                                              ↓
    Dashboard Request → SparkStreamingGenerator → Forecast DataFrame
                                              ↓
                           Weather Variables + Alerts → Visualization
    ```
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
**Data Sources:**
- `monthly.parquet`: Monthly temperature statistics
- `yearly.parquet`: Yearly temperature averages
- `climatology.parquet`: Multi-year climate patterns
""")
