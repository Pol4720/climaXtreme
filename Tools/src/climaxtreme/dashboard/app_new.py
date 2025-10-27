"""
climaXtreme Dashboard - Main Application
Modern Streamlit dashboard using pages architecture.
"""

import streamlit as st
from pathlib import Path
import sys

# Ensure the package is importable
try:
    from climaxtreme.dashboard.utils import configure_sidebar, DataSource
except ImportError:
    _src_dir = Path(__file__).resolve().parents[2]
    if str(_src_dir) not in sys.path:
        sys.path.insert(0, str(_src_dir))
    from climaxtreme.dashboard.utils import configure_sidebar, DataSource


def run_dashboard(host: str = "localhost", port: int = 8501, data_dir: str = None) -> None:
    """
    Launch the Streamlit dashboard.
    
    Args:
        host: Host to run the dashboard on
        port: Port to run the dashboard on
        data_dir: Optional data directory (for backward compatibility)
    """
    import subprocess
    import os
    
    app_path = Path(__file__).resolve()
    
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        "--server.address", host,
        "--server.port", str(port),
        "--server.headless", "true"
    ]
    
    if data_dir:
        os.environ["CLIMAXTREME_DATA_DIR"] = data_dir
    
    print(f"🚀 Launching climaXtreme Dashboard at http://{host}:{port}")
    print(f"📊 Dashboard uses multi-page architecture with HDFS support")
    print(f"🛑 Press Ctrl+C to stop the server")
    
    subprocess.run(cmd)


def main():
    """Main dashboard home page."""
    
    # Page configuration (MUST be first Streamlit command)
    st.set_page_config(
        page_title="climaXtreme Dashboard",
        page_icon="🌡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Configure sidebar
    configure_sidebar()
    
    # Main page content
    st.title("🌡️ climaXtreme Dashboard")
    st.markdown("### Interactive Climate Data Analysis Platform")
    
    # Welcome section
    st.markdown("""
    Welcome to the **climaXtreme Dashboard** - a comprehensive platform for analyzing 
    historical global temperature data using Apache Spark processed datasets.
    
    This dashboard provides:
    - 📈 **Temporal Analysis**: Track temperature trends over time
    - 🌍 **Spatial Analysis**: Explore regional and continental patterns
    - 📊 **Statistical Analysis**: Deep dive into correlations and distributions
    - ⚡ **Extreme Events**: Identify temperature anomalies and extremes
    - 🎯 **Seasonal Patterns**: Understand seasonal variations
    """)
    
    # Initialize data source
    data_source = DataSource()
    
    # System status
    st.markdown("---")
    st.subheader("📊 System Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if data_source.use_hdfs:
            st.metric(
                "Data Source", 
                "HDFS",
                help="Reading from Hadoop Distributed File System"
            )
            st.caption(f"🐘 {data_source.hdfs_host}:{data_source.hdfs_port}")
        else:
            st.metric(
                "Data Source",
                "Local Files",
                help="Reading from local filesystem"
            )
    
    with col2:
        # Check if we can access descriptive stats
        try:
            stats_df = data_source.load_parquet('descriptive_stats.parquet')
            if stats_df is not None:
                st.metric("Status", "✅ Connected", delta="Ready")
            else:
                st.metric("Status", "⚠️ No Data", delta="Check config")
        except Exception as e:
            st.metric("Status", "❌ Error", delta="Check logs")
    
    with col3:
        from climaxtreme.dashboard.utils import get_available_parquets
        parquets = get_available_parquets()
        st.metric("Available Datasets", len(parquets))
    
    # Available datasets
    st.markdown("---")
    st.subheader("📁 Available Datasets")
    
    from climaxtreme.dashboard.utils import get_available_parquets
    parquets = get_available_parquets()
    
    # Display datasets in a nice grid
    cols = st.columns(2)
    for idx, (filename, description) in enumerate(parquets.items()):
        with cols[idx % 2]:
            with st.container():
                st.markdown(f"**{filename.replace('.parquet', '').replace('_', ' ').title()}**")
                st.caption(f"📄 {description}")
                
                # Try to load and show basic info
                try:
                    df = data_source.load_parquet(filename)
                    if df is not None:
                        st.success(f"✅ {len(df):,} rows available")
                    else:
                        st.warning("⚠️ Not found")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)[:50]}")
    
    # Navigation guide
    st.markdown("---")
    st.subheader("🧭 Navigation Guide")
    
    st.markdown("""
    Use the sidebar to navigate between different analysis pages:
    
    1. **📊 Overview** - System status and quick statistics
    2. **📈 Temporal Analysis** - Monthly and yearly trends
    3. **🌡️ Anomalies** - Temperature anomalies and climatology
    4. **🍂 Seasonal Analysis** - Seasonal patterns and variations
    5. **⚡ Extreme Events** - Extreme temperature thresholds
    6. **🗺️ Regional Analysis** - Analysis by geographic region
    7. **🌐 Continental Analysis** - Continental-level insights
    8. **📊 Statistical Analysis** - Correlations and statistical tests
    
    Each page is optimized to work with the specific parquet files generated by Spark processing.
    """)
    
    # Quick stats section
    st.markdown("---")
    st.subheader("⚡ Quick Statistics")
    
    try:
        stats_df = data_source.load_parquet('descriptive_stats.parquet')
        if stats_df is not None and not stats_df.empty:
            # Show key stats
            col1, col2, col3, col4 = st.columns(4)
            
            # Try to extract some interesting stats
            temp_stats = stats_df[stats_df['variable'] == 'avg_temperature'].iloc[0] if 'avg_temperature' in stats_df['variable'].values else None
            
            if temp_stats is not None:
                with col1:
                    st.metric("Global Mean Temp", f"{temp_stats.get('mean', 0):.2f}°C")
                with col2:
                    st.metric("Std Deviation", f"{temp_stats.get('std_dev', 0):.2f}°C")
                with col3:
                    st.metric("Min Temp", f"{temp_stats.get('min', 0):.2f}°C")
                with col4:
                    st.metric("Max Temp", f"{temp_stats.get('max', 0):.2f}°C")
            else:
                st.info("📊 Descriptive statistics available - navigate to Statistical Analysis for details")
        else:
            st.info("💡 Load a dataset to see quick statistics")
    except Exception as e:
        st.info(f"Quick stats will appear when data is loaded")
    
    # Footer
    st.markdown("---")
    st.caption("🌡️ climaXtreme Dashboard | Powered by Apache Spark + Streamlit | Data stored in HDFS")


if __name__ == "__main__":
    main()
