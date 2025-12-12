"""
📊 EDA Validation Dashboard Page

This page validates synthetic data quality using Spark SQL:
- Distribution analysis of generated data
- Statistical tests to verify data quality
- Comparison with historical patterns
- Anomaly detection validation
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import json

# Page config
st.set_page_config(
    page_title="EDA Validation - climaXtreme",
    page_icon="📊",
    layout="wide"
)

st.title("📊 EDA Validation - Synthetic Data Quality")
st.markdown("""
Validate synthetic data quality using **Spark SQL** distributed statistics.
Compare generated data against expected distributions and historical patterns.
""")

# Sidebar
st.sidebar.header("⚙️ Validation Settings")

data_to_validate = st.sidebar.selectbox(
    "Data to Validate",
    options=[
        "synthetic_hourly.parquet",
        "synthetic_daily.parquet",
        "synthetic_alerts.parquet",
        "streaming_output (if available)"
    ],
    index=0
)

sample_size = st.sidebar.slider(
    "Sample Size for Analysis",
    min_value=1000,
    max_value=100000,
    value=10000,
    step=1000,
    help="Number of records to sample for validation"
)

validate_btn = st.sidebar.button(
    "🔍 Run Validation",
    type="primary",
    use_container_width=True
)

# Session state
if 'validation_results' not in st.session_state:
    st.session_state.validation_results = None
if 'validation_data' not in st.session_state:
    st.session_state.validation_data = None


def run_spark_validation(data_source: str, sample_size: int):
    """Run EDA validation using Spark SQL."""
    import subprocess
    
    # Handle streaming output path
    if "streaming" in data_source:
        hdfs_path = "/data/climaxtreme/streaming/output"
    else:
        hdfs_path = f"/data/climaxtreme/synthetic/{data_source}"
    
    script = f'''
import json
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from datetime import datetime

spark = SparkSession.builder \\
    .appName("Dashboard-EDAValidation") \\
    .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \\
    .config("spark.sql.parquet.datetimeRebaseModeInRead", "CORRECTED") \\
    .getOrCreate()

# Load data
try:
    df = spark.read.parquet("hdfs://climaxtreme-namenode:9000{hdfs_path}")
except Exception as e:
    print("VALIDATION_ERROR:" + str(e))
    spark.stop()
    exit()

# Sample if needed
total_count = df.count()
if total_count > {sample_size}:
    df = df.sample(fraction={sample_size}/total_count, seed=42)

sample_count = df.count()

# Find temperature column
temp_cols = ['temperature', 'temperature_hourly', 'avg_temperature', 'AverageTemperature']
temp_col = None
for col in temp_cols:
    if col in df.columns:
        temp_col = col
        break

results = {{
    'timestamp': datetime.now().isoformat(),
    'data_source': '{data_source}',
    'total_records': total_count,
    'sample_size': sample_count,
    'columns': df.columns,
    'validations': []
}}

# Temperature validation
if temp_col:
    temp_stats = df.select(
        F.count(temp_col).alias("count"),
        F.mean(temp_col).alias("mean"),
        F.stddev(temp_col).alias("std"),
        F.min(temp_col).alias("min"),
        F.max(temp_col).alias("max"),
        F.expr(f"percentile_approx({{temp_col}}, 0.25)").alias("q25"),
        F.expr(f"percentile_approx({{temp_col}}, 0.5)").alias("median"),
        F.expr(f"percentile_approx({{temp_col}}, 0.75)").alias("q75"),
        F.skewness(temp_col).alias("skewness"),
        F.kurtosis(temp_col).alias("kurtosis")
    ).first().asDict()
    
    # Validate temperature range
    is_valid = -60 <= temp_stats['min'] <= 60 and -40 <= temp_stats['max'] <= 60
    
    results['validations'].append({{
        'test_name': 'Temperature Distribution',
        'passed': is_valid,
        'statistics': {{k: float(v) if v is not None else None for k, v in temp_stats.items()}}
    }})

# Humidity validation
if 'humidity' in df.columns or 'humidity_pct' in df.columns:
    hum_col = 'humidity' if 'humidity' in df.columns else 'humidity_pct'
    
    violations = df.filter((F.col(hum_col) < 0) | (F.col(hum_col) > 100)).count()
    violation_rate = violations / sample_count if sample_count > 0 else 0
    
    hum_stats = df.select(
        F.mean(hum_col).alias("mean"),
        F.stddev(hum_col).alias("std"),
        F.min(hum_col).alias("min"),
        F.max(hum_col).alias("max")
    ).first().asDict()
    
    results['validations'].append({{
        'test_name': 'Humidity Bounds (0-100%)',
        'passed': violation_rate < 0.01,
        'violation_rate': float(violation_rate),
        'violations': int(violations),
        'statistics': {{k: float(v) if v is not None else None for k, v in hum_stats.items()}}
    }})

# Wind speed validation
wind_cols = ['wind_speed', 'wind_speed_kmh']
wind_col = None
for col in wind_cols:
    if col in df.columns:
        wind_col = col
        break

if wind_col:
    wind_stats = df.select(
        F.mean(wind_col).alias("mean"),
        F.stddev(wind_col).alias("std"),
        F.min(wind_col).alias("min"),
        F.max(wind_col).alias("max")
    ).first().asDict()
    
    # Wind should be positive and reasonable
    is_valid = wind_stats['min'] >= 0 and wind_stats['max'] <= 400
    
    results['validations'].append({{
        'test_name': 'Wind Speed Validity',
        'passed': is_valid,
        'statistics': {{k: float(v) if v is not None else None for k, v in wind_stats.items()}}
    }})

# Alert distribution validation
alert_cols = ['alert_level', 'alert_type']
for alert_col in alert_cols:
    if alert_col in df.columns:
        dist = df.groupBy(alert_col).count().collect()
        dist_dict = {{row[alert_col]: row['count'] for row in dist}}
        
        total = sum(dist_dict.values())
        
        if alert_col == 'alert_level':
            green_ratio = dist_dict.get('green', 0) / total if total > 0 else 0
            is_valid = green_ratio > 0.5  # At least 50% should be safe
            
            results['validations'].append({{
                'test_name': 'Alert Level Distribution',
                'passed': is_valid,
                'distribution': dist_dict,
                'green_ratio': float(green_ratio)
            }})

# Climate zone distribution
if 'climate_zone' in df.columns:
    zones = df.groupBy('climate_zone').count().collect()
    zone_dict = {{row['climate_zone']: row['count'] for row in zones}}
    
    results['validations'].append({{
        'test_name': 'Climate Zone Coverage',
        'passed': len(zone_dict) >= 3,  # At least 3 zones represented
        'distribution': zone_dict,
        'n_zones': len(zone_dict)
    }})

# Calculate overall score
passed = sum(1 for v in results['validations'] if v.get('passed', False))
total_tests = len(results['validations'])
results['overall_score'] = (passed / total_tests * 100) if total_tests > 0 else 0
results['tests_passed'] = passed
results['tests_total'] = total_tests

# Get sample data for visualization
sample_data = df.limit(5000).toPandas().to_json(orient='records', date_format='iso')

print("VALIDATION_RESULTS_START")
print(json.dumps(results))
print("VALIDATION_RESULTS_END")
print("SAMPLE_DATA_START")
print(sample_data)
print("SAMPLE_DATA_END")

spark.stop()
'''
    
    cmd = ["docker", "exec", "climaxtreme-processor", "python", "-c", script]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    
    if "VALIDATION_ERROR:" in result.stdout:
        error_msg = result.stdout.split("VALIDATION_ERROR:")[1].split("\n")[0]
        raise Exception(f"Data loading error: {error_msg}")
    
    if result.returncode != 0 and "VALIDATION_RESULTS_START" not in result.stdout:
        raise Exception(f"Validation failed: {result.stderr}")
    
    # Parse results
    output = result.stdout
    
    # Extract validation results
    results_start = output.find("VALIDATION_RESULTS_START") + len("VALIDATION_RESULTS_START")
    results_end = output.find("VALIDATION_RESULTS_END")
    results_json = output[results_start:results_end].strip()
    validation_results = json.loads(results_json)
    
    # Extract sample data
    data_start = output.find("SAMPLE_DATA_START") + len("SAMPLE_DATA_START")
    data_end = output.find("SAMPLE_DATA_END")
    data_json = output[data_start:data_end].strip()
    sample_data = pd.DataFrame(json.loads(data_json))
    
    return validation_results, sample_data


# Main content
if validate_btn:
    with st.spinner(f"🔍 Running EDA validation on {data_to_validate}..."):
        try:
            results, sample_df = run_spark_validation(data_to_validate, sample_size)
            
            st.session_state.validation_results = results
            st.session_state.validation_data = sample_df
            
            st.success(f"✅ Validation complete! Score: {results['overall_score']:.1f}%")
            
        except Exception as e:
            st.error(f"❌ Validation failed: {e}")
            st.exception(e)

# Display results
if st.session_state.validation_results is not None:
    results = st.session_state.validation_results
    df = st.session_state.validation_data
    
    st.markdown("---")
    
    # Overall score
    score = results['overall_score']
    score_color = "🟢" if score >= 80 else "🟡" if score >= 60 else "🔴"
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            f"{score_color} Validation Score",
            f"{score:.1f}%"
        )
    with col2:
        st.metric(
            "✅ Tests Passed",
            f"{results['tests_passed']}/{results['tests_total']}"
        )
    with col3:
        st.metric(
            "📊 Total Records",
            f"{results['total_records']:,}"
        )
    with col4:
        st.metric(
            "🔬 Sample Analyzed",
            f"{results['sample_size']:,}"
        )
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs([
        "🧪 Test Results",
        "📈 Distribution Analysis",
        "📋 Data Sample"
    ])
    
    with tab1:
        st.subheader("Validation Test Results")
        
        for validation in results['validations']:
            test_name = validation['test_name']
            passed = validation.get('passed', False)
            
            icon = "✅" if passed else "❌"
            
            with st.expander(f"{icon} {test_name}", expanded=not passed):
                if 'statistics' in validation:
                    stats = validation['statistics']
                    
                    # Create stats table
                    stats_df = pd.DataFrame([{
                        'Metric': k.replace('_', ' ').title(),
                        'Value': f"{v:.4f}" if isinstance(v, float) else str(v)
                    } for k, v in stats.items() if v is not None])
                    
                    st.dataframe(stats_df, use_container_width=True, hide_index=True)
                
                if 'distribution' in validation:
                    dist = validation['distribution']
                    
                    fig = px.bar(
                        x=list(dist.keys()),
                        y=list(dist.values()),
                        title=f"{test_name} Distribution",
                        labels={'x': 'Category', 'y': 'Count'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                if 'violation_rate' in validation:
                    st.metric(
                        "Violation Rate",
                        f"{validation['violation_rate']*100:.2f}%",
                        delta=f"{validation['violations']} violations"
                    )
                
                if 'green_ratio' in validation:
                    st.metric(
                        "Safe (Green) Ratio",
                        f"{validation['green_ratio']*100:.1f}%"
                    )
    
    with tab2:
        st.subheader("Distribution Analysis")
        
        if df is not None and len(df) > 0:
            # Find numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Temperature distribution
                temp_cols = ['temperature', 'temperature_hourly', 'avg_temperature']
                temp_col = next((c for c in temp_cols if c in df.columns), None)
                
                if temp_col:
                    fig = px.histogram(
                        df, x=temp_col, nbins=50,
                        title="Temperature Distribution",
                        color_discrete_sequence=['#FF6B6B']
                    )
                    fig.add_vline(x=df[temp_col].mean(), line_dash="dash", 
                                  annotation_text="Mean")
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Humidity distribution
                hum_cols = ['humidity', 'humidity_pct']
                hum_col = next((c for c in hum_cols if c in df.columns), None)
                
                if hum_col:
                    fig = px.histogram(
                        df, x=hum_col, nbins=50,
                        title="Humidity Distribution",
                        color_discrete_sequence=['#4ECDC4']
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # More distributions
            col3, col4 = st.columns(2)
            
            with col3:
                wind_cols = ['wind_speed', 'wind_speed_kmh']
                wind_col = next((c for c in wind_cols if c in df.columns), None)
                
                if wind_col:
                    fig = px.histogram(
                        df, x=wind_col, nbins=50,
                        title="Wind Speed Distribution",
                        color_discrete_sequence=['#45B7D1']
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col4:
                if 'pressure' in df.columns or 'pressure_hpa' in df.columns:
                    press_col = 'pressure' if 'pressure' in df.columns else 'pressure_hpa'
                    fig = px.histogram(
                        df, x=press_col, nbins=50,
                        title="Pressure Distribution",
                        color_discrete_sequence=['#96CEB4']
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Climate zone breakdown
            if 'climate_zone' in df.columns and temp_col:
                st.markdown("### Temperature by Climate Zone")
                fig = px.box(
                    df, x='climate_zone', y=temp_col,
                    title="Temperature Distribution per Climate Zone",
                    color='climate_zone'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Data Sample")
        
        if df is not None:
            st.markdown(f"**Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns")
            st.markdown(f"**Columns:** {', '.join(df.columns[:15])}{'...' if len(df.columns) > 15 else ''}")
            
            st.dataframe(df.head(100), use_container_width=True, hide_index=True)
            
            # Download
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 Download Sample CSV",
                data=csv,
                file_name=f"eda_sample_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

else:
    st.info("""
    👆 **Click "Run Validation"** to analyze synthetic data quality.
    
    This page performs EDA validation using **Spark SQL**:
    - Distribution statistics (mean, std, skewness, kurtosis)
    - Bounds validation (humidity 0-100%, wind speed ≥ 0)
    - Alert level distribution analysis
    - Climate zone coverage verification
    """)
    
    st.markdown("""
    ### 📋 Validation Tests
    
    | Test | Description | Pass Criteria |
    |------|-------------|---------------|
    | Temperature Distribution | Validates temperature range | -60°C to 60°C |
    | Humidity Bounds | Checks humidity percentage | 0-100%, <1% violations |
    | Wind Speed Validity | Validates wind measurements | 0-400 km/h |
    | Alert Distribution | Checks alert levels | >50% green (safe) |
    | Climate Zone Coverage | Verifies geographic diversity | ≥3 zones |
    """)

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.markdown("""
**Available Datasets:**
- `synthetic_hourly.parquet`: Full hourly data
- `synthetic_daily.parquet`: Daily aggregations
- `synthetic_alerts.parquet`: Alert records
""")
