"""
🔮 Intensity Prediction Page
ML-based weather event intensity prediction and forecasting.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict
from pathlib import Path
import joblib

try:
    from climaxtreme.dashboard.utils import configure_sidebar, DataSource, show_data_info
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from climaxtreme.dashboard.utils import configure_sidebar, DataSource, show_data_info


def load_model(model_dir: str = "models/intensity_model") -> Optional[Dict]:
    """Load trained intensity prediction model."""
    try:
        model_path = Path(model_dir)
        
        if not model_path.exists():
            return None
        
        model = joblib.load(model_path / "intensity_rf_model.joblib")
        scaler = joblib.load(model_path / "scaler.joblib")
        metadata = joblib.load(model_path / "metadata.joblib")
        
        return {
            'model': model,
            'scaler': scaler,
            'features': metadata['features'],
            'metrics': {
                'rmse': metadata['rmse'],
                'r2': metadata['r2']
            },
            'feature_importance': metadata['feature_importance']
        }
    except Exception as e:
        return None


def predict_intensity(model_data: Dict, features_df: pd.DataFrame) -> np.ndarray:
    """Make intensity predictions using loaded model."""
    model = model_data['model']
    scaler = model_data['scaler']
    feature_cols = model_data['features']
    
    # Ensure all required features are present
    available_features = [f for f in feature_cols if f in features_df.columns]
    
    if len(available_features) < len(feature_cols):
        missing = set(feature_cols) - set(available_features)
        st.warning(f"Missing features: {missing}. Predictions may be less accurate.")
    
    X = features_df[available_features].fillna(0)
    X_scaled = scaler.transform(X)
    
    predictions = model.predict(X_scaled)
    return np.clip(predictions, 0, 1)


def create_simple_prediction_model():
    """Create a simple rule-based prediction for demo purposes."""
    def predict(df):
        # Simple heuristic model
        intensity = np.zeros(len(df))
        
        if 'anomaly_score' in df.columns:
            intensity += np.abs(df['anomaly_score'].fillna(0)) * 0.3
        
        if 'wind_speed_kmh' in df.columns:
            intensity += (df['wind_speed_kmh'].fillna(0) / 200) * 0.3
        
        if 'rain_mm' in df.columns:
            intensity += (df['rain_mm'].fillna(0) / 100) * 0.2
        
        if 'temperature_hourly' in df.columns:
            temp_deviation = np.abs(df['temperature_hourly'].fillna(20) - 20)
            intensity += (temp_deviation / 40) * 0.2
        
        return np.clip(intensity, 0, 1)
    
    return predict


def create_prediction_chart(df: pd.DataFrame, actual_col: str, pred_col: str) -> go.Figure:
    """Create chart comparing actual vs predicted intensity."""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Actual vs Predicted Intensity', 'Prediction Error'),
        shared_xaxes=True,
        vertical_spacing=0.15,
        row_heights=[0.7, 0.3]
    )
    
    # Actual vs predicted
    fig.add_trace(
        go.Scatter(
            y=df[actual_col],
            mode='lines',
            name='Actual',
            line=dict(color='#3498DB', width=1)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            y=df[pred_col],
            mode='lines',
            name='Predicted',
            line=dict(color='#E74C3C', width=1, dash='dot')
        ),
        row=1, col=1
    )
    
    # Error
    error = df[actual_col] - df[pred_col]
    fig.add_trace(
        go.Scatter(
            y=error,
            mode='lines',
            name='Error',
            line=dict(color='#9B59B6', width=1),
            fill='tozeroy',
            fillcolor='rgba(155, 89, 182, 0.2)'
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=500,
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    fig.update_yaxes(title_text="Intensity", row=1, col=1)
    fig.update_yaxes(title_text="Error", row=2, col=1)
    
    return fig


def create_feature_importance_chart(importance: Dict[str, float]) -> go.Figure:
    """Create feature importance visualization."""
    df = pd.DataFrame([
        {'Feature': k.replace('_', ' ').title(), 'Importance': v}
        for k, v in sorted(importance.items(), key=lambda x: x[1], reverse=True)
    ])
    
    fig = px.bar(
        df,
        x='Importance',
        y='Feature',
        orientation='h',
        title='🎯 Feature Importance',
        color='Importance',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        height=400,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig


def create_intensity_distribution(df: pd.DataFrame, pred_col: str) -> go.Figure:
    """Create intensity distribution visualization."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Intensity Distribution', 'Intensity by Event Type')
    )
    
    # Histogram
    fig.add_trace(
        go.Histogram(
            x=df[pred_col],
            nbinsx=30,
            marker_color='#3498DB',
            name='Distribution'
        ),
        row=1, col=1
    )
    
    # Box plot by event type if available
    if 'event_type' in df.columns:
        event_types = df['event_type'].unique()
        colors = px.colors.qualitative.Set2
        
        for i, event in enumerate(event_types):
            event_data = df[df['event_type'] == event][pred_col]
            fig.add_trace(
                go.Box(
                    y=event_data,
                    name=event,
                    marker_color=colors[i % len(colors)]
                ),
                row=1, col=2
            )
    
    fig.update_layout(height=400, showlegend=False)
    fig.update_xaxes(title_text='Intensity', row=1, col=1)
    fig.update_yaxes(title_text='Count', row=1, col=1)
    fig.update_yaxes(title_text='Intensity', row=1, col=2)
    
    return fig


def main():
    st.set_page_config(
        page_title="Intensity Prediction - climaXtreme",
        page_icon="🔮",
        layout="wide"
    )
    
    configure_sidebar()
    
    st.title("🔮 Event Intensity Prediction")
    st.markdown("""
    Machine learning-based prediction of weather event intensity.
    Uses ensemble models trained on synthetic climate data.
    """)
    
    # Model info expander
    with st.expander("📖 Model Information", expanded=False):
        st.markdown("""
        ### Model Architecture
        
        **Algorithm**: Random Forest Regressor (ensemble of decision trees)
        
        **Features Used**:
        - Temporal: hour, month, day of week
        - Geographic: latitude, longitude
        - Meteorological: temperature, humidity, pressure, wind speed, rain
        - Derived: anomaly score (z-score vs climatology)
        
        **Target**: Event intensity (normalized 0-1)
        
        **Training**: Trained on synthetic climate events with cross-validation
        
        ### Intensity Scale
        - **0.0 - 0.2**: Minor / Normal conditions
        - **0.2 - 0.4**: Moderate event
        - **0.4 - 0.6**: Significant event
        - **0.6 - 0.8**: Severe event
        - **0.8 - 1.0**: Extreme / Emergency level
        """)
    
    # Load model
    model_data = load_model()
    
    if model_data is None:
        st.info("""
        ℹ️ **No trained model found.** 
        
        Using simple heuristic prediction model for demonstration.
        
        To train a proper model, run:
        ```bash
        climaxtreme train-intensity-model --data-path DATA/synthetic/synthetic_hourly.parquet
        ```
        """)
        use_simple_model = True
        simple_predictor = create_simple_prediction_model()
    else:
        st.success(f"✅ Model loaded! R² = {model_data['metrics']['r2']:.4f}, RMSE = {model_data['metrics']['rmse']:.4f}")
        use_simple_model = False
    
    # Load data
    data_source = DataSource()
    
    with st.spinner("Loading synthetic data..."):
        try:
            df = data_source.load_parquet('synthetic_hourly.parquet')
            if df is None:
                df = data_source.load_parquet('synthetic/synthetic_hourly.parquet')
        except:
            df = None
    
    if df is None or df.empty:
        st.warning("No synthetic data available. Generating demo data...")
        
        # Generate demo data
        np.random.seed(42)
        n = 1000
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n, freq='h'),
            'hour': np.random.randint(0, 24, n),
            'month': np.random.randint(1, 13, n),
            'lat_decimal': np.random.uniform(-60, 70, n),
            'lon_decimal': np.random.uniform(-180, 180, n),
            'temperature_hourly': np.random.normal(20, 10, n),
            'humidity_pct': np.random.uniform(30, 90, n),
            'pressure_hpa': np.random.normal(1013, 15, n),
            'wind_speed_kmh': np.abs(np.random.normal(20, 15, n)),
            'rain_mm': np.abs(np.random.exponential(5, n)),
            'anomaly_score': np.random.normal(0, 1.5, n),
            'event_type': np.random.choice(['NORMAL', 'STORM', 'HEATWAVE', 'COLDSNAP'], n, p=[0.7, 0.15, 0.1, 0.05]),
            'event_intensity': np.clip(np.random.beta(2, 5, n), 0, 1)
        })
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Predictions", 
        "🎯 Feature Analysis",
        "🧪 Interactive Predictor",
        "📈 Model Performance"
    ])
    
    with tab1:
        st.subheader("Intensity Predictions")
        
        # Sample data for visualization
        sample_size = min(5000, len(df))
        sample_df = df.sample(n=sample_size, random_state=42).copy()
        
        # Make predictions
        with st.spinner("Making predictions..."):
            if use_simple_model:
                sample_df['predicted_intensity'] = simple_predictor(sample_df)
            else:
                sample_df['predicted_intensity'] = predict_intensity(model_data, sample_df)
        
        # Comparison chart
        if 'event_intensity' in sample_df.columns:
            fig_compare = create_prediction_chart(
                sample_df.sort_values('timestamp' if 'timestamp' in sample_df.columns else sample_df.index).head(500),
                'event_intensity',
                'predicted_intensity'
            )
            st.plotly_chart(fig_compare, use_container_width=True)
            
            # Metrics
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            y_true = sample_df['event_intensity']
            y_pred = sample_df['predicted_intensity']
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("R² Score", f"{r2_score(y_true, y_pred):.4f}")
            with col2:
                st.metric("RMSE", f"{np.sqrt(mean_squared_error(y_true, y_pred)):.4f}")
            with col3:
                st.metric("MAE", f"{mean_absolute_error(y_true, y_pred):.4f}")
            with col4:
                correlation = np.corrcoef(y_true, y_pred)[0, 1]
                st.metric("Correlation", f"{correlation:.4f}")
        
        # Distribution chart
        fig_dist = create_intensity_distribution(sample_df, 'predicted_intensity')
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with tab2:
        st.subheader("Feature Analysis")
        
        if not use_simple_model and 'feature_importance' in model_data:
            fig_importance = create_feature_importance_chart(model_data['feature_importance'])
            st.plotly_chart(fig_importance, use_container_width=True)
        else:
            st.info("Feature importance available only with trained model.")
            
            # Show correlation analysis instead
            st.markdown("### Feature Correlations with Intensity")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if 'event_intensity' in numeric_cols:
                correlations = df[numeric_cols].corr()['event_intensity'].drop('event_intensity').sort_values(ascending=False)
                
                fig_corr = px.bar(
                    x=correlations.values,
                    y=correlations.index,
                    orientation='h',
                    title='Correlation with Event Intensity',
                    labels={'x': 'Correlation', 'y': 'Feature'},
                    color=correlations.values,
                    color_continuous_scale='RdBu_r'
                )
                fig_corr.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_corr, use_container_width=True)
    
    with tab3:
        st.subheader("Interactive Intensity Predictor")
        st.markdown("Adjust the parameters below to predict event intensity:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            input_temp = st.slider("Temperature (°C)", -30.0, 50.0, 25.0, 0.5)
            input_humidity = st.slider("Humidity (%)", 0.0, 100.0, 65.0, 1.0)
            input_pressure = st.slider("Pressure (hPa)", 950.0, 1050.0, 1013.0, 1.0)
            input_wind = st.slider("Wind Speed (km/h)", 0.0, 200.0, 20.0, 1.0)
        
        with col2:
            input_rain = st.slider("Precipitation (mm)", 0.0, 100.0, 5.0, 0.5)
            input_anomaly = st.slider("Anomaly Score (σ)", -5.0, 5.0, 0.0, 0.1)
            input_month = st.selectbox("Month", list(range(1, 13)), index=5)
            input_hour = st.slider("Hour of Day", 0, 23, 12)
        
        # Create input dataframe
        input_df = pd.DataFrame({
            'temperature_hourly': [input_temp],
            'humidity_pct': [input_humidity],
            'pressure_hpa': [input_pressure],
            'wind_speed_kmh': [input_wind],
            'rain_mm': [input_rain],
            'anomaly_score': [input_anomaly],
            'month': [input_month],
            'hour': [input_hour],
            'lat_decimal': [0.0],
            'lon_decimal': [0.0]
        })
        
        # Predict
        if use_simple_model:
            predicted = simple_predictor(input_df)[0]
        else:
            predicted = predict_intensity(model_data, input_df)[0]
        
        # Display result
        st.markdown("---")
        st.markdown("### Prediction Result")
        
        # Gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=predicted * 100,
            title={'text': "Predicted Intensity"},
            number={'suffix': '%'},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#2C3E50"},
                'steps': [
                    {'range': [0, 20], 'color': "#82E0AA"},
                    {'range': [20, 40], 'color': "#F9E79F"},
                    {'range': [40, 60], 'color': "#F5B041"},
                    {'range': [60, 80], 'color': "#E74C3C"},
                    {'range': [80, 100], 'color': "#8E44AD"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 80
                }
            }
        ))
        fig_gauge.update_layout(height=300)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        with col2:
            # Severity interpretation
            if predicted < 0.2:
                level = "Minor"
                color = "#27AE60"
                emoji = "✅"
            elif predicted < 0.4:
                level = "Moderate"
                color = "#F1C40F"
                emoji = "⚠️"
            elif predicted < 0.6:
                level = "Significant"
                color = "#E67E22"
                emoji = "🟠"
            elif predicted < 0.8:
                level = "Severe"
                color = "#E74C3C"
                emoji = "🔴"
            else:
                level = "Extreme"
                color = "#8E44AD"
                emoji = "🚨"
            
            st.markdown(f"""
            <div style='background-color: {color}; padding: 20px; border-radius: 10px; text-align: center; color: white;'>
                <h2>{emoji} {level}</h2>
                <p>Intensity: {predicted:.2%}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab4:
        st.subheader("Model Performance Analysis")
        
        if not use_simple_model:
            st.markdown("### Trained Model Metrics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("R² Score", f"{model_data['metrics']['r2']:.4f}")
                st.metric("RMSE", f"{model_data['metrics']['rmse']:.4f}")
            
            with col2:
                st.markdown("**Features Used:**")
                for feat in model_data['features']:
                    st.write(f"- {feat.replace('_', ' ').title()}")
        else:
            st.info("""
            Model performance metrics will be available after training.
            
            The simple heuristic model uses rule-based calculations:
            - 30% weight on anomaly score
            - 30% weight on wind speed
            - 20% weight on precipitation
            - 20% weight on temperature deviation
            
            For better predictions, train the ML model using:
            ```bash
            climaxtreme train-intensity-model
            ```
            """)
        
        # Residual analysis if we have predictions
        if 'predicted_intensity' in sample_df.columns and 'event_intensity' in sample_df.columns:
            st.markdown("### Residual Analysis")
            
            residuals = sample_df['event_intensity'] - sample_df['predicted_intensity']
            
            fig_residuals = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Residual Distribution', 'Predicted vs Actual')
            )
            
            fig_residuals.add_trace(
                go.Histogram(x=residuals, nbinsx=50, marker_color='#3498DB'),
                row=1, col=1
            )
            
            fig_residuals.add_trace(
                go.Scatter(
                    x=sample_df['predicted_intensity'],
                    y=sample_df['event_intensity'],
                    mode='markers',
                    marker=dict(size=3, opacity=0.5, color='#E74C3C')
                ),
                row=1, col=2
            )
            
            # Add perfect prediction line
            fig_residuals.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode='lines',
                    line=dict(color='black', dash='dash'),
                    name='Perfect'
                ),
                row=1, col=2
            )
            
            fig_residuals.update_layout(height=400, showlegend=False)
            fig_residuals.update_xaxes(title_text='Residual', row=1, col=1)
            fig_residuals.update_xaxes(title_text='Predicted', row=1, col=2)
            fig_residuals.update_yaxes(title_text='Actual', row=1, col=2)
            
            st.plotly_chart(fig_residuals, use_container_width=True)


if __name__ == "__main__":
    main()
