"""
🔮 Intensity Prediction Page
Predicción de intensidad de eventos climáticos basada en anomalías detectadas.
Usa datos de HDFS (anomalies.parquet) para análisis y predicción.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple
from pathlib import Path
import subprocess
import json

# Configuración de página
st.set_page_config(
    page_title="Predicción de Intensidad - climaXtreme",
    page_icon="🔮",
    layout="wide"
)

try:
    from climaxtreme.dashboard.utils import configure_sidebar
    from climaxtreme.dashboard.components.data_checker import show_hdfs_connection_status
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from climaxtreme.dashboard.utils import configure_sidebar
    from climaxtreme.dashboard.components.data_checker import show_hdfs_connection_status


@st.cache_data(ttl=300)
def load_anomalies_from_hdfs(sample_size: int = 50000) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Cargar datos de anomalías desde HDFS.
    
    Returns:
        (DataFrame, mensaje de estado)
    """
    try:
        # Intentar con PySpark directamente (dentro de contenedor)
        try:
            from pyspark.sql import SparkSession
            import os
            
            if os.path.exists("/.dockerenv"):
                spark = SparkSession.builder \
                    .appName("LoadAnomalies") \
                    .config("spark.driver.memory", "1g") \
                    .config("spark.ui.enabled", "false") \
                    .getOrCreate()
                
                df_spark = spark.read.parquet(
                    "hdfs://climaxtreme-namenode:9000/data/climaxtreme/processed/anomalies.parquet"
                )
                
                # Tomar muestra
                total = df_spark.count()
                fraction = min(1.0, sample_size / total)
                df_sample = df_spark.sample(fraction=fraction, seed=42).limit(sample_size)
                
                df = df_sample.toPandas()
                spark.stop()
                
                return df, f"✅ Cargados {len(df):,} registros de {total:,} totales desde HDFS"
        except:
            pass
        
        # Fallback: usar subprocess con docker
        script = f'''
import json
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("LoadAnomalies").getOrCreate()
try:
    df = spark.read.parquet("hdfs://climaxtreme-namenode:9000/data/climaxtreme/processed/anomalies.parquet")
    total = df.count()
    sample = df.sample(fraction=min(1.0, {sample_size}/total), seed=42).limit({sample_size})
    data = sample.toPandas().to_json(orient='records', date_format='iso')
    print("DATA_START")
    print(data)
    print("DATA_END")
    print(f"TOTAL:{total}")
except Exception as e:
    print(f"ERROR:{e}")
spark.stop()
'''
        
        cmd = ["docker", "exec", "climaxtreme-processor", "python", "-c", script]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if "DATA_START" in result.stdout:
            start = result.stdout.find("DATA_START") + len("DATA_START")
            end = result.stdout.find("DATA_END")
            data_json = result.stdout[start:end].strip()
            
            # Extraer total
            total = 0
            if "TOTAL:" in result.stdout:
                total_line = result.stdout.split("TOTAL:")[1].split("\n")[0]
                total = int(total_line)
            
            df = pd.DataFrame(json.loads(data_json))
            return df, f"✅ Cargados {len(df):,} registros de {total:,} totales desde HDFS"
        
        return None, "❌ No se pudieron cargar datos de HDFS"
        
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


def create_anomaly_intensity_model():
    """
    Modelo heurístico para calcular intensidad basado en temp_zscore.
    """
    def predict(df: pd.DataFrame) -> np.ndarray:
        intensity = np.zeros(len(df))
        
        # Usar temp_zscore como principal indicador de intensidad
        if 'temp_zscore' in df.columns:
            # Normalizar z-score a 0-1 (valores entre -5 y 5 típicamente)
            zscore = df['temp_zscore'].fillna(0).clip(-5, 5)
            intensity = (np.abs(zscore) / 5.0) * 0.6  # 60% peso al z-score
        
        # Agregar componente estacional (meses extremos más intensos)
        if 'month' in df.columns:
            # Enero/Julio son típicamente más extremos
            month_factor = np.where(
                df['month'].isin([1, 2, 7, 8, 12]),
                0.2, 0.1
            )
            intensity += month_factor
        
        # Factor por incertidumbre (mayor incertidumbre = menos confiable)
        if 'uncertainty' in df.columns:
            uncertainty_factor = (1 - df['uncertainty'].fillna(0).clip(0, 5) / 10) * 0.2
            intensity += uncertainty_factor
        
        return np.clip(intensity, 0, 1)
    
    return predict


def create_intensity_distribution_chart(df: pd.DataFrame, intensity_col: str) -> go.Figure:
    """Crear gráfico de distribución de intensidad."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Distribución de Intensidad', 'Intensidad por Continente'),
        specs=[[{"type": "histogram"}, {"type": "box"}]]
    )
    
    # Histograma
    fig.add_trace(
        go.Histogram(
            x=df[intensity_col],
            nbinsx=30,
            marker_color='#3498DB',
            name='Distribución'
        ),
        row=1, col=1
    )
    
    # Box plot por continente
    if 'continent' in df.columns:
        continents = df['continent'].dropna().unique()
        colors = px.colors.qualitative.Set2
        
        for i, continent in enumerate(continents[:6]):  # Limitar a 6
            continent_data = df[df['continent'] == continent][intensity_col]
            fig.add_trace(
                go.Box(
                    y=continent_data,
                    name=continent[:12],
                    marker_color=colors[i % len(colors)]
                ),
                row=1, col=2
            )
    
    fig.update_layout(height=400, showlegend=False)
    fig.update_xaxes(title_text='Intensidad', row=1, col=1)
    fig.update_yaxes(title_text='Frecuencia', row=1, col=1)
    fig.update_yaxes(title_text='Intensidad', row=1, col=2)
    
    return fig


def create_zscore_vs_intensity_chart(df: pd.DataFrame) -> go.Figure:
    """Crear gráfico de relación zscore vs intensidad."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Z-Score de Temperatura', 'Relación Z-Score vs Intensidad')
    )
    
    # Histograma de z-scores
    if 'temp_zscore' in df.columns:
        fig.add_trace(
            go.Histogram(
                x=df['temp_zscore'],
                nbinsx=50,
                marker_color='#E74C3C',
                name='Z-Score'
            ),
            row=1, col=1
        )
        
        # Scatter z-score vs intensidad
        if 'predicted_intensity' in df.columns:
            sample = df.sample(min(1000, len(df)))
            fig.add_trace(
                go.Scatter(
                    x=sample['temp_zscore'],
                    y=sample['predicted_intensity'],
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=sample['predicted_intensity'],
                        colorscale='RdYlGn_r',
                        opacity=0.6
                    ),
                    name='Intensidad'
                ),
                row=1, col=2
            )
    
    fig.update_layout(height=400, showlegend=False)
    fig.update_xaxes(title_text='Z-Score (σ)', row=1, col=1)
    fig.update_xaxes(title_text='Z-Score (σ)', row=1, col=2)
    fig.update_yaxes(title_text='Frecuencia', row=1, col=1)
    fig.update_yaxes(title_text='Intensidad Predicha', row=1, col=2)
    
    return fig


def create_temporal_intensity_chart(df: pd.DataFrame) -> go.Figure:
    """Crear gráfico de evolución temporal de intensidad."""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Intensidad Media por Año', 'Intensidad Media por Mes'),
        vertical_spacing=0.15
    )
    
    # Por año
    if 'year' in df.columns and 'predicted_intensity' in df.columns:
        yearly = df.groupby('year')['predicted_intensity'].mean().reset_index()
        fig.add_trace(
            go.Scatter(
                x=yearly['year'],
                y=yearly['predicted_intensity'],
                mode='lines+markers',
                line=dict(color='#3498DB', width=2),
                marker=dict(size=6),
                name='Intensidad Anual'
            ),
            row=1, col=1
        )
    
    # Por mes
    if 'month' in df.columns and 'predicted_intensity' in df.columns:
        monthly = df.groupby('month')['predicted_intensity'].mean().reset_index()
        fig.add_trace(
            go.Bar(
                x=['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 
                   'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'][:len(monthly)],
                y=monthly['predicted_intensity'],
                marker_color='#E74C3C',
                name='Intensidad Mensual'
            ),
            row=2, col=1
        )
    
    fig.update_layout(height=500, showlegend=False)
    fig.update_yaxes(title_text='Intensidad Media', row=1, col=1)
    fig.update_yaxes(title_text='Intensidad Media', row=2, col=1)
    
    return fig


def create_geographic_intensity_map(df: pd.DataFrame) -> go.Figure:
    """Crear mapa de intensidad geográfica."""
    if 'lat_numeric' not in df.columns or 'lon_numeric' not in df.columns:
        return go.Figure()
    
    # Agregar por ciudad
    if 'city' in df.columns:
        city_stats = df.groupby(['city', 'country', 'lat_numeric', 'lon_numeric']).agg({
            'predicted_intensity': 'mean',
            'temp_zscore': 'mean'
        }).reset_index()
        
        fig = go.Figure(go.Scattergeo(
            lat=city_stats['lat_numeric'],
            lon=city_stats['lon_numeric'],
            mode='markers',
            marker=dict(
                size=city_stats['predicted_intensity'] * 15 + 5,
                color=city_stats['predicted_intensity'],
                colorscale='RdYlGn_r',
                colorbar=dict(title='Intensidad'),
                opacity=0.7
            ),
            text=city_stats.apply(
                lambda r: f"{r['city']}, {r['country']}<br>Intensidad: {r['predicted_intensity']:.2f}",
                axis=1
            ),
            hoverinfo='text'
        ))
    else:
        # Sin ciudad, usar puntos individuales
        sample = df.sample(min(1000, len(df)))
        fig = go.Figure(go.Scattergeo(
            lat=sample['lat_numeric'],
            lon=sample['lon_numeric'],
            mode='markers',
            marker=dict(
                size=8,
                color=sample['predicted_intensity'],
                colorscale='RdYlGn_r',
                colorbar=dict(title='Intensidad'),
                opacity=0.6
            )
        ))
    
    fig.update_layout(
        title='🗺️ Mapa de Intensidad por Ubicación',
        geo=dict(
            showland=True,
            landcolor='rgb(243, 243, 243)',
            showocean=True,
            oceancolor='rgb(210, 235, 255)',
            showcountries=True,
            projection_type='natural earth'
        ),
        height=500
    )
    
    return fig


def main():
    configure_sidebar()
    show_hdfs_connection_status()
    
    st.title("🔮 Predicción de Intensidad de Eventos")
    st.markdown("""
    Análisis y predicción de intensidad de eventos climáticos basado en anomalías
    de temperatura detectadas en datos históricos (HDFS).
    """)
    
    # Info del modelo
    with st.expander("📖 Información del Modelo", expanded=False):
        st.markdown("""
        ### Modelo de Intensidad Basado en Anomalías
        
        **Algoritmo**: Modelo heurístico basado en Z-Score de temperatura
        
        **Variables utilizadas**:
        - **temp_zscore**: Desviación estándar de la temperatura respecto a la climatología (60% peso)
        - **month**: Factor estacional (20% peso) - meses extremos ponderados más alto
        - **uncertainty**: Factor de incertidumbre del dato (20% peso)
        
        **Interpretación de Intensidad**:
        - **0.0 - 0.2**: Menor / Condiciones normales
        - **0.2 - 0.4**: Evento moderado
        - **0.4 - 0.6**: Evento significativo
        - **0.6 - 0.8**: Evento severo
        - **0.8 - 1.0**: Extremo / Nivel de emergencia
        
        **Fuente de datos**: anomalies.parquet desde HDFS
        """)
    
    # Cargar datos
    with st.spinner("Cargando datos de anomalías desde HDFS..."):
        df, status_msg = load_anomalies_from_hdfs(sample_size=50000)
    
    if df is None or len(df) == 0:
        st.error(f"""
        ❌ No se encontraron datos de anomalías
        
        {status_msg}
        
        **Solución**: Asegúrate de que el procesamiento de datos se haya completado
        y que `anomalies.parquet` exista en HDFS.
        """)
        st.stop()
    
    st.success(status_msg)
    
    # Crear predicciones de intensidad
    predictor = create_anomaly_intensity_model()
    df['predicted_intensity'] = predictor(df)
    
    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Registros Analizados", f"{len(df):,}")
    with col2:
        avg_intensity = df['predicted_intensity'].mean()
        st.metric("📈 Intensidad Media", f"{avg_intensity:.2%}")
    with col3:
        anomalies_count = (df['is_anomaly'] == True).sum() if 'is_anomaly' in df.columns else 0
        st.metric("⚠️ Anomalías Detectadas", f"{anomalies_count:,}")
    with col4:
        extreme_count = (df['predicted_intensity'] > 0.6).sum()
        st.metric("🔴 Eventos Severos", f"{extreme_count:,}")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Análisis de Intensidad", 
        "🗺️ Distribución Geográfica",
        "🧪 Predictor Interactivo",
        "📈 Análisis Temporal"
    ])
    
    with tab1:
        st.subheader("Distribución de Intensidad")
        
        fig_dist = create_intensity_distribution_chart(df, 'predicted_intensity')
        st.plotly_chart(fig_dist, use_container_width=True)
        
        st.subheader("Relación Z-Score vs Intensidad")
        fig_zscore = create_zscore_vs_intensity_chart(df)
        st.plotly_chart(fig_zscore, use_container_width=True)
        
        # Top ciudades con mayor intensidad
        if 'city' in df.columns:
            st.subheader("🏙️ Ciudades con Mayor Intensidad Promedio")
            city_intensity = df.groupby(['city', 'country']).agg({
                'predicted_intensity': 'mean',
                'temp_zscore': 'mean'
            }).reset_index().sort_values('predicted_intensity', ascending=False).head(15)
            
            fig_cities = px.bar(
                city_intensity,
                x='predicted_intensity',
                y='city',
                orientation='h',
                color='predicted_intensity',
                color_continuous_scale='RdYlGn_r',
                title='Top 15 Ciudades por Intensidad',
                labels={'predicted_intensity': 'Intensidad', 'city': 'Ciudad'}
            )
            fig_cities.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500)
            st.plotly_chart(fig_cities, use_container_width=True)
    
    with tab2:
        st.subheader("Mapa de Intensidad")
        
        fig_map = create_geographic_intensity_map(df)
        st.plotly_chart(fig_map, use_container_width=True)
        
        # Intensidad por continente
        if 'continent' in df.columns:
            st.subheader("Intensidad por Continente")
            continent_stats = df.groupby('continent').agg({
                'predicted_intensity': ['mean', 'std', 'count']
            }).round(3)
            continent_stats.columns = ['Intensidad Media', 'Desv. Estándar', 'Registros']
            continent_stats = continent_stats.sort_values('Intensidad Media', ascending=False)
            st.dataframe(continent_stats, use_container_width=True)
    
    with tab3:
        st.subheader("Predictor Interactivo de Intensidad")
        st.markdown("Ajusta los parámetros para predecir la intensidad de un evento:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            input_zscore = st.slider(
                "Z-Score de Temperatura (σ)",
                -5.0, 5.0, 0.0, 0.1,
                help="Desviaciones estándar respecto a la media histórica"
            )
            input_month = st.selectbox(
                "Mes", 
                [(i, m) for i, m in enumerate(['Enero', 'Febrero', 'Marzo', 'Abril', 
                                               'Mayo', 'Junio', 'Julio', 'Agosto',
                                               'Septiembre', 'Octubre', 'Noviembre', 'Diciembre'], 1)],
                format_func=lambda x: x[1]
            )[0]
        
        with col2:
            input_uncertainty = st.slider(
                "Incertidumbre del dato",
                0.0, 5.0, 1.0, 0.1,
                help="Mayor incertidumbre = menor confianza"
            )
            input_temp = st.slider(
                "Temperatura (°C) - referencia",
                -40.0, 50.0, 20.0, 0.5
            )
        
        # Calcular predicción
        input_df = pd.DataFrame({
            'temp_zscore': [input_zscore],
            'month': [input_month],
            'uncertainty': [input_uncertainty],
            'temperature': [input_temp]
        })
        
        predicted = predictor(input_df)[0]
        
        # Mostrar resultado
        st.markdown("---")
        st.markdown("### 🎯 Resultado de la Predicción")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=predicted * 100,
                title={'text': "Intensidad Predicha"},
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
                        'value': 60
                    }
                }
            ))
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        with col2:
            # Interpretación
            if predicted < 0.2:
                level, color, emoji = "Menor", "#27AE60", "✅"
                desc = "Condiciones normales"
            elif predicted < 0.4:
                level, color, emoji = "Moderado", "#F1C40F", "⚠️"
                desc = "Desviación notable"
            elif predicted < 0.6:
                level, color, emoji = "Significativo", "#E67E22", "🟠"
                desc = "Evento importante"
            elif predicted < 0.8:
                level, color, emoji = "Severo", "#E74C3C", "🔴"
                desc = "Condiciones peligrosas"
            else:
                level, color, emoji = "Extremo", "#8E44AD", "🚨"
                desc = "Emergencia climática"
            
            st.markdown(f"""
            <div style='background-color: {color}; padding: 20px; border-radius: 10px; 
                        text-align: center; color: white;'>
                <h2>{emoji} {level}</h2>
                <p>Intensidad: {predicted:.1%}</p>
                <small>{desc}</small>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            **Factores:**
            - Z-Score: {input_zscore:.1f}σ
            - Mes: {['', 'Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 
                    'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre'][input_month]}
            - Incertidumbre: {input_uncertainty:.1f}
            """)
    
    with tab4:
        st.subheader("Evolución Temporal de la Intensidad")
        
        fig_temporal = create_temporal_intensity_chart(df)
        st.plotly_chart(fig_temporal, use_container_width=True)
        
        # Tabla de estadísticas por año
        if 'year' in df.columns:
            st.subheader("Estadísticas por Año")
            yearly_stats = df.groupby('year').agg({
                'predicted_intensity': ['mean', 'std', 'max'],
                'temp_zscore': 'mean'
            }).round(3)
            yearly_stats.columns = ['Intensidad Media', 'Desv. Std', 'Intensidad Máx', 'Z-Score Medio']
            st.dataframe(yearly_stats.tail(20), use_container_width=True)


if __name__ == "__main__":
    main()
