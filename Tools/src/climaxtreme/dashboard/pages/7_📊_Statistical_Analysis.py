"""
Página de Análisis Estadístico - Estadísticas descriptivas, correlaciones y tests Chi-cuadrado
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

st.set_page_config(page_title="Análisis Estadístico", page_icon="📊", layout="wide")
configure_sidebar()

st.title("📊 Análisis Estadístico")
st.markdown("Análisis riguroso de distribuciones, correlaciones y pruebas de independencia")

data_source = DataSource()

tab1, tab2, tab3 = st.tabs([
    "📈 Estadísticas Descriptivas", 
    "🔗 Correlaciones", 
    "🧪 Test Chi-Cuadrado"
])

# =============================================================================
# TAB 1: ESTADÍSTICAS DESCRIPTIVAS
# =============================================================================
with tab1:
    st.header("📈 Análisis Estadístico por Variable")
    
    stats_df = data_source.load_parquet('descriptive_stats.parquet')
    
    if stats_df is not None and not stats_df.empty:
        show_data_info(stats_df, "Dataset de Estadísticas Descriptivas")
        
        # Asegurar columna 'variable'
        if 'variable' not in stats_df.columns:
            stats_df['variable'] = 'temperature'
        
        # Selector de variable
        variables = stats_df['variable'].unique().tolist()
        
        if len(variables) > 1:
            selected_var = st.selectbox(
                "🔍 Seleccionar Variable a Analizar:",
                variables,
                format_func=lambda x: x.replace('_', ' ').title()
            )
            row = stats_df[stats_df['variable'] == selected_var].iloc[0]
        else:
            selected_var = variables[0]
            row = stats_df.iloc[0]
        
        var_title = selected_var.replace('_', ' ').title()
        
        st.markdown(f"## 📊 Análisis de: **{var_title}**")
        st.markdown("---")
        
        # =====================================================================
        # SECCIÓN 1: RESUMEN EJECUTIVO
        # =====================================================================
        st.subheader("📋 Resumen Ejecutivo")
        
        # Crear cards de resumen en una fila
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            n = row.get('count', 0)
            st.metric(
                label="📊 Observaciones",
                value=f"{int(n):,}" if pd.notna(n) else "N/A"
            )
        
        with col2:
            mean_val = row.get('mean', 0)
            st.metric(
                label="📍 Media",
                value=f"{mean_val:.2f}°C" if pd.notna(mean_val) else "N/A"
            )
        
        with col3:
            std_val = row.get('std', 0)
            st.metric(
                label="📏 Desv. Estándar",
                value=f"{std_val:.2f}°C" if pd.notna(std_val) else "N/A"
            )
        
        with col4:
            normality = row.get('normality_assessment', 'Desconocido')
            if 'Normal' in normality and 'Non' not in normality:
                st.metric(label="🎯 Normalidad", value="✅ Normal")
            elif 'Moderately' in normality:
                st.metric(label="🎯 Normalidad", value="⚠️ Moderada")
            else:
                st.metric(label="🎯 Normalidad", value="❌ No Normal")
        
        st.markdown("---")
        
        # =====================================================================
        # SECCIÓN 2: MEDIDAS DE TENDENCIA CENTRAL Y DISPERSIÓN
        # =====================================================================
        with st.expander("📐 **1. Medidas de Tendencia Central y Dispersión**", expanded=True):
            
            col_left, col_right = st.columns(2)
            
            with col_left:
                st.markdown("#### 📍 Tendencia Central")
                
                central_data = {
                    "Medida": ["Media (μ)", "Mediana", "Error Estándar (SE)"],
                    "Valor": [
                        f"{row.get('mean', 0):.4f}" if pd.notna(row.get('mean')) else "N/A",
                        f"{row.get('median', 0):.4f}" if pd.notna(row.get('median')) else "N/A",
                        f"{row.get('se_mean', 0):.4f}" if pd.notna(row.get('se_mean')) else "N/A"
                    ],
                    "Interpretación": [
                        "Promedio aritmético de todos los valores",
                        "Valor que divide la distribución en dos mitades",
                        "Precisión de la estimación de la media"
                    ]
                }
                st.dataframe(pd.DataFrame(central_data), hide_index=True, use_container_width=True)
            
            with col_right:
                st.markdown("#### 📏 Dispersión")
                
                dispersion_data = {
                    "Medida": ["Desv. Estándar (σ)", "Varianza (σ²)", "Coef. Variación (CV)", "Rango", "Rango Intercuartil (IQR)"],
                    "Valor": [
                        f"{row.get('std', 0):.4f}" if pd.notna(row.get('std')) else "N/A",
                        f"{row.get('variance', 0):.4f}" if pd.notna(row.get('variance')) else "N/A",
                        f"{row.get('cv_percent', 0):.2f}%" if pd.notna(row.get('cv_percent')) else "N/A",
                        f"{row.get('range', 0):.4f}" if pd.notna(row.get('range')) else "N/A",
                        f"{row.get('iqr', 0):.4f}" if pd.notna(row.get('iqr')) else "N/A"
                    ]
                }
                st.dataframe(pd.DataFrame(dispersion_data), hide_index=True, use_container_width=True)
        
        # =====================================================================
        # SECCIÓN 3: PERCENTILES Y DISTRIBUCIÓN
        # =====================================================================
        with st.expander("📊 **2. Percentiles y Distribución de Cuantiles**", expanded=True):
            
            st.markdown("#### Distribución por Percentiles")
            
            # Tabla de percentiles
            percentile_data = {
                "Percentil": ["Mínimo (0%)", "P5", "P10", "Q1 (25%)", "Mediana (50%)", 
                             "Q3 (75%)", "P90", "P95", "Máximo (100%)"],
                "Valor": [
                    f"{row.get('min', 0):.2f}" if pd.notna(row.get('min')) else "N/A",
                    f"{row.get('p5', 0):.2f}" if pd.notna(row.get('p5')) else "N/A",
                    f"{row.get('p10', 0):.2f}" if pd.notna(row.get('p10')) else "N/A",
                    f"{row.get('q1', 0):.2f}" if pd.notna(row.get('q1')) else "N/A",
                    f"{row.get('median', 0):.2f}" if pd.notna(row.get('median')) else "N/A",
                    f"{row.get('q3', 0):.2f}" if pd.notna(row.get('q3')) else "N/A",
                    f"{row.get('p90', 0):.2f}" if pd.notna(row.get('p90')) else "N/A",
                    f"{row.get('p95', 0):.2f}" if pd.notna(row.get('p95')) else "N/A",
                    f"{row.get('max', 0):.2f}" if pd.notna(row.get('max')) else "N/A"
                ]
            }
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.dataframe(pd.DataFrame(percentile_data), hide_index=True, use_container_width=True)
            
            with col2:
                # Gráfico de percentiles
                percentiles_keys = ['min', 'p5', 'p10', 'q1', 'median', 'q3', 'p90', 'p95', 'max']
                percentiles_labels = ['Min', 'P5', 'P10', 'Q1', 'Med', 'Q3', 'P90', 'P95', 'Max']
                p_values = [row.get(k, 0) or 0 for k in percentiles_keys]
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=percentiles_labels,
                    y=p_values,
                    mode='lines+markers+text',
                    text=[f"{v:.1f}" for v in p_values],
                    textposition="top center",
                    line=dict(color='#1f77b4', width=3),
                    marker=dict(size=12, color='#1f77b4')
                ))
                
                # Línea de la media
                mean_val = row.get('mean', 0) or 0
                fig.add_hline(y=mean_val, line_dash="dash", line_color="red",
                             annotation_text=f"Media: {mean_val:.2f}")
                
                fig.update_layout(
                    title="Distribución de Percentiles",
                    xaxis_title="Percentil",
                    yaxis_title="Valor (°C)",
                    height=350,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # =====================================================================
        # SECCIÓN 4: FORMA DE LA DISTRIBUCIÓN
        # =====================================================================
        with st.expander("📈 **3. Análisis de Forma de la Distribución**", expanded=True):
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### ⚖️ Asimetría (Skewness)")
                
                skew = row.get('skewness', 0) or 0
                skew_interp = row.get('skewness_interpretation', 'Desconocido')
                
                # Traducir interpretación
                skew_es = skew_interp.replace('Highly Left-Skewed', 'Muy Sesgada a la Izquierda') \
                                     .replace('Moderately Left-Skewed', 'Moderadamente Sesgada a la Izquierda') \
                                     .replace('Approximately Symmetric', 'Aproximadamente Simétrica') \
                                     .replace('Moderately Right-Skewed', 'Moderadamente Sesgada a la Derecha') \
                                     .replace('Highly Right-Skewed', 'Muy Sesgada a la Derecha')
                
                st.metric("Coeficiente de Asimetría (γ₁)", f"{skew:.4f}")
                
                if skew < -0.5:
                    st.error(f"📉 {skew_es}")
                    st.caption("Cola larga hacia valores bajos")
                elif skew > 0.5:
                    st.warning(f"📈 {skew_es}")
                    st.caption("Cola larga hacia valores altos")
                else:
                    st.success(f"⚖️ {skew_es}")
                    st.caption("Distribución balanceada")
                
                st.markdown("""
                **Guía de Interpretación:**
                | Valor | Interpretación |
                |-------|----------------|
                | γ₁ < -1 | Muy sesgada izquierda |
                | -1 a -0.5 | Moderadamente sesgada izquierda |
                | -0.5 a 0.5 | **Simétrica** ✓ |
                | 0.5 a 1 | Moderadamente sesgada derecha |
                | γ₁ > 1 | Muy sesgada derecha |
                """)
            
            with col2:
                st.markdown("#### 🔔 Curtosis (Kurtosis)")
                
                kurt = row.get('kurtosis', 0) or 0
                kurt_interp = row.get('kurtosis_interpretation', 'Desconocido')
                
                # Traducir interpretación
                kurt_es = kurt_interp.replace('Platykurtic (Light Tails)', 'Platicúrtica (Colas Ligeras)') \
                                     .replace('Mesokurtic (Normal Tails)', 'Mesocúrtica (Colas Normales)') \
                                     .replace('Leptokurtic (Heavy Tails)', 'Leptocúrtica (Colas Pesadas)')
                
                st.metric("Exceso de Curtosis (γ₂)", f"{kurt:.4f}")
                
                if kurt < -1:
                    st.info(f"📊 {kurt_es}")
                    st.caption("Menos valores extremos que la normal")
                elif kurt > 1:
                    st.warning(f"📊 {kurt_es}")
                    st.caption("Más valores extremos que la normal")
                else:
                    st.success(f"📊 {kurt_es}")
                    st.caption("Similar a distribución normal")
                
                st.markdown("""
                **Guía de Interpretación:**
                | Valor | Tipo | Descripción |
                |-------|------|-------------|
                | γ₂ < -1 | Platicúrtica | Colas más ligeras |
                | -1 a 1 | **Mesocúrtica** ✓ | Similar a normal |
                | γ₂ > 1 | Leptocúrtica | Colas más pesadas |
                """)
        
        # =====================================================================
        # SECCIÓN 5: TEST DE NORMALIDAD
        # =====================================================================
        with st.expander("🎯 **4. Evaluación de Normalidad**", expanded=True):
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                jb_stat = row.get('jb_statistic', None)
                st.metric(
                    "Estadístico Jarque-Bera",
                    f"{jb_stat:.2f}" if pd.notna(jb_stat) else "N/A"
                )
                st.caption("JB = n/6 × (S² + K²/4)")
                st.caption("Valores altos → No normalidad")
            
            with col2:
                normality = row.get('normality_assessment', 'Desconocido')
                
                # Traducir
                norm_es = normality.replace('Approximately Normal', 'Aproximadamente Normal') \
                                   .replace('Moderately Non-Normal', 'Moderadamente No Normal') \
                                   .replace('Significantly Non-Normal', 'Significativamente No Normal')
                
                if 'Approximately Normal' in normality:
                    st.success(f"### ✅ {norm_es}")
                    st.markdown("La variable **sigue aproximadamente** una distribución normal.")
                elif 'Moderately' in normality:
                    st.warning(f"### ⚠️ {norm_es}")
                    st.markdown("Desviaciones **moderadas** de la normalidad.")
                else:
                    st.error(f"### ❌ {norm_es}")
                    st.markdown("La variable **no sigue** una distribución normal.")
            
            with col3:
                st.markdown("#### Criterios de Decisión")
                st.markdown("""
                | Condición | Resultado |
                |-----------|-----------|
                | \|Skew\| < 0.5 **Y** \|Kurt\| < 1 | ✅ Normal |
                | \|Skew\| < 1 **Y** \|Kurt\| < 2 | ⚠️ Moderada |
                | Otro caso | ❌ No Normal |
                """)
        
        # =====================================================================
        # SECCIÓN 6: DETECCIÓN DE OUTLIERS
        # =====================================================================
        with st.expander("🔍 **5. Detección de Valores Atípicos (Outliers)**", expanded=True):
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Método del Rango Intercuartil (IQR)")
                
                q1 = row.get('q1', 0) or 0
                q3 = row.get('q3', 0) or 0
                iqr = row.get('iqr', 0) or 0
                lower = row.get('lower_fence', None)
                upper = row.get('upper_fence', None)
                
                outlier_data = {
                    "Parámetro": ["Q1 (Cuartil 1)", "Q3 (Cuartil 3)", "IQR (Q3 - Q1)", 
                                  "Límite Inferior", "Límite Superior"],
                    "Fórmula": ["P25", "P75", "Q3 - Q1", "Q1 - 1.5×IQR", "Q3 + 1.5×IQR"],
                    "Valor": [
                        f"{q1:.2f}",
                        f"{q3:.2f}",
                        f"{iqr:.2f}",
                        f"{lower:.2f}" if pd.notna(lower) else "N/A",
                        f"{upper:.2f}" if pd.notna(upper) else "N/A"
                    ]
                }
                st.dataframe(pd.DataFrame(outlier_data), hide_index=True, use_container_width=True)
            
            with col2:
                st.markdown("#### Resultados")
                
                outlier_n = row.get('outlier_count', 0) or 0
                outlier_pct = row.get('outlier_percent', 0) or 0
                
                mcol1, mcol2 = st.columns(2)
                
                with mcol1:
                    st.metric("Cantidad de Outliers", f"{int(outlier_n):,}")
                
                with mcol2:
                    if outlier_pct < 1:
                        st.metric("Porcentaje", f"{outlier_pct:.2f}%", delta="Bajo ✓", delta_color="normal")
                    elif outlier_pct < 5:
                        st.metric("Porcentaje", f"{outlier_pct:.2f}%", delta="Moderado", delta_color="off")
                    else:
                        st.metric("Porcentaje", f"{outlier_pct:.2f}%", delta="Alto ⚠", delta_color="inverse")
                
                st.info(f"""
                **Interpretación:**
                - Valores **menores a {lower:.2f}°C** se consideran outliers bajos
                - Valores **mayores a {upper:.2f}°C** se consideran outliers altos
                - El **{outlier_pct:.2f}%** de los datos son valores atípicos
                """ if pd.notna(lower) and pd.notna(upper) else "No se pudieron calcular los límites")
        
        # =====================================================================
        # SECCIÓN 7: VISUALIZACIONES
        # =====================================================================
        with st.expander("📊 **6. Visualizaciones**", expanded=True):
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Diagrama de Caja (Box Plot)")
                
                q1 = row.get('q1', 0) or 0
                q3 = row.get('q3', 0) or 0
                median_val = row.get('median', 0) or 0
                min_val = row.get('min', 0) or 0
                max_val = row.get('max', 0) or 0
                mean_val = row.get('mean', 0) or 0
                lower_fence = row.get('lower_fence', min_val) or min_val
                upper_fence = row.get('upper_fence', max_val) or max_val
                
                fig_box = go.Figure()
                
                fig_box.add_trace(go.Box(
                    q1=[q1],
                    median=[median_val],
                    q3=[q3],
                    lowerfence=[max(lower_fence, min_val)],
                    upperfence=[min(upper_fence, max_val)],
                    mean=[mean_val],
                    name=var_title,
                    boxmean='sd',
                    orientation='h',
                    marker=dict(color='#1f77b4'),
                    fillcolor='lightsteelblue'
                ))
                
                fig_box.update_layout(
                    title=f"Box Plot - {var_title}",
                    xaxis_title="Valor (°C)",
                    height=300,
                    showlegend=False
                )
                
                st.plotly_chart(fig_box, use_container_width=True)
            
            with col2:
                st.markdown("#### Histograma Estimado")
                
                # Generar datos sintéticos basados en estadísticas
                n_points = 500
                mean = row.get('mean', 0) or 0
                std = row.get('std', 1) or 1
                skew_val = row.get('skewness', 0) or 0
                
                np.random.seed(42)
                if abs(skew_val) < 0.5:
                    synthetic_data = np.random.normal(mean, std, n_points)
                elif skew_val > 0:
                    synthetic_data = np.random.lognormal(np.log(max(mean, 1)), std/mean if mean > 0 else 0.5, n_points)
                    synthetic_data = (synthetic_data - synthetic_data.mean()) / synthetic_data.std() * std + mean
                else:
                    synthetic_data = -np.random.lognormal(0, 0.5, n_points)
                    synthetic_data = (synthetic_data - synthetic_data.mean()) / synthetic_data.std() * std + mean
                
                min_val = row.get('min', mean - 3*std) or (mean - 3*std)
                max_val = row.get('max', mean + 3*std) or (mean + 3*std)
                synthetic_data = np.clip(synthetic_data, min_val, max_val)
                
                fig_hist = go.Figure()
                
                fig_hist.add_trace(go.Histogram(
                    x=synthetic_data,
                    nbinsx=30,
                    name='Distribución',
                    marker_color='lightgreen',
                    opacity=0.7
                ))
                
                fig_hist.add_vline(x=mean, line_dash="solid", line_color="red",
                                   annotation_text=f"Media: {mean:.2f}")
                fig_hist.add_vline(x=row.get('median', mean) or mean, line_dash="dash",
                                   line_color="blue", annotation_text="Mediana")
                
                fig_hist.update_layout(
                    title=f"Distribución Estimada - {var_title}",
                    xaxis_title="Valor (°C)",
                    yaxis_title="Frecuencia",
                    height=300,
                    showlegend=False
                )
                
                st.plotly_chart(fig_hist, use_container_width=True)
        
        # =====================================================================
        # TABLA RESUMEN COMPLETA
        # =====================================================================
        with st.expander("📋 **Tabla Completa de Estadísticas**"):
            
            full_stats = {
                "Categoría": [
                    "Tendencia Central", "Tendencia Central", "Tendencia Central",
                    "Dispersión", "Dispersión", "Dispersión", "Dispersión", "Dispersión",
                    "Percentiles", "Percentiles", "Percentiles", "Percentiles", "Percentiles",
                    "Percentiles", "Percentiles", "Percentiles", "Percentiles",
                    "Forma", "Forma",
                    "Normalidad",
                    "Outliers", "Outliers", "Outliers", "Outliers"
                ],
                "Estadístico": [
                    "N (Observaciones)", "Media (μ)", "Mediana",
                    "Desv. Estándar (σ)", "Varianza (σ²)", "Error Estándar", "Coef. Variación", "Rango",
                    "Mínimo", "P5", "P10", "Q1 (25%)", "Mediana (50%)",
                    "Q3 (75%)", "P90", "P95", "Máximo",
                    "Asimetría (Skewness)", "Curtosis (Kurtosis)",
                    "Estadístico Jarque-Bera",
                    "IQR", "Límite Inferior", "Límite Superior", "% Outliers"
                ],
                "Valor": [
                    f"{int(row.get('count', 0)):,}" if pd.notna(row.get('count')) else "N/A",
                    f"{row.get('mean', 0):.4f}" if pd.notna(row.get('mean')) else "N/A",
                    f"{row.get('median', 0):.4f}" if pd.notna(row.get('median')) else "N/A",
                    f"{row.get('std', 0):.4f}" if pd.notna(row.get('std')) else "N/A",
                    f"{row.get('variance', 0):.4f}" if pd.notna(row.get('variance')) else "N/A",
                    f"{row.get('se_mean', 0):.4f}" if pd.notna(row.get('se_mean')) else "N/A",
                    f"{row.get('cv_percent', 0):.2f}%" if pd.notna(row.get('cv_percent')) else "N/A",
                    f"{row.get('range', 0):.4f}" if pd.notna(row.get('range')) else "N/A",
                    f"{row.get('min', 0):.4f}" if pd.notna(row.get('min')) else "N/A",
                    f"{row.get('p5', 0):.4f}" if pd.notna(row.get('p5')) else "N/A",
                    f"{row.get('p10', 0):.4f}" if pd.notna(row.get('p10')) else "N/A",
                    f"{row.get('q1', 0):.4f}" if pd.notna(row.get('q1')) else "N/A",
                    f"{row.get('median', 0):.4f}" if pd.notna(row.get('median')) else "N/A",
                    f"{row.get('q3', 0):.4f}" if pd.notna(row.get('q3')) else "N/A",
                    f"{row.get('p90', 0):.4f}" if pd.notna(row.get('p90')) else "N/A",
                    f"{row.get('p95', 0):.4f}" if pd.notna(row.get('p95')) else "N/A",
                    f"{row.get('max', 0):.4f}" if pd.notna(row.get('max')) else "N/A",
                    f"{row.get('skewness', 0):.4f}" if pd.notna(row.get('skewness')) else "N/A",
                    f"{row.get('kurtosis', 0):.4f}" if pd.notna(row.get('kurtosis')) else "N/A",
                    f"{row.get('jb_statistic', 0):.4f}" if pd.notna(row.get('jb_statistic')) else "N/A",
                    f"{row.get('iqr', 0):.4f}" if pd.notna(row.get('iqr')) else "N/A",
                    f"{row.get('lower_fence', 0):.4f}" if pd.notna(row.get('lower_fence')) else "N/A",
                    f"{row.get('upper_fence', 0):.4f}" if pd.notna(row.get('upper_fence')) else "N/A",
                    f"{row.get('outlier_percent', 0):.2f}%" if pd.notna(row.get('outlier_percent')) else "N/A"
                ]
            }
            
            st.dataframe(pd.DataFrame(full_stats), hide_index=True, use_container_width=True, height=600)
    
    else:
        st.error("❌ No se pudo cargar descriptive_stats.parquet")


# =============================================================================
# TAB 2: CORRELACIONES
# =============================================================================
with tab2:
    st.header("🔗 Matriz de Correlaciones")
    
    corr_df = data_source.load_parquet('correlation_matrix.parquet')
    
    if corr_df is not None and not corr_df.empty:
        show_data_info(corr_df, "Dataset de Correlaciones")
        
        # Manejar diferentes formatos de datos
        if 'variable_1' in corr_df.columns and 'variable_2' in corr_df.columns:
            variables = sorted(corr_df['variable_1'].unique())
            corr_matrix = pd.DataFrame(index=variables, columns=variables)
            
            for _, row in corr_df.iterrows():
                var1 = row['variable_1']
                var2 = row['variable_2']
                corr = row['correlation']
                corr_matrix.loc[var1, var2] = corr
                corr_matrix.loc[var2, var1] = corr
            
            corr_matrix = corr_matrix.astype(float)
            long_corr = corr_df.copy()
            if 'abs_correlation' not in long_corr.columns:
                long_corr['abs_correlation'] = long_corr['correlation'].abs()
        else:
            if 'variable' in corr_df.columns:
                corr_df = corr_df.set_index('variable')
            
            variables = list(corr_df.columns)
            corr_matrix = corr_df.astype(float)
            
            long_corr_rows = []
            for var1 in variables:
                for var2 in variables:
                    corr_val = corr_matrix.loc[var1, var2] if var1 in corr_matrix.index else 0.0
                    long_corr_rows.append({
                        'variable_1': var1,
                        'variable_2': var2,
                        'correlation': corr_val,
                        'abs_correlation': abs(corr_val)
                    })
            long_corr = pd.DataFrame(long_corr_rows)
        
        # Mapa de calor
        st.subheader("🗺️ Mapa de Calor de Correlaciones")
        
        fig = px.imshow(
            corr_matrix,
            labels=dict(color="Correlación"),
            x=list(corr_matrix.columns),
            y=list(corr_matrix.index),
            color_continuous_scale='RdBu',
            zmin=-1,
            zmax=1,
            title="Matriz de Correlación de Pearson",
            aspect="auto",
            text_auto=".2f"
        )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Guía de interpretación
        st.subheader("📖 Guía de Interpretación")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Coeficiente de Correlación (r):**
            
            | Valor | Interpretación |
            |-------|----------------|
            | r = +1 | Correlación positiva perfecta |
            | r = 0 | Sin correlación lineal |
            | r = -1 | Correlación negativa perfecta |
            """)
        
        with col2:
            st.markdown("""
            **Fuerza de la Correlación:**
            
            | \|r\| | Fuerza |
            |-------|--------|
            | > 0.7 | 🔴 Fuerte |
            | 0.4 - 0.7 | 🟡 Moderada |
            | < 0.4 | 🟢 Débil |
            """)
        
        # Top correlaciones
        st.subheader("🏆 Correlaciones Más Relevantes")
        
        top_corr = long_corr[long_corr['variable_1'] != long_corr['variable_2']].copy()
        top_corr['pair'] = top_corr.apply(
            lambda x: tuple(sorted([x['variable_1'], x['variable_2']])), axis=1
        )
        top_corr = top_corr.drop_duplicates(subset='pair')
        top_corr = top_corr.sort_values('abs_correlation', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📈 Correlaciones Positivas Más Fuertes")
            positive = top_corr[top_corr['correlation'] > 0].head(5)
            if positive.empty:
                st.info("No se encontraron correlaciones positivas significativas")
            else:
                for _, r in positive.iterrows():
                    strength = "🔴" if r['abs_correlation'] > 0.7 else "🟡" if r['abs_correlation'] > 0.4 else "🟢"
                    st.write(f"{strength} **{r['variable_1']}** ↔ **{r['variable_2']}**: `{r['correlation']:.3f}`")
        
        with col2:
            st.markdown("#### 📉 Correlaciones Negativas Más Fuertes")
            negative = top_corr[top_corr['correlation'] < 0].head(5)
            if negative.empty:
                st.info("No se encontraron correlaciones negativas significativas")
            else:
                for _, r in negative.iterrows():
                    strength = "🔴" if r['abs_correlation'] > 0.7 else "🟡" if r['abs_correlation'] > 0.4 else "🟢"
                    st.write(f"{strength} **{r['variable_1']}** ↔ **{r['variable_2']}**: `{r['correlation']:.3f}`")
        
        # Tabla completa
        with st.expander("📋 Tabla Completa de Correlaciones"):
            display_corr = top_corr[['variable_1', 'variable_2', 'correlation', 'abs_correlation']].copy()
            display_corr.columns = ['Variable 1', 'Variable 2', 'Correlación', 'Correlación Abs.']
            display_corr = display_corr.round(4)
            st.dataframe(display_corr, hide_index=True, use_container_width=True)
    
    else:
        st.error("❌ No se pudo cargar correlation_matrix.parquet")


# =============================================================================
# TAB 3: TEST CHI-CUADRADO
# =============================================================================
with tab3:
    st.header("🧪 Test de Independencia Chi-Cuadrado")
    
    chi_df = data_source.load_parquet('chi_square_tests.parquet')
    
    if chi_df is not None and not chi_df.empty:
        show_data_info(chi_df, "Dataset de Tests Chi-Cuadrado")
        
        # Guía de interpretación
        st.subheader("📖 ¿Qué es el Test Chi-Cuadrado?")
        
        st.info("""
        El **test Chi-cuadrado de independencia** evalúa si existe una relación estadísticamente 
        significativa entre dos variables categóricas.
        
        **Hipótesis:**
        - **H₀ (Nula):** Las variables son independientes (no hay relación)
        - **H₁ (Alternativa):** Las variables son dependientes (existe relación)
        
        **Regla de Decisión:**
        - **p-valor < 0.05:** Rechazamos H₀ → Las variables están relacionadas ✅
        - **p-valor ≥ 0.05:** No rechazamos H₀ → Las variables son independientes
        """)
        
        # Verificar formato de datos
        if 'test' in chi_df.columns and 'chi_square_statistic' in chi_df.columns:
            # Formato con resultados de test
            for idx, row in chi_df.iterrows():
                test_name = row['test']
                var1 = row['variable_1']
                var2 = row['variable_2']
                
                with st.expander(f"📊 {test_name}", expanded=True):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"**Variables:** `{var1}` × `{var2}`")
                        
                        mcol1, mcol2, mcol3 = st.columns(3)
                        with mcol1:
                            st.metric("Estadístico χ²", f"{row['chi_square_statistic']:.2f}")
                        with mcol2:
                            st.metric("p-valor", f"{row['p_value']:.4f}")
                        with mcol3:
                            st.metric("Grados de Libertad", int(row['degrees_of_freedom']))
                    
                    with col2:
                        is_sig = row['is_significant']
                        if is_sig:
                            st.success("### ✅ SIGNIFICATIVO")
                            st.write("Las variables **están relacionadas**")
                        else:
                            st.info("### ❌ NO SIGNIFICATIVO")
                            st.write("Las variables **son independientes**")
        
        else:
            # Formato de tabla de contingencia
            st.subheader("📊 Tabla de Contingencia: Estación × Categoría de Temperatura")
            
            if 'season' in chi_df.columns:
                temp_categories = [c for c in chi_df.columns if c != 'season']
                
                display_contingency = chi_df.copy()
                display_contingency = display_contingency.set_index('season')
                
                # Traducir estaciones
                season_translation = {
                    'Winter': 'Invierno', 'Spring': 'Primavera', 
                    'Summer': 'Verano', 'Fall': 'Otoño'
                }
                display_contingency.index = display_contingency.index.map(
                    lambda x: season_translation.get(x, x)
                )
                
                # Traducir categorías
                cat_translation = {
                    'Cold': 'Frío (<0°C)', 'Mild': 'Templado (0-15°C)',
                    'Warm': 'Cálido (15-25°C)', 'Hot': 'Caliente (>25°C)'
                }
                display_contingency.columns = [cat_translation.get(c, c) for c in display_contingency.columns]
                
                st.markdown("#### Frecuencias Observadas")
                st.dataframe(display_contingency, use_container_width=True)
                
                # Mapa de calor
                fig = px.imshow(
                    display_contingency.values,
                    labels=dict(x="Categoría de Temperatura", y="Estación", color="Frecuencia"),
                    x=list(display_contingency.columns),
                    y=list(display_contingency.index),
                    color_continuous_scale='Blues',
                    title="Distribución de Temperaturas por Estación",
                    aspect="auto",
                    text_auto=True
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Calcular chi-cuadrado
                try:
                    from scipy import stats
                    
                    contingency_array = chi_df.set_index('season').values.astype(float)
                    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_array)
                    
                    st.subheader("🧪 Resultado del Test Chi-Cuadrado")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Estadístico χ²", f"{chi2:.2f}")
                    with col2:
                        st.metric("p-valor", f"{p_value:.2e}" if p_value < 0.0001 else f"{p_value:.4f}")
                    with col3:
                        st.metric("Grados de Libertad", int(dof))
                    with col4:
                        is_significant = p_value < 0.05
                        if is_significant:
                            st.metric("Resultado", "✅ Significativo")
                        else:
                            st.metric("Resultado", "❌ No Significativo")
                    
                    # Interpretación
                    st.subheader("📝 Interpretación")
                    
                    if is_significant:
                        st.success(f"""
                        ### ✅ Las variables están RELACIONADAS
                        
                        **p-valor = {p_value:.2e}** (< 0.05)
                        
                        Existe una relación estadísticamente significativa entre la **Estación del año** 
                        y la **Categoría de temperatura**. Esto significa que la distribución de temperaturas 
                        (frío/templado/cálido/caliente) varía significativamente según la estación.
                        
                        **Conclusión:** Es esperable para datos climáticos, ya que las temperaturas 
                        están naturalmente asociadas a las estaciones.
                        """)
                    else:
                        st.info(f"""
                        ### ❌ Las variables son INDEPENDIENTES
                        
                        **p-valor = {p_value:.4f}** (≥ 0.05)
                        
                        No existe una relación estadísticamente significativa entre las variables.
                        """)
                    
                    # Frecuencias esperadas
                    with st.expander("📊 Frecuencias Esperadas (bajo hipótesis de independencia)"):
                        expected_df = pd.DataFrame(
                            expected,
                            index=display_contingency.index,
                            columns=display_contingency.columns
                        ).round(1)
                        st.dataframe(expected_df, use_container_width=True)
                        st.caption("Si las variables fueran independientes, estas serían las frecuencias esperadas.")
                
                except ImportError:
                    st.warning("⚠️ scipy no está disponible para calcular el estadístico chi-cuadrado")
                except Exception as e:
                    st.warning(f"⚠️ No se pudo calcular el test chi-cuadrado: {e}")
            else:
                st.dataframe(chi_df, use_container_width=True)
    
    else:
        st.error("❌ No se pudo cargar chi_square_tests.parquet")
