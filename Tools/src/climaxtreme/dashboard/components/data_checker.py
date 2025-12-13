"""
Componente reutilizable para verificación de datos sintéticos.

Este módulo proporciona un widget que se puede incluir en cualquier página
que necesite datos sintéticos, verificando automáticamente su disponibilidad
en HDFS y ofreciendo opciones al usuario.
"""

import streamlit as st
import pandas as pd
from typing import Optional, Tuple, Callable
from enum import Enum


class UserAction(Enum):
    """Acción seleccionada por el usuario."""
    NONE = "none"
    USE_EXISTING = "use_existing"
    REGENERATE = "regenerate"
    GENERATE = "generate"
    CONTINUE_WITHOUT = "continue_without"


def check_synthetic_data_availability(
    page_name: str = "esta página",
    required_dataset: str = "synthetic_hourly.parquet",
    min_records: int = 1000,
    min_cities: int = 10,
    show_details: bool = True
) -> Tuple[Optional[pd.DataFrame], UserAction]:
    """
    Verifica disponibilidad de datos sintéticos y muestra UI apropiada.
    
    Esta función debe llamarse al inicio de cada página que necesite
    datos sintéticos. Maneja todo el flujo de:
    1. Verificar conexión HDFS
    2. Verificar existencia de datos
    3. Verificar suficiencia
    4. Mostrar opciones al usuario
    5. Cargar datos si están disponibles
    
    Args:
        page_name: Nombre de la página para mensajes
        required_dataset: Dataset necesario
        min_records: Mínimo de registros requeridos
        min_cities: Mínimo de ciudades requeridas
        show_details: Si mostrar detalles de los datos
        
    Returns:
        Tuple de (DataFrame o None, UserAction)
    """
    try:
        from climaxtreme.dashboard.synthetic_manager import (
            SyntheticDataManager,
            DataStatus,
            render_data_status_card
        )
    except ImportError:
        st.error("❌ Error importando SyntheticDataManager")
        return None, UserAction.NONE
    
    # Inicializar manager
    manager = SyntheticDataManager()
    
    # Verificar conexión
    if not manager.check_hdfs_connection():
        st.error("""
        ❌ **No se puede conectar a HDFS**
        
        Verifica que el cluster de Docker esté activo:
        ```bash
        docker-compose up -d
        ```
        """)
        
        if st.button("🔄 Reintentar conexión"):
            st.rerun()
        
        return None, UserAction.NONE
    
    # Obtener estado
    with st.spinner("Verificando datos en HDFS..."):
        status = manager.get_hdfs_status()
    
    # Obtener info del dataset específico
    dataset_info = status.datasets.get(required_dataset)
    
    # CASO 1: Datos disponibles y suficientes
    if dataset_info and dataset_info.exists and dataset_info.record_count >= min_records:
        if show_details:
            with st.expander("📊 Estado de datos sintéticos", expanded=False):
                render_data_status_card(status)
                
                st.markdown(f"""
                **Dataset:** `{required_dataset}`
                - Registros: **{dataset_info.record_count:,}**
                - Ciudades: **{dataset_info.n_cities}**
                - Esquema válido: {'✅' if dataset_info.schema_valid else '❌'}
                """)
        
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            use_existing = st.button(
                "✅ Usar datos existentes",
                type="primary",
                use_container_width=True,
                key=f"use_existing_{page_name}"
            )
        
        with col2:
            regenerate = st.button(
                "🔄 Regenerar datos",
                use_container_width=True,
                key=f"regenerate_{page_name}"
            )
        
        with col3:
            st.markdown("")  # Espaciador
        
        if regenerate:
            st.info("👉 Redirigiendo al **Streaming Hub** para regenerar datos...")
            st.switch_page("pages/6_🌊_Streaming_Hub.py")
            return None, UserAction.REGENERATE
        
        if use_existing or st.session_state.get(f'auto_load_{page_name}', False):
            # Cargar datos
            with st.spinner(f"Cargando {required_dataset}..."):
                df = manager.load_dataset(required_dataset)
            
            if df is not None:
                st.session_state[f'auto_load_{page_name}'] = True
                return df, UserAction.USE_EXISTING
            else:
                st.error("Error cargando datos")
                return None, UserAction.NONE
        
        return None, UserAction.NONE
    
    # CASO 2: Datos insuficientes
    elif dataset_info and dataset_info.exists and dataset_info.record_count < min_records:
        st.warning(f"""
        ⚠️ **Datos insuficientes para {page_name}**
        
        - Registros disponibles: **{dataset_info.record_count:,}**
        - Registros requeridos: **{min_records:,}**
        - Ciudades disponibles: **{dataset_info.n_cities}**
        - Ciudades requeridas: **{min_cities}**
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("➕ Generar más datos", type="primary", 
                        use_container_width=True, key=f"gen_more_{page_name}"):
                st.switch_page("pages/6_🌊_Streaming_Hub.py")
                return None, UserAction.GENERATE
        
        with col2:
            if st.button("✅ Usar lo disponible", use_container_width=True,
                        key=f"use_partial_{page_name}"):
                with st.spinner("Cargando datos disponibles..."):
                    df = manager.load_dataset(required_dataset)
                if df is not None:
                    return df, UserAction.USE_EXISTING
        
        return None, UserAction.NONE
    
    # CASO 3: No hay datos
    else:
        st.error(f"""
        ❌ **No se encontraron datos sintéticos**
        
        Se requiere el dataset `{required_dataset}` para {page_name}.
        
        Puedes generar los datos desde el **Streaming Hub**.
        """)
        
        if st.button("🚀 Ir al Streaming Hub", type="primary", 
                    use_container_width=True, key=f"goto_hub_{page_name}"):
            st.switch_page("pages/6_🌊_Streaming_Hub.py")
        
        return None, UserAction.GENERATE


def synthetic_data_required(
    page_name: str = "esta página",
    required_dataset: str = "synthetic_hourly.parquet",
    min_records: int = 1000
) -> Callable:
    """
    Decorador para páginas que requieren datos sintéticos.
    
    Uso:
        @synthetic_data_required("Climate Heatmaps")
        def render_page(df: pd.DataFrame):
            # df ya está cargado y verificado
            st.write(df.head())
    
    Args:
        page_name: Nombre de la página
        required_dataset: Dataset requerido
        min_records: Mínimo de registros
        
    Returns:
        Decorador que wrappea la función
    """
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            df, action = check_synthetic_data_availability(
                page_name=page_name,
                required_dataset=required_dataset,
                min_records=min_records
            )
            
            if df is not None:
                return func(df, *args, **kwargs)
            else:
                # No mostrar nada más si no hay datos
                st.stop()
        
        return wrapper
    return decorator


def render_data_source_selector(
    available_datasets: list = None,
    default_dataset: str = "synthetic_hourly.parquet"
) -> Tuple[str, Optional[pd.DataFrame]]:
    """
    Renderiza un selector de fuente de datos.
    
    Permite al usuario elegir entre diferentes datasets disponibles
    y muestra información sobre cada uno.
    
    Args:
        available_datasets: Lista de datasets disponibles
        default_dataset: Dataset por defecto
        
    Returns:
        Tuple de (dataset seleccionado, DataFrame o None)
    """
    try:
        from climaxtreme.dashboard.synthetic_manager import SyntheticDataManager
    except ImportError:
        return default_dataset, None
    
    if available_datasets is None:
        available_datasets = [
            "synthetic_hourly.parquet",
            "synthetic_alerts.parquet",
            "synthetic_storms.parquet",
            "synthetic_events.parquet"
        ]
    
    manager = SyntheticDataManager()
    status = manager.get_hdfs_status()
    
    # Filtrar solo datasets que existen
    existing_datasets = [
        d for d in available_datasets 
        if d in status.datasets and status.datasets[d].exists
    ]
    
    if not existing_datasets:
        st.warning("No hay datasets sintéticos disponibles")
        return default_dataset, None
    
    # Selector
    selected = st.selectbox(
        "📁 Seleccionar dataset",
        options=existing_datasets,
        index=0 if default_dataset not in existing_datasets else existing_datasets.index(default_dataset),
        format_func=lambda x: f"{x} ({status.datasets[x].record_count:,} registros)" if x in status.datasets else x
    )
    
    # Cargar si se solicita
    if st.button("📥 Cargar dataset", key="load_selected_dataset"):
        with st.spinner(f"Cargando {selected}..."):
            df = manager.load_dataset(selected)
        return selected, df
    
    return selected, None


def show_hdfs_connection_status():
    """Muestra el estado de conexión a HDFS en el sidebar."""
    try:
        from climaxtreme.dashboard.synthetic_manager import SyntheticDataManager
        
        manager = SyntheticDataManager()
        connected = manager.check_hdfs_connection()
        
        if connected:
            st.sidebar.success("🟢 HDFS Conectado")
        else:
            st.sidebar.error("🔴 HDFS Desconectado")
            
    except Exception as e:
        st.sidebar.warning(f"⚠️ Error verificando HDFS: {e}")
