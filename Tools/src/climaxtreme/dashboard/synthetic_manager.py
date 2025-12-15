"""
SyntheticDataManager - Gestor centralizado de datos sintéticos.

Este módulo proporciona:
- Verificación de datos sintéticos en HDFS
- Verificación de suficiencia de datos para ML/análisis
- Gestión de metadatos de generación
- Interfaz unificada para todas las páginas del dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import logging
import json
import subprocess
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

class DataStatus(Enum):
    """Estado de los datos sintéticos."""
    NOT_FOUND = "not_found"
    INSUFFICIENT = "insufficient"
    AVAILABLE = "available"
    GENERATING = "generating"
    ERROR = "error"


@dataclass
class DatasetInfo:
    """Información sobre un dataset sintético."""
    name: str
    path: str
    exists: bool = False
    record_count: int = 0
    n_cities: int = 0
    n_countries: int = 0
    date_range: Tuple[str, str] = ("", "")
    columns: List[str] = field(default_factory=list)
    size_mb: float = 0.0
    last_modified: str = ""
    schema_valid: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass 
class GenerationMetadata:
    """Metadatos de una generación de datos."""
    generation_id: str
    timestamp: str
    config: Dict[str, Any]
    n_records_generated: int
    n_cities: int
    duration_seconds: float
    status: str
    datasets_created: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HDFSStatus:
    """Estado completo de HDFS para datos sintéticos."""
    connected: bool = False
    synthetic_path_exists: bool = False
    datasets: Dict[str, DatasetInfo] = field(default_factory=dict)
    total_records: int = 0
    total_size_mb: float = 0.0
    last_generation: Optional[GenerationMetadata] = None
    status: DataStatus = DataStatus.NOT_FOUND
    message: str = ""


# ============================================================================
# Configuración de Datasets Esperados
# ============================================================================

EXPECTED_DATASETS = {
    'synthetic_hourly.parquet': {
        'description': 'Datos horarios sintéticos de clima',
        'min_records': 10000,
        'required_columns': ['City', 'Country', 'timestamp', 'temperature_hourly', 
                            'rain_mm', 'wind_speed_kmh', 'humidity_pct'],
        'priority': 1  # Crítico para la mayoría de visualizaciones
    },
    'synthetic_storms.parquet': {
        'description': 'Datos de seguimiento de tormentas',
        'min_records': 100,
        'required_columns': ['storm_id', 'timestamp', 'latitude', 'longitude', 
                            'wind_speed_kmh', 'category'],
        'priority': 2
    },
    'synthetic_alerts.parquet': {
        'description': 'Alertas meteorológicas sintéticas',
        'min_records': 500,
        'required_columns': ['alert_type', 'alert_level', 'timestamp', 'City'],
        'priority': 2
    },
    'synthetic_events.parquet': {
        'description': 'Eventos climáticos extremos',
        'min_records': 200,
        'required_columns': ['event_type', 'event_intensity', 'timestamp'],
        'priority': 3
    }
}

# Requisitos mínimos para diferentes tipos de análisis
MIN_REQUIREMENTS = {
    'visualization': {'records': 1000, 'cities': 10},
    'statistics': {'records': 5000, 'cities': 20},
    'ml_training': {'records': 50000, 'cities': 50},
    'streaming_demo': {'records': 10000, 'cities': 30}
}


# ============================================================================
# SyntheticDataManager
# ============================================================================

class SyntheticDataManager:
    """
    Gestor centralizado para datos sintéticos.
    
    Proporciona una interfaz unificada para:
    - Verificar disponibilidad de datos en HDFS
    - Evaluar suficiencia para diferentes tipos de análisis
    - Cargar datos de forma eficiente
    - Gestionar metadatos de generación
    """
    
    def __init__(self):
        """Inicializa el gestor con configuración de sesión."""
        self.hdfs_host = st.session_state.get('hdfs_host', 'climaxtreme-namenode')
        self.hdfs_port = st.session_state.get('hdfs_port', 9000)
        self.synthetic_base_path = "/data/climaxtreme/synthetic"
        self.streaming_path = "/data/climaxtreme/streaming"
        self._reader = None
        
    @property
    def reader(self):
        """Lazy load del HDFSReader."""
        if self._reader is None:
            try:
                from climaxtreme.utils.hdfs_reader import HDFSReader
                self._reader = HDFSReader(self.hdfs_host, self.hdfs_port)
            except ImportError:
                logger.warning("HDFSReader no disponible")
        return self._reader
    
    # ========================================================================
    # Verificación de HDFS
    # ========================================================================
    
    def check_hdfs_connection(self) -> bool:
        """Verifica la conexión a HDFS."""
        try:
            import requests
            url = f"http://{self.hdfs_host}:9870/webhdfs/v1/?op=GETFILESTATUS"
            response = requests.get(url, timeout=5)
            return response.status_code in [200, 404]  # 404 es OK, significa que conectó
        except Exception as e:
            logger.error(f"Error conectando a HDFS: {e}")
            return False
    
    def check_path_exists(self, path: str) -> bool:
        """Verifica si un path existe en HDFS."""
        try:
            import requests
            url = f"http://{self.hdfs_host}:9870/webhdfs/v1{path}?op=GETFILESTATUS"
            response = requests.get(url, timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def list_hdfs_directory(self, path: str) -> List[Dict[str, Any]]:
        """Lista contenido de un directorio HDFS."""
        try:
            import requests
            url = f"http://{self.hdfs_host}:9870/webhdfs/v1{path}?op=LISTSTATUS"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data.get('FileStatuses', {}).get('FileStatus', [])
            return []
        except Exception as e:
            logger.error(f"Error listando directorio HDFS: {e}")
            return []
    
    def get_file_info(self, path: str) -> Optional[Dict[str, Any]]:
        """Obtiene información de un archivo/directorio en HDFS."""
        try:
            import requests
            url = f"http://{self.hdfs_host}:9870/webhdfs/v1{path}?op=GETFILESTATUS"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return response.json().get('FileStatus', {})
            return None
        except Exception:
            return None
    
    # ========================================================================
    # Verificación de Datasets
    # ========================================================================
    
    def get_dataset_info(self, dataset_name: str) -> DatasetInfo:
        """
        Obtiene información detallada de un dataset.
        
        Args:
            dataset_name: Nombre del dataset (e.g., 'synthetic_hourly.parquet')
            
        Returns:
            DatasetInfo con toda la información disponible
        """
        path = f"{self.synthetic_base_path}/{dataset_name}"
        info = DatasetInfo(name=dataset_name, path=path)
        
        # Verificar existencia
        if not self.check_path_exists(path):
            return info
        
        info.exists = True
        
        # Obtener tamaño
        file_info = self.get_file_info(path)
        if file_info:
            info.size_mb = file_info.get('length', 0) / (1024 * 1024)
            mod_time = file_info.get('modificationTime', 0)
            if mod_time:
                info.last_modified = datetime.fromtimestamp(mod_time / 1000).isoformat()
        
        # Cargar muestra para obtener estadísticas
        try:
            df = self.load_dataset_sample(dataset_name, n_rows=10000)
            if df is not None and not df.empty:
                info.record_count = self._get_full_count(dataset_name)
                info.columns = list(df.columns)
                
                # Ciudades y países
                if 'City' in df.columns:
                    info.n_cities = df['City'].nunique()
                if 'Country' in df.columns:
                    info.n_countries = df['Country'].nunique()
                
                # Rango de fechas
                date_cols = ['timestamp', 'dt', 'date', 'forecast_timestamp']
                for col in date_cols:
                    if col in df.columns:
                        dates = pd.to_datetime(df[col], errors='coerce')
                        info.date_range = (
                            dates.min().isoformat() if pd.notna(dates.min()) else "",
                            dates.max().isoformat() if pd.notna(dates.max()) else ""
                        )
                        break
                
                # Validar esquema
                expected = EXPECTED_DATASETS.get(dataset_name, {})
                required_cols = expected.get('required_columns', [])
                info.schema_valid = all(col in info.columns for col in required_cols)
                
        except Exception as e:
            logger.warning(f"Error obteniendo info de {dataset_name}: {e}")
        
        return info
    
    def _get_full_count(self, dataset_name: str) -> int:
        """Obtiene el conteo total de registros (puede ser costoso)."""
        try:
            # Ejecutar count en Spark via Docker
            script = f"""
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("CountRecords").getOrCreate()
df = spark.read.parquet("hdfs://climaxtreme-namenode:9000/data/climaxtreme/synthetic/{dataset_name}")
print(f"COUNT:{df.count()}")
spark.stop()
"""
            result = subprocess.run(
                ["docker", "exec", "climaxtreme-processor", "python", "-c", script],
                capture_output=True, text=True, timeout=60
            )
            
            for line in result.stdout.split('\n'):
                if line.startswith('COUNT:'):
                    return int(line.split(':')[1])
            
            return 0
        except Exception:
            return 0
    
    # ========================================================================
    # Estado Completo de HDFS
    # ========================================================================
    
    def get_hdfs_status(self) -> HDFSStatus:
        """
        Obtiene el estado completo de los datos sintéticos en HDFS.
        
        Returns:
            HDFSStatus con toda la información
        """
        status = HDFSStatus()
        
        # Verificar conexión
        status.connected = self.check_hdfs_connection()
        if not status.connected:
            status.status = DataStatus.ERROR
            status.message = "No se puede conectar a HDFS"
            return status
        
        # Verificar path de sintéticos
        status.synthetic_path_exists = self.check_path_exists(self.synthetic_base_path)
        if not status.synthetic_path_exists:
            status.status = DataStatus.NOT_FOUND
            status.message = "El directorio de datos sintéticos no existe en HDFS"
            return status
        
        # Obtener info de cada dataset esperado
        for dataset_name in EXPECTED_DATASETS.keys():
            info = self.get_dataset_info(dataset_name)
            status.datasets[dataset_name] = info
            if info.exists:
                status.total_records += info.record_count
                status.total_size_mb += info.size_mb
        
        # Cargar metadatos de última generación
        status.last_generation = self.get_last_generation_metadata()
        
        # Determinar estado general
        if status.total_records == 0:
            status.status = DataStatus.NOT_FOUND
            status.message = "No hay datos sintéticos generados"
        elif not self.is_sufficient_for('visualization', status):
            status.status = DataStatus.INSUFFICIENT
            status.message = "Datos insuficientes para visualización básica"
        else:
            status.status = DataStatus.AVAILABLE
            status.message = f"Disponibles {status.total_records:,} registros"
        
        return status
    
    # ========================================================================
    # Verificación de Suficiencia
    # ========================================================================
    
    def is_sufficient_for(self, use_case: str, status: Optional[HDFSStatus] = None) -> bool:
        """
        Verifica si los datos son suficientes para un caso de uso.
        
        Args:
            use_case: 'visualization', 'statistics', 'ml_training', 'streaming_demo'
            status: HDFSStatus opcional (se obtiene si no se proporciona)
            
        Returns:
            True si los datos son suficientes
        """
        if status is None:
            status = self.get_hdfs_status()
        
        requirements = MIN_REQUIREMENTS.get(use_case, MIN_REQUIREMENTS['visualization'])
        
        # Verificar dataset principal
        hourly_info = status.datasets.get('synthetic_hourly.parquet')
        if hourly_info is None or not hourly_info.exists:
            return False
        
        return (
            hourly_info.record_count >= requirements['records'] and
            hourly_info.n_cities >= requirements['cities']
        )
    
    def get_sufficiency_report(self, status: Optional[HDFSStatus] = None) -> Dict[str, Any]:
        """
        Genera un reporte de suficiencia para todos los casos de uso.
        
        Returns:
            Diccionario con estado de cada caso de uso
        """
        if status is None:
            status = self.get_hdfs_status()
        
        report = {}
        for use_case, requirements in MIN_REQUIREMENTS.items():
            is_sufficient = self.is_sufficient_for(use_case, status)
            hourly = status.datasets.get('synthetic_hourly.parquet')
            
            report[use_case] = {
                'sufficient': is_sufficient,
                'required_records': requirements['records'],
                'required_cities': requirements['cities'],
                'current_records': hourly.record_count if hourly else 0,
                'current_cities': hourly.n_cities if hourly else 0,
                'progress_records': min(100, (hourly.record_count / requirements['records'] * 100) if hourly else 0),
                'progress_cities': min(100, (hourly.n_cities / requirements['cities'] * 100) if hourly else 0)
            }
        
        return report
    
    # ========================================================================
    # Carga de Datos
    # ========================================================================
    
    def load_dataset(self, dataset_name: str) -> Optional[pd.DataFrame]:
        """
        Carga un dataset completo desde HDFS.
        
        Args:
            dataset_name: Nombre del dataset
            
        Returns:
            DataFrame o None si hay error
        """
        try:
            path = f"{self.synthetic_base_path}/{dataset_name}"
            if self.reader:
                return self.reader.read_parquet(path)
            return None
        except Exception as e:
            logger.error(f"Error cargando {dataset_name}: {e}")
            return None
    
    def load_dataset_sample(self, dataset_name: str, n_rows: int = 1000) -> Optional[pd.DataFrame]:
        """
        Carga una muestra de un dataset (más eficiente).
        
        Args:
            dataset_name: Nombre del dataset
            n_rows: Número de filas a cargar
            
        Returns:
            DataFrame con muestra
        """
        try:
            df = self.load_dataset(dataset_name)
            if df is not None and len(df) > n_rows:
                return df.sample(n=n_rows, random_state=42)
            return df
        except Exception as e:
            logger.error(f"Error cargando muestra de {dataset_name}: {e}")
            return None
    
    @st.cache_data(ttl=60, show_spinner=False)
    def load_synthetic_hourly(_self, sample_size: Optional[int] = None) -> Optional[pd.DataFrame]:
        """
        Carga datos horarios sintéticos con caché.
        
        Args:
            sample_size: Si se especifica, carga solo una muestra
            
        Returns:
            DataFrame con datos horarios
        """
        df = _self.load_dataset('synthetic_hourly.parquet')
        if df is not None and sample_size and len(df) > sample_size:
            return df.sample(n=sample_size, random_state=42)
        return df
    
    # ========================================================================
    # Metadatos de Generación
    # ========================================================================
    
    def get_last_generation_metadata(self) -> Optional[GenerationMetadata]:
        """Obtiene metadatos de la última generación."""
        try:
            # Intentar leer archivo de metadatos desde HDFS
            import requests
            url = f"http://{self.hdfs_host}:9870/webhdfs/v1{self.synthetic_base_path}/generation_metadata.json?op=OPEN"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return GenerationMetadata(**data)
            return None
        except Exception as e:
            logger.debug(f"No se encontraron metadatos de generación: {e}")
            return None
    
    def save_generation_metadata(self, metadata: GenerationMetadata) -> bool:
        """Guarda metadatos de generación en HDFS."""
        try:
            # Guardar via Docker/Spark
            metadata_json = json.dumps(metadata.to_dict())
            script = f"""
import json
with open('/tmp/generation_metadata.json', 'w') as f:
    f.write('''{metadata_json}''')

# Copiar a HDFS usando hdfs dfs
import subprocess
subprocess.run(['hdfs', 'dfs', '-put', '-f', '/tmp/generation_metadata.json', 
                '{self.synthetic_base_path}/generation_metadata.json'])
"""
            result = subprocess.run(
                ["docker", "exec", "climaxtreme-processor", "python", "-c", script],
                capture_output=True, text=True, timeout=30
            )
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Error guardando metadatos: {e}")
            return False
    
    # ========================================================================
    # Gestión de Caché
    # ========================================================================
    
    def clear_cache(self):
        """Limpia el caché de datos."""
        st.cache_data.clear()
        self._reader = None
    
    def clear_synthetic_data(self) -> bool:
        """Elimina todos los datos sintéticos de HDFS."""
        try:
            result = subprocess.run(
                ["docker", "exec", "climaxtreme-processor", "hdfs", "dfs", "-rm", "-r", "-f",
                 f"{self.synthetic_base_path}/*"],
                capture_output=True, text=True, timeout=60
            )
            self.clear_cache()
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Error eliminando datos sintéticos: {e}")
            return False


# ============================================================================
# Componentes UI de Streamlit
# ============================================================================

def render_data_status_card(status: HDFSStatus):
    """
    Renderiza una tarjeta de estado de datos.
    
    Args:
        status: Estado de HDFS
    """
    # Determinar estilo según estado
    if status.status == DataStatus.AVAILABLE:
        icon = "✅"
        color = "green"
        bg_color = "#d4edda"
    elif status.status == DataStatus.INSUFFICIENT:
        icon = "⚠️"
        color = "orange"
        bg_color = "#fff3cd"
    elif status.status == DataStatus.GENERATING:
        icon = "🔄"
        color = "blue"
        bg_color = "#cce5ff"
    else:
        icon = "❌"
        color = "red"
        bg_color = "#f8d7da"
    
    st.markdown(f"""
    <div style='background-color:{bg_color}; padding:20px; border-radius:10px; margin:10px 0;'>
        <h3 style='color:{color}; margin:0;'>{icon} Estado de Datos Sintéticos</h3>
        <p style='margin:10px 0 0 0;'>{status.message}</p>
        <hr style='margin:10px 0;'>
        <div style='display:flex; gap:20px;'>
            <div><strong>Registros:</strong> {status.total_records:,}</div>
            <div><strong>Tamaño:</strong> {status.total_size_mb:.2f} MB</div>
            <div><strong>Conexión HDFS:</strong> {'✓' if status.connected else '✗'}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_dataset_table(status: HDFSStatus):
    """
    Renderiza tabla de datasets disponibles.
    
    Args:
        status: Estado de HDFS
    """
    rows = []
    for name, info in status.datasets.items():
        expected = EXPECTED_DATASETS.get(name, {})
        rows.append({
            'Dataset': name.replace('.parquet', ''),
            'Estado': '✅ Disponible' if info.exists else '❌ No encontrado',
            'Registros': f"{info.record_count:,}" if info.exists else '-',
            'Ciudades': info.n_cities if info.exists else '-',
            'Esquema': '✓' if info.schema_valid else '✗',
            'Tamaño': f"{info.size_mb:.1f} MB" if info.exists else '-',
            'Mínimo Req.': f"{expected.get('min_records', 0):,}"
        })
    
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_sufficiency_progress(report: Dict[str, Any]):
    """
    Renderiza barras de progreso de suficiencia.
    
    Args:
        report: Reporte de suficiencia
    """
    use_case_names = {
        'visualization': '📊 Visualización',
        'statistics': '📈 Estadísticas',
        'ml_training': '🤖 Entrenamiento ML',
        'streaming_demo': '🌊 Demo Streaming'
    }
    
    cols = st.columns(2)
    for i, (use_case, data) in enumerate(report.items()):
        with cols[i % 2]:
            name = use_case_names.get(use_case, use_case)
            
            if data['sufficient']:
                st.success(f"{name}: ✅ Suficiente")
            else:
                st.warning(f"{name}: ⚠️ Insuficiente")
            
            progress = min(data['progress_records'], data['progress_cities']) / 100
            st.progress(progress, text=f"Registros: {data['current_records']:,}/{data['required_records']:,} | Ciudades: {data['current_cities']}/{data['required_cities']}")


def render_data_options(status: HDFSStatus) -> str:
    """
    Renderiza opciones de datos y retorna la acción seleccionada.
    
    Args:
        status: Estado de HDFS
        
    Returns:
        'use_existing', 'regenerate', 'generate', o 'none'
    """
    if status.status == DataStatus.AVAILABLE:
        st.info(f"📊 **Datos disponibles:** {status.total_records:,} registros")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("✅ Usar datos existentes", type="primary", use_container_width=True):
                return 'use_existing'
        with col2:
            if st.button("🔄 Regenerar datos", use_container_width=True):
                return 'regenerate'
        with col3:
            if st.button("⚙️ Configurar generación", use_container_width=True):
                return 'configure'
                
    elif status.status == DataStatus.INSUFFICIENT:
        st.warning(f"⚠️ **Datos insuficientes:** {status.total_records:,} registros")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("➕ Generar más datos", type="primary", use_container_width=True):
                return 'generate'
        with col2:
            if st.button("✅ Usar lo disponible", use_container_width=True):
                return 'use_existing'
                
    else:
        st.error("❌ **No se encontraron datos sintéticos**")
        
        if st.button("🚀 Generar datos sintéticos", type="primary", use_container_width=True):
            return 'generate'
    
    return 'none'


def get_synthetic_data_or_redirect(
    manager: SyntheticDataManager,
    min_use_case: str = 'visualization',
    page_name: str = "esta página"
) -> Tuple[Optional[pd.DataFrame], bool]:
    """
    Función helper para páginas que necesitan datos sintéticos.
    
    Verifica disponibilidad, muestra opciones al usuario, y retorna datos o
    indica si debe redirigir al Streaming Hub.
    
    Args:
        manager: Instancia de SyntheticDataManager
        min_use_case: Caso de uso mínimo requerido
        page_name: Nombre de la página para mensajes
        
    Returns:
        Tuple de (DataFrame o None, should_redirect)
    """
    # Obtener estado
    with st.spinner("Verificando datos en HDFS..."):
        status = manager.get_hdfs_status()
    
    # Si hay error de conexión
    if not status.connected:
        st.error("❌ No se puede conectar a HDFS. Verifica que el cluster esté activo.")
        return None, False
    
    # Mostrar estado
    render_data_status_card(status)
    
    # Si no hay datos o son insuficientes
    if status.status in [DataStatus.NOT_FOUND, DataStatus.INSUFFICIENT]:
        st.markdown("---")
        st.markdown(f"### 🌊 Se requieren datos sintéticos para {page_name}")
        
        action = render_data_options(status)
        
        if action in ['generate', 'regenerate', 'configure']:
            st.info("👉 Serás redirigido al **Streaming Hub** para configurar la generación de datos.")
            return None, True
        elif action == 'use_existing' and status.total_records > 0:
            df = manager.load_synthetic_hourly()
            return df, False
        
        return None, False
    
    # Datos disponibles
    action = render_data_options(status)
    
    if action in ['regenerate', 'configure']:
        return None, True
    
    # Cargar datos
    with st.spinner("Cargando datos sintéticos..."):
        df = manager.load_synthetic_hourly()
    
    return df, False
