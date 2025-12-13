"""
Kafka Producer - Productor de datos sintéticos climáticos para Kafka.

Este módulo proporciona:
- Producción de datos a tópicos Kafka
- Serialización JSON de eventos climáticos
- Integración con Spark Structured Streaming
- Soporte para múltiples tópicos (weather, alerts, storms)
"""

import os
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, asdict, field
import threading

logger = logging.getLogger(__name__)

# ============================================================================
# Configuración
# ============================================================================

KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'climaxtreme-kafka:9092')

# Tópicos de Kafka
TOPICS = {
    'weather': 'climaxtreme-weather',
    'alerts': 'climaxtreme-alerts', 
    'storms': 'climaxtreme-storms',
    'predictions': 'climaxtreme-predictions',
    'progress': 'climaxtreme-progress'
}


@dataclass
class KafkaConfig:
    """Configuración del producer Kafka."""
    bootstrap_servers: str = KAFKA_BOOTSTRAP_SERVERS
    
    # Tópicos
    weather_topic: str = TOPICS['weather']
    alerts_topic: str = TOPICS['alerts']
    storms_topic: str = TOPICS['storms']
    predictions_topic: str = TOPICS['predictions']
    progress_topic: str = TOPICS['progress']
    
    # Configuración del producer
    batch_size: int = 16384
    linger_ms: int = 100
    buffer_memory: int = 33554432
    acks: str = '1'
    compression_type: str = 'gzip'
    
    # Reintentos
    retries: int = 3
    retry_backoff_ms: int = 100
    
    # Timeouts
    request_timeout_ms: int = 30000
    delivery_timeout_ms: int = 120000


@dataclass  
class WeatherEvent:
    """Evento de clima para Kafka."""
    city: str
    country: str
    latitude: float
    longitude: float
    timestamp: str
    temperature_c: float
    humidity_pct: float
    pressure_hpa: float
    wind_speed_kmh: float
    wind_direction: int
    rain_mm: float
    cloud_cover_pct: float
    uv_index: float
    climate_zone: str
    event_id: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())


@dataclass
class AlertEvent:
    """Evento de alerta para Kafka."""
    alert_id: str
    city: str
    country: str
    latitude: float
    longitude: float
    timestamp: str
    alert_type: str  # HEAT, COLD, STORM, FLOOD, WIND
    alert_level: str  # WATCH, WARNING, EMERGENCY
    temperature_c: float
    threshold_value: float
    description: str
    expires_at: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())


@dataclass
class StormEvent:
    """Evento de tormenta para Kafka."""
    storm_id: str
    storm_name: str
    timestamp: str
    latitude: float
    longitude: float
    category: int
    max_wind_kmh: float
    central_pressure_hpa: float
    movement_speed_kmh: float
    movement_direction: int
    affected_radius_km: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())


# ============================================================================
# Kafka Producer Class
# ============================================================================

class ClimateKafkaProducer:
    """
    Productor Kafka para eventos climáticos.
    
    Maneja la producción de eventos de clima, alertas y tormentas
    a los tópicos correspondientes de Kafka.
    """
    
    def __init__(self, config: Optional[KafkaConfig] = None):
        """
        Inicializar producer.
        
        Args:
            config: Configuración de Kafka
        """
        self.config = config or KafkaConfig()
        self._producer = None
        self._is_connected = False
        self._stats = {
            'messages_sent': 0,
            'messages_failed': 0,
            'bytes_sent': 0,
            'last_error': None
        }
        
        logger.info(f"KafkaProducer initialized (servers: {self.config.bootstrap_servers})")
    
    def connect(self) -> bool:
        """
        Conectar al broker Kafka.
        
        Returns:
            True si la conexión fue exitosa
        """
        try:
            from kafka import KafkaProducer
            from kafka.errors import NoBrokersAvailable
            
            self._producer = KafkaProducer(
                bootstrap_servers=self.config.bootstrap_servers.split(','),
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                acks=self.config.acks,
                retries=self.config.retries,
                batch_size=self.config.batch_size,
                linger_ms=self.config.linger_ms,
                buffer_memory=self.config.buffer_memory,
                compression_type=self.config.compression_type,
                request_timeout_ms=self.config.request_timeout_ms
            )
            
            self._is_connected = True
            logger.info("Connected to Kafka broker")
            return True
            
        except ImportError:
            logger.error("kafka-python not installed. Run: pip install kafka-python")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            self._stats['last_error'] = str(e)
            return False
    
    def disconnect(self) -> None:
        """Cerrar conexión con Kafka."""
        if self._producer:
            self._producer.flush()
            self._producer.close()
            self._producer = None
            self._is_connected = False
            logger.info("Disconnected from Kafka broker")
    
    def _ensure_connected(self) -> bool:
        """Asegurar que hay conexión activa."""
        if not self._is_connected:
            return self.connect()
        return True
    
    def send_weather_event(
        self,
        event: WeatherEvent,
        callback: Optional[Callable] = None
    ) -> bool:
        """
        Enviar evento de clima a Kafka.
        
        Args:
            event: Evento de clima
            callback: Callback opcional para confirmación
            
        Returns:
            True si se envió exitosamente
        """
        if not self._ensure_connected():
            return False
        
        try:
            key = f"{event.city}_{event.country}"
            future = self._producer.send(
                self.config.weather_topic,
                key=key,
                value=event.to_dict()
            )
            
            if callback:
                future.add_callback(callback)
            
            self._stats['messages_sent'] += 1
            self._stats['bytes_sent'] += len(event.to_json())
            return True
            
        except Exception as e:
            logger.error(f"Failed to send weather event: {e}")
            self._stats['messages_failed'] += 1
            self._stats['last_error'] = str(e)
            return False
    
    def send_alert_event(self, event: AlertEvent) -> bool:
        """Enviar evento de alerta a Kafka."""
        if not self._ensure_connected():
            return False
        
        try:
            key = f"{event.alert_type}_{event.alert_level}"
            self._producer.send(
                self.config.alerts_topic,
                key=key,
                value=event.to_dict()
            )
            self._stats['messages_sent'] += 1
            return True
            
        except Exception as e:
            logger.error(f"Failed to send alert event: {e}")
            self._stats['messages_failed'] += 1
            return False
    
    def send_storm_event(self, event: StormEvent) -> bool:
        """Enviar evento de tormenta a Kafka."""
        if not self._ensure_connected():
            return False
        
        try:
            self._producer.send(
                self.config.storms_topic,
                key=event.storm_id,
                value=event.to_dict()
            )
            self._stats['messages_sent'] += 1
            return True
            
        except Exception as e:
            logger.error(f"Failed to send storm event: {e}")
            self._stats['messages_failed'] += 1
            return False
    
    def send_batch(
        self,
        events: List[Dict[str, Any]],
        topic: str,
        key_field: Optional[str] = None
    ) -> Dict[str, int]:
        """
        Enviar batch de eventos a un tópico.
        
        Args:
            events: Lista de eventos (dicts)
            topic: Nombre del tópico
            key_field: Campo a usar como key
            
        Returns:
            Estadísticas del batch
        """
        if not self._ensure_connected():
            return {'sent': 0, 'failed': len(events)}
        
        sent = 0
        failed = 0
        
        for event in events:
            try:
                key = str(event.get(key_field, '')) if key_field else None
                self._producer.send(topic, key=key, value=event)
                sent += 1
            except Exception as e:
                failed += 1
                logger.warning(f"Failed to send event: {e}")
        
        # Flush para asegurar envío
        self._producer.flush()
        
        self._stats['messages_sent'] += sent
        self._stats['messages_failed'] += failed
        
        return {'sent': sent, 'failed': failed}
    
    def send_progress(
        self,
        progress_pct: float,
        records_generated: int,
        status: str,
        details: Optional[Dict] = None
    ) -> bool:
        """
        Enviar evento de progreso.
        
        Args:
            progress_pct: Porcentaje de progreso (0-100)
            records_generated: Registros generados
            status: Estado actual
            details: Detalles adicionales
        """
        if not self._ensure_connected():
            return False
        
        try:
            event = {
                'timestamp': datetime.now().isoformat(),
                'progress_pct': progress_pct,
                'records_generated': records_generated,
                'status': status,
                'details': details or {}
            }
            
            self._producer.send(
                self.config.progress_topic,
                key='progress',
                value=event
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to send progress: {e}")
            return False
    
    def flush(self) -> None:
        """Flush mensajes pendientes."""
        if self._producer:
            self._producer.flush()
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del producer."""
        return {
            **self._stats,
            'is_connected': self._is_connected,
            'bootstrap_servers': self.config.bootstrap_servers
        }
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()


# ============================================================================
# Streaming Producer con Kafka
# ============================================================================

class KafkaStreamingProducer:
    """
    Producer que genera datos sintéticos y los publica a Kafka.
    
    Combina la generación de datos con la publicación a Kafka,
    permitiendo consumo en tiempo real.
    """
    
    def __init__(
        self,
        kafka_config: Optional[KafkaConfig] = None,
        n_cities: int = 50,
        batch_size: int = 100,
        interval_seconds: float = 1.0
    ):
        """
        Inicializar streaming producer.
        
        Args:
            kafka_config: Configuración de Kafka
            n_cities: Número de ciudades a generar
            batch_size: Tamaño del batch por intervalo
            interval_seconds: Intervalo entre batches
        """
        self.kafka_config = kafka_config or KafkaConfig()
        self.n_cities = n_cities
        self.batch_size = batch_size
        self.interval_seconds = interval_seconds
        
        self._producer = ClimateKafkaProducer(self.kafka_config)
        self._running = False
        self._thread = None
        self._stats = {
            'batches_produced': 0,
            'total_records': 0,
            'start_time': None,
            'errors': []
        }
        
        # Ciudades de ejemplo
        self._cities = self._load_sample_cities()
    
    def _load_sample_cities(self) -> List[Dict[str, Any]]:
        """Cargar lista de ciudades de ejemplo."""
        import random
        
        # Ciudades globales representativas
        cities = [
            {"city": "New York", "country": "United States", "lat": 40.7128, "lon": -74.0060, "zone": "temperate"},
            {"city": "London", "country": "United Kingdom", "lat": 51.5074, "lon": -0.1278, "zone": "temperate"},
            {"city": "Tokyo", "country": "Japan", "lat": 35.6762, "lon": 139.6503, "zone": "temperate"},
            {"city": "Sydney", "country": "Australia", "lat": -33.8688, "lon": 151.2093, "zone": "temperate"},
            {"city": "Dubai", "country": "United Arab Emirates", "lat": 25.2048, "lon": 55.2708, "zone": "desert"},
            {"city": "Singapore", "country": "Singapore", "lat": 1.3521, "lon": 103.8198, "zone": "tropical"},
            {"city": "Mumbai", "country": "India", "lat": 19.0760, "lon": 72.8777, "zone": "tropical"},
            {"city": "São Paulo", "country": "Brazil", "lat": -23.5505, "lon": -46.6333, "zone": "tropical"},
            {"city": "Moscow", "country": "Russia", "lat": 55.7558, "lon": 37.6173, "zone": "continental"},
            {"city": "Cairo", "country": "Egypt", "lat": 30.0444, "lon": 31.2357, "zone": "desert"},
            {"city": "Paris", "country": "France", "lat": 48.8566, "lon": 2.3522, "zone": "temperate"},
            {"city": "Beijing", "country": "China", "lat": 39.9042, "lon": 116.4074, "zone": "continental"},
            {"city": "Los Angeles", "country": "United States", "lat": 34.0522, "lon": -118.2437, "zone": "mediterranean"},
            {"city": "Miami", "country": "United States", "lat": 25.7617, "lon": -80.1918, "zone": "tropical"},
            {"city": "Mexico City", "country": "Mexico", "lat": 19.4326, "lon": -99.1332, "zone": "temperate"},
            {"city": "Buenos Aires", "country": "Argentina", "lat": -34.6037, "lon": -58.3816, "zone": "temperate"},
            {"city": "Cape Town", "country": "South Africa", "lat": -33.9249, "lon": 18.4241, "zone": "mediterranean"},
            {"city": "Stockholm", "country": "Sweden", "lat": 59.3293, "lon": 18.0686, "zone": "continental"},
            {"city": "Bangkok", "country": "Thailand", "lat": 13.7563, "lon": 100.5018, "zone": "tropical"},
            {"city": "Toronto", "country": "Canada", "lat": 43.6532, "lon": -79.3832, "zone": "continental"},
        ]
        
        # Expandir con variaciones si necesitamos más ciudades
        while len(cities) < self.n_cities:
            base = random.choice(cities[:20])
            new_city = {
                **base,
                "city": f"{base['city']} Area {len(cities)}",
                "lat": base['lat'] + random.uniform(-0.5, 0.5),
                "lon": base['lon'] + random.uniform(-0.5, 0.5)
            }
            cities.append(new_city)
        
        return cities[:self.n_cities]
    
    def _generate_weather_event(
        self,
        city: Dict[str, Any],
        timestamp: datetime
    ) -> WeatherEvent:
        """Generar evento de clima para una ciudad."""
        import random
        import math
        
        # Temperatura base según zona climática
        zone_temps = {
            'tropical': (25, 35),
            'temperate': (5, 25),
            'continental': (-10, 30),
            'desert': (15, 45),
            'mediterranean': (10, 30),
            'polar': (-30, 5)
        }
        
        zone = city.get('zone', 'temperate')
        temp_min, temp_max = zone_temps.get(zone, (10, 25))
        
        # Ciclo diurno
        hour = timestamp.hour
        diurnal_factor = math.sin((hour - 6) * math.pi / 12)
        temp_range = temp_max - temp_min
        base_temp = temp_min + temp_range / 2
        temperature = base_temp + (temp_range / 4) * diurnal_factor
        temperature += random.gauss(0, 2)  # Ruido
        
        # Otras variables
        humidity = random.gauss(60, 20)
        humidity = max(10, min(100, humidity))
        
        pressure = random.gauss(1013, 10)
        
        wind_speed = abs(random.gauss(15, 10))
        wind_direction = random.randint(0, 359)
        
        rain = 0
        if random.random() < 0.2:  # 20% probabilidad de lluvia
            rain = random.expovariate(0.2)
        
        cloud_cover = random.gauss(50, 30)
        cloud_cover = max(0, min(100, cloud_cover))
        
        uv_index = max(0, 11 * math.sin(hour * math.pi / 12) * (1 - cloud_cover/200))
        
        return WeatherEvent(
            city=city['city'],
            country=city['country'],
            latitude=city['lat'],
            longitude=city['lon'],
            timestamp=timestamp.isoformat(),
            temperature_c=round(temperature, 1),
            humidity_pct=round(humidity, 1),
            pressure_hpa=round(pressure, 1),
            wind_speed_kmh=round(wind_speed, 1),
            wind_direction=wind_direction,
            rain_mm=round(rain, 2),
            cloud_cover_pct=round(cloud_cover, 1),
            uv_index=round(uv_index, 1),
            climate_zone=zone,
            event_id=f"{city['city']}_{timestamp.strftime('%Y%m%d%H%M%S')}"
        )
    
    def _check_alert(
        self,
        weather: WeatherEvent,
        timestamp: datetime
    ) -> Optional[AlertEvent]:
        """Verificar si el evento genera una alerta."""
        import random
        import uuid
        
        alert = None
        
        # Alerta de calor
        if weather.temperature_c >= 35:
            level = 'WARNING' if weather.temperature_c < 40 else 'EMERGENCY'
            alert = AlertEvent(
                alert_id=str(uuid.uuid4())[:8],
                city=weather.city,
                country=weather.country,
                latitude=weather.latitude,
                longitude=weather.longitude,
                timestamp=timestamp.isoformat(),
                alert_type='HEAT',
                alert_level=level,
                temperature_c=weather.temperature_c,
                threshold_value=35.0 if level == 'WARNING' else 40.0,
                description=f"Extreme heat alert: {weather.temperature_c}°C",
                expires_at=(timestamp + timedelta(hours=6)).isoformat()
            )
        
        # Alerta de frío
        elif weather.temperature_c <= -10:
            level = 'WARNING' if weather.temperature_c > -20 else 'EMERGENCY'
            alert = AlertEvent(
                alert_id=str(uuid.uuid4())[:8],
                city=weather.city,
                country=weather.country,
                latitude=weather.latitude,
                longitude=weather.longitude,
                timestamp=timestamp.isoformat(),
                alert_type='COLD',
                alert_level=level,
                temperature_c=weather.temperature_c,
                threshold_value=-10.0 if level == 'WARNING' else -20.0,
                description=f"Extreme cold alert: {weather.temperature_c}°C",
                expires_at=(timestamp + timedelta(hours=6)).isoformat()
            )
        
        # Alerta de viento
        elif weather.wind_speed_kmh >= 60:
            level = 'WARNING' if weather.wind_speed_kmh < 90 else 'EMERGENCY'
            alert = AlertEvent(
                alert_id=str(uuid.uuid4())[:8],
                city=weather.city,
                country=weather.country,
                latitude=weather.latitude,
                longitude=weather.longitude,
                timestamp=timestamp.isoformat(),
                alert_type='WIND',
                alert_level=level,
                temperature_c=weather.temperature_c,
                threshold_value=60.0 if level == 'WARNING' else 90.0,
                description=f"High wind alert: {weather.wind_speed_kmh} km/h",
                expires_at=(timestamp + timedelta(hours=3)).isoformat()
            )
        
        return alert
    
    def produce_batch(self, current_time: Optional[datetime] = None) -> Dict[str, int]:
        """
        Producir un batch de eventos.
        
        Args:
            current_time: Timestamp para los eventos
            
        Returns:
            Estadísticas del batch
        """
        import random
        
        current_time = current_time or datetime.now()
        
        stats = {
            'weather_events': 0,
            'alert_events': 0,
            'errors': 0
        }
        
        # Seleccionar ciudades para este batch
        cities_sample = random.sample(self._cities, min(self.batch_size, len(self._cities)))
        
        for city in cities_sample:
            try:
                # Generar evento de clima
                weather_event = self._generate_weather_event(city, current_time)
                
                if self._producer.send_weather_event(weather_event):
                    stats['weather_events'] += 1
                else:
                    stats['errors'] += 1
                
                # Verificar alertas
                alert = self._check_alert(weather_event, current_time)
                if alert:
                    if self._producer.send_alert_event(alert):
                        stats['alert_events'] += 1
                
            except Exception as e:
                logger.error(f"Error producing event for {city['city']}: {e}")
                stats['errors'] += 1
        
        # Flush
        self._producer.flush()
        
        self._stats['batches_produced'] += 1
        self._stats['total_records'] += stats['weather_events']
        
        return stats
    
    def _streaming_loop(self) -> None:
        """Loop principal de streaming."""
        logger.info("Starting Kafka streaming loop")
        
        while self._running:
            try:
                stats = self.produce_batch()
                
                # Enviar progreso
                self._producer.send_progress(
                    progress_pct=100.0,  # Streaming continuo
                    records_generated=self._stats['total_records'],
                    status='streaming',
                    details=stats
                )
                
                time.sleep(self.interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in streaming loop: {e}")
                self._stats['errors'].append(str(e))
                time.sleep(1)  # Esperar antes de reintentar
        
        logger.info("Kafka streaming loop stopped")
    
    def start_streaming(self) -> bool:
        """
        Iniciar streaming continuo.
        
        Returns:
            True si se inició correctamente
        """
        if self._running:
            logger.warning("Streaming already running")
            return False
        
        if not self._producer.connect():
            logger.error("Failed to connect to Kafka")
            return False
        
        self._running = True
        self._stats['start_time'] = datetime.now().isoformat()
        
        self._thread = threading.Thread(target=self._streaming_loop, daemon=True)
        self._thread.start()
        
        logger.info(f"Kafka streaming started ({self.n_cities} cities, {self.interval_seconds}s interval)")
        return True
    
    def stop_streaming(self) -> None:
        """Detener streaming."""
        self._running = False
        
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        
        self._producer.disconnect()
        logger.info("Kafka streaming stopped")
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del producer."""
        return {
            **self._stats,
            'is_running': self._running,
            'kafka_stats': self._producer.get_stats()
        }
    
    def __enter__(self):
        self.start_streaming()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_streaming()


# ============================================================================
# Utilidades
# ============================================================================

def create_kafka_topics(
    bootstrap_servers: str = KAFKA_BOOTSTRAP_SERVERS,
    num_partitions: int = 3,
    replication_factor: int = 1
) -> Dict[str, bool]:
    """
    Crear los tópicos de Kafka necesarios.
    
    Args:
        bootstrap_servers: Servidores Kafka
        num_partitions: Número de particiones por tópico
        replication_factor: Factor de replicación
        
    Returns:
        Diccionario con estado de creación por tópico
    """
    try:
        from kafka.admin import KafkaAdminClient, NewTopic
        from kafka.errors import TopicAlreadyExistsError
        
        admin = KafkaAdminClient(
            bootstrap_servers=bootstrap_servers.split(','),
            client_id='climaxtreme-admin'
        )
        
        results = {}
        
        for name, topic in TOPICS.items():
            try:
                new_topic = NewTopic(
                    name=topic,
                    num_partitions=num_partitions,
                    replication_factor=replication_factor
                )
                admin.create_topics([new_topic])
                results[name] = True
                logger.info(f"Created topic: {topic}")
                
            except TopicAlreadyExistsError:
                results[name] = True  # Ya existe
                logger.info(f"Topic already exists: {topic}")
                
            except Exception as e:
                results[name] = False
                logger.error(f"Failed to create topic {topic}: {e}")
        
        admin.close()
        return results
        
    except ImportError:
        logger.error("kafka-python not installed")
        return {name: False for name in TOPICS}
    except Exception as e:
        logger.error(f"Failed to create topics: {e}")
        return {name: False for name in TOPICS}


def check_kafka_connection(
    bootstrap_servers: str = KAFKA_BOOTSTRAP_SERVERS,
    timeout_ms: int = 5000
) -> Dict[str, Any]:
    """
    Verificar conexión con Kafka.
    
    Args:
        bootstrap_servers: Servidores Kafka
        timeout_ms: Timeout en milisegundos
        
    Returns:
        Estado de la conexión
    """
    try:
        from kafka import KafkaConsumer
        from kafka.errors import NoBrokersAvailable
        
        consumer = KafkaConsumer(
            bootstrap_servers=bootstrap_servers.split(','),
            consumer_timeout_ms=timeout_ms
        )
        
        # Obtener metadata
        topics = consumer.topics()
        
        consumer.close()
        
        return {
            'connected': True,
            'bootstrap_servers': bootstrap_servers,
            'topics_available': list(topics),
            'climaxtreme_topics': [t for t in topics if 'climaxtreme' in t]
        }
        
    except ImportError:
        return {
            'connected': False,
            'error': 'kafka-python not installed'
        }
    except Exception as e:
        return {
            'connected': False,
            'error': str(e)
        }
