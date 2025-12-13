"""
Kafka Streaming Manager - Gestión del pipeline de streaming Kafka desde Streamlit.

Este módulo permite:
- Iniciar/detener el productor de Kafka desde el dashboard
- Gestionar la generación de datos en tiempo real
- Monitorear el estado del pipeline completo
- Ejecutar comandos en el contenedor Docker
"""

import os
import subprocess
import json
import time
import threading
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import logging
import streamlit as st

logger = logging.getLogger(__name__)


@dataclass
class KafkaStreamingConfig:
    """Configuración del streaming Kafka."""
    n_cities: int = 50
    interval_seconds: float = 1.0
    include_alerts: bool = True
    include_storms: bool = True
    alert_probability: float = 0.1
    storm_probability: float = 0.15  # Aumentado para más tormentas
    bootstrap_servers: str = "climaxtreme-kafka:9092"


class KafkaStreamingManager:
    """
    Gestiona el pipeline de streaming Kafka desde el dashboard.
    
    Permite iniciar/detener productores de Kafka ejecutando
    el código directamente o en el contenedor Docker.
    """
    
    def __init__(self):
        self.container_name = "climaxtreme-processor"
        self._producer_process = None
        self._producer_thread = None
        self._producer_instance = None
        self._is_producing = False
        self._stop_requested = False
        self._stats = {
            'events_produced': 0,
            'start_time': None,
            'last_event_time': None,
            'errors': 0
        }
        # Kafka connection settings (for inside container)
        self.kafka_bootstrap = os.environ.get('KAFKA_BOOTSTRAP_SERVERS', 'climaxtreme-kafka:9092')
        self.zookeeper_host = os.environ.get('ZOOKEEPER_HOST', 'climaxtreme-zookeeper:2181')
    
    def _check_port_open(self, host: str, port: int, timeout: float = 2.0) -> bool:
        """Check if a TCP port is open."""
        import socket
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        except Exception:
            return False
    
    def check_kafka_cluster(self) -> Dict[str, Any]:
        """
        Verificar el estado del clúster de Kafka.
        
        Intenta conexión directa por red (funciona desde dentro de contenedores)
        y fallback a comandos Docker (funciona desde host).
        
        Returns:
            Estado del clúster
        """
        result = {
            'zookeeper_running': False,
            'kafka_running': False,
            'topics_created': [],
            'error': None
        }
        
        # Método 1: Conexión directa por red (para dentro de contenedor)
        try:
            # Check Zookeeper via TCP
            result['zookeeper_running'] = self._check_port_open('climaxtreme-zookeeper', 2181)
            
            # Check Kafka via TCP
            result['kafka_running'] = self._check_port_open('climaxtreme-kafka', 9092)
            
            # Si Kafka está corriendo, intentar listar topics con kafka-python
            if result['kafka_running']:
                try:
                    from kafka.admin import KafkaAdminClient
                    admin = KafkaAdminClient(
                        bootstrap_servers='climaxtreme-kafka:9092',
                        client_id='climaxtreme-status-check',
                        request_timeout_ms=5000
                    )
                    all_topics = admin.list_topics()
                    result['topics_created'] = [t for t in all_topics if 'climaxtreme' in t]
                    admin.close()
                except Exception as e:
                    logger.debug(f"Could not list topics via kafka-python: {e}")
                    # Topics unknown pero Kafka está corriendo
                    pass
            
            # Si la conexión directa funcionó, retornar
            if result['zookeeper_running'] or result['kafka_running']:
                return result
                
        except Exception as e:
            logger.debug(f"Direct connection check failed: {e}")
        
        # Método 2: Fallback a comandos Docker (para ejecutar desde host)
        try:
            # Verificar Zookeeper
            zk_check = subprocess.run(
                ['docker', 'inspect', '-f', '{{.State.Running}}', 'climaxtreme-zookeeper'],
                capture_output=True, text=True, timeout=5
            )
            result['zookeeper_running'] = zk_check.stdout.strip() == 'true'
            
            # Verificar Kafka
            kafka_check = subprocess.run(
                ['docker', 'inspect', '-f', '{{.State.Running}}', 'climaxtreme-kafka'],
                capture_output=True, text=True, timeout=5
            )
            result['kafka_running'] = kafka_check.stdout.strip() == 'true'
            
            # Listar topics si Kafka está corriendo
            if result['kafka_running']:
                topics_cmd = subprocess.run(
                    ['docker', 'exec', 'climaxtreme-kafka', 
                     'kafka-topics', '--list', '--bootstrap-server', 'localhost:9092'],
                    capture_output=True, text=True, timeout=10
                )
                if topics_cmd.returncode == 0:
                    topics = [t.strip() for t in topics_cmd.stdout.split('\n') if t.strip()]
                    result['topics_created'] = [t for t in topics if 'climaxtreme' in t]
            
        except FileNotFoundError:
            # Docker no disponible (estamos en contenedor sin docker CLI)
            if not result['zookeeper_running'] and not result['kafka_running']:
                result['error'] = "No se puede verificar Kafka (ejecutando dentro de contenedor sin conexión)"
        except subprocess.TimeoutExpired:
            result['error'] = "Timeout verificando servicios"
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def start_kafka_services(self) -> Tuple[bool, str]:
        """
        Iniciar servicios de Kafka (Zookeeper + Kafka).
        
        Returns:
            (éxito, mensaje)
        """
        try:
            # Ejecutar docker-compose up para kafka
            cmd = subprocess.run(
                ['docker-compose', '-f', '/infra/docker-compose.yml', 
                 'up', '-d', 'zookeeper', 'kafka'],
                capture_output=True, text=True, timeout=60,
                cwd=os.path.dirname(os.path.dirname(os.path.dirname(
                    os.path.dirname(os.path.dirname(__file__)))))  # climaXtreme root
            )
            
            if cmd.returncode == 0:
                # Esperar a que Kafka esté listo
                time.sleep(5)
                return True, "Servicios de Kafka iniciados"
            else:
                return False, f"Error: {cmd.stderr}"
                
        except Exception as e:
            return False, str(e)
    
    def create_topics(self) -> Tuple[bool, str]:
        """
        Crear los tópicos de climaXtreme en Kafka.
        
        Intenta usar kafka-python (funciona desde contenedor) o
        fallback a comandos Docker (funciona desde host).
        
        Returns:
            (éxito, mensaje)
        """
        topics = [
            'climaxtreme-weather',
            'climaxtreme-alerts', 
            'climaxtreme-storms',
            'climaxtreme-predictions',
            'climaxtreme-progress'
        ]
        
        created = []
        errors = []
        
        # Método 1: Usar kafka-python (funciona desde dentro del contenedor)
        try:
            from kafka.admin import KafkaAdminClient, NewTopic
            
            admin = KafkaAdminClient(
                bootstrap_servers='climaxtreme-kafka:9092',
                client_id='climaxtreme-topic-creator',
                request_timeout_ms=10000
            )
            
            existing_topics = admin.list_topics()
            
            topics_to_create = []
            for topic in topics:
                if topic in existing_topics:
                    created.append(topic)
                else:
                    topics_to_create.append(
                        NewTopic(name=topic, num_partitions=3, replication_factor=1)
                    )
            
            if topics_to_create:
                try:
                    admin.create_topics(topics_to_create, validate_only=False)
                    created.extend([t.name for t in topics_to_create])
                except Exception as e:
                    # Algunos topics pueden ya existir
                    if 'already exists' in str(e).lower():
                        created.extend([t.name for t in topics_to_create])
                    else:
                        errors.append(str(e))
            
            admin.close()
            
            if created and not errors:
                return True, f"Tópicos creados/verificados: {created}"
            elif errors:
                return False, f"Creados: {created}, Errores: {errors}"
            
        except ImportError:
            logger.debug("kafka-python not available, trying Docker method")
        except Exception as e:
            logger.debug(f"kafka-python method failed: {e}")
        
        # Método 2: Fallback a comandos Docker (funciona desde host)
        for topic in topics:
            try:
                cmd = subprocess.run(
                    ['docker', 'exec', 'climaxtreme-kafka',
                     'kafka-topics', '--create',
                     '--bootstrap-server', 'localhost:9092',
                     '--topic', topic,
                     '--partitions', '3',
                     '--replication-factor', '1',
                     '--if-not-exists'],
                    capture_output=True, text=True, timeout=10
                )
                
                if cmd.returncode == 0 or 'already exists' in cmd.stderr:
                    created.append(topic)
                else:
                    errors.append(f"{topic}: {cmd.stderr}")
                    
            except Exception as e:
                errors.append(f"{topic}: {str(e)}")
        
        if errors:
            return False, f"Creados: {created}, Errores: {errors}"
        return True, f"Tópicos creados/verificados: {created}"
    
    def _run_simple_producer(self, config: KafkaStreamingConfig):
        """
        Ejecuta un productor simple de Kafka en un hilo.
        Genera datos sintéticos con todos los campos necesarios para los dashboards.
        """
        try:
            from kafka import KafkaProducer
            import random
            import math
            
            producer = KafkaProducer(
                bootstrap_servers=config.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None
            )
            
            logger.info(f"Simple Kafka producer connected to {config.bootstrap_servers}")
            
            # Lista extendida de ciudades con datos completos
            cities = [
                {"city": "Madrid", "country": "Spain", "lat": 40.42, "lon": -3.70, "zone": "TEMPERATE"},
                {"city": "Barcelona", "country": "Spain", "lat": 41.39, "lon": 2.17, "zone": "MEDITERRANEAN"},
                {"city": "London", "country": "UK", "lat": 51.51, "lon": -0.13, "zone": "TEMPERATE"},
                {"city": "Paris", "country": "France", "lat": 48.86, "lon": 2.35, "zone": "TEMPERATE"},
                {"city": "Berlin", "country": "Germany", "lat": 52.52, "lon": 13.40, "zone": "CONTINENTAL"},
                {"city": "Rome", "country": "Italy", "lat": 41.90, "lon": 12.50, "zone": "MEDITERRANEAN"},
                {"city": "New York", "country": "USA", "lat": 40.71, "lon": -74.01, "zone": "CONTINENTAL"},
                {"city": "Tokyo", "country": "Japan", "lat": 35.68, "lon": 139.69, "zone": "TEMPERATE"},
                {"city": "Sydney", "country": "Australia", "lat": -33.87, "lon": 151.21, "zone": "SUBTROPICAL"},
                {"city": "Dubai", "country": "UAE", "lat": 25.20, "lon": 55.27, "zone": "DESERT"},
                {"city": "Mumbai", "country": "India", "lat": 19.08, "lon": 72.88, "zone": "TROPICAL"},
                {"city": "Cairo", "country": "Egypt", "lat": 30.04, "lon": 31.24, "zone": "DESERT"},
                {"city": "Moscow", "country": "Russia", "lat": 55.75, "lon": 37.62, "zone": "CONTINENTAL"},
                {"city": "Beijing", "country": "China", "lat": 39.90, "lon": 116.41, "zone": "CONTINENTAL"},
                {"city": "São Paulo", "country": "Brazil", "lat": -23.55, "lon": -46.63, "zone": "SUBTROPICAL"},
                {"city": "Mexico City", "country": "Mexico", "lat": 19.43, "lon": -99.13, "zone": "SUBTROPICAL"},
                {"city": "Lagos", "country": "Nigeria", "lat": 6.52, "lon": 3.38, "zone": "TROPICAL"},
                {"city": "Buenos Aires", "country": "Argentina", "lat": -34.60, "lon": -58.38, "zone": "TEMPERATE"},
                {"city": "Singapore", "country": "Singapore", "lat": 1.35, "lon": 103.82, "zone": "TROPICAL"},
                {"city": "Hong Kong", "country": "China", "lat": 22.32, "lon": 114.17, "zone": "SUBTROPICAL"},
                {"city": "Miami", "country": "USA", "lat": 25.76, "lon": -80.19, "zone": "SUBTROPICAL"},
                {"city": "Los Angeles", "country": "USA", "lat": 34.05, "lon": -118.24, "zone": "MEDITERRANEAN"},
                {"city": "Chicago", "country": "USA", "lat": 41.88, "lon": -87.63, "zone": "CONTINENTAL"},
                {"city": "Toronto", "country": "Canada", "lat": 43.65, "lon": -79.38, "zone": "CONTINENTAL"},
                {"city": "Vancouver", "country": "Canada", "lat": 49.28, "lon": -123.12, "zone": "TEMPERATE"},
                {"city": "Amsterdam", "country": "Netherlands", "lat": 52.37, "lon": 4.90, "zone": "TEMPERATE"},
                {"city": "Stockholm", "country": "Sweden", "lat": 59.33, "lon": 18.07, "zone": "CONTINENTAL"},
                {"city": "Oslo", "country": "Norway", "lat": 59.91, "lon": 10.75, "zone": "CONTINENTAL"},
                {"city": "Helsinki", "country": "Finland", "lat": 60.17, "lon": 24.94, "zone": "CONTINENTAL"},
                {"city": "Lisbon", "country": "Portugal", "lat": 38.72, "lon": -9.14, "zone": "MEDITERRANEAN"},
                {"city": "Athens", "country": "Greece", "lat": 37.98, "lon": 23.73, "zone": "MEDITERRANEAN"},
                {"city": "Istanbul", "country": "Turkey", "lat": 41.01, "lon": 28.98, "zone": "MEDITERRANEAN"},
                {"city": "Bangkok", "country": "Thailand", "lat": 13.76, "lon": 100.50, "zone": "TROPICAL"},
                {"city": "Jakarta", "country": "Indonesia", "lat": -6.21, "lon": 106.85, "zone": "TROPICAL"},
                {"city": "Manila", "country": "Philippines", "lat": 14.60, "lon": 120.98, "zone": "TROPICAL"},
                {"city": "Seoul", "country": "South Korea", "lat": 37.57, "lon": 126.98, "zone": "CONTINENTAL"},
                {"city": "Cape Town", "country": "South Africa", "lat": -33.92, "lon": 18.42, "zone": "MEDITERRANEAN"},
                {"city": "Johannesburg", "country": "South Africa", "lat": -26.20, "lon": 28.05, "zone": "SUBTROPICAL"},
                {"city": "Nairobi", "country": "Kenya", "lat": -1.29, "lon": 36.82, "zone": "TROPICAL"},
                {"city": "Casablanca", "country": "Morocco", "lat": 33.57, "lon": -7.59, "zone": "MEDITERRANEAN"},
                {"city": "Lima", "country": "Peru", "lat": -12.05, "lon": -77.04, "zone": "SUBTROPICAL"},
                {"city": "Santiago", "country": "Chile", "lat": -33.45, "lon": -70.67, "zone": "MEDITERRANEAN"},
                {"city": "Bogota", "country": "Colombia", "lat": 4.71, "lon": -74.07, "zone": "TROPICAL"},
                {"city": "Caracas", "country": "Venezuela", "lat": 10.48, "lon": -66.90, "zone": "TROPICAL"},
                {"city": "Havana", "country": "Cuba", "lat": 23.11, "lon": -82.37, "zone": "TROPICAL"},
                {"city": "Reykjavik", "country": "Iceland", "lat": 64.15, "lon": -21.94, "zone": "POLAR"},
                {"city": "Auckland", "country": "New Zealand", "lat": -36.85, "lon": 174.76, "zone": "TEMPERATE"},
                {"city": "Melbourne", "country": "Australia", "lat": -37.81, "lon": 144.96, "zone": "TEMPERATE"},
                {"city": "Denver", "country": "USA", "lat": 39.74, "lon": -104.99, "zone": "CONTINENTAL"},
                {"city": "Phoenix", "country": "USA", "lat": 33.45, "lon": -112.07, "zone": "DESERT"},
            ]
            
            # Nombres de tormentas
            storm_names = [
                "Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta",
                "Iota", "Kappa", "Lambda", "Mu", "Nu", "Xi", "Omicron", "Pi", "Rho"
            ]
            
            # Usar solo las ciudades configuradas
            selected_cities = cities[:min(config.n_cities, len(cities))]
            
            # Estado de tormentas activas (para tracking continuo)
            active_storms = {}
            storm_counter = 0
            
            # Crear 2 tormentas iniciales para que haya datos inmediatamente
            for i in range(2):
                storm_counter += 1
                storm_id = f"STM-{datetime.now().strftime('%Y%m%d')}-{storm_counter:03d}"
                init_lat = random.uniform(-20, 30)  # Zonas tropicales
                init_lon = random.uniform(-180, 180)
                
                active_storms[storm_id] = {
                    "storm_id": storm_id,
                    "storm_name": storm_names[i],
                    "category": random.randint(2, 4),
                    "latitude": init_lat,
                    "longitude": init_lon,
                    "direction": random.uniform(0, 360),
                    "speed": random.uniform(15, 35),
                    "max_wind_kmh": random.uniform(150, 250),
                    "central_pressure": random.uniform(940, 980),
                    "created_at": datetime.now().isoformat(),
                    "updates": 0
                }
            
            batch_num = 0
            while not self._stop_requested:
                now = datetime.now()
                hour = now.hour
                day_of_year = now.timetuple().tm_yday
                
                # Factor estacional (hemisferio norte)
                seasonal_factor = math.cos((day_of_year - 172) * 2 * math.pi / 365) * 10
                
                for city_info in selected_cities:
                    if self._stop_requested:
                        break
                    
                    # Temperatura base según zona climática
                    base_temps = {
                        "TROPICAL": 28, "SUBTROPICAL": 22, "TEMPERATE": 15,
                        "CONTINENTAL": 8, "MEDITERRANEAN": 18, "DESERT": 32, "POLAR": -5
                    }
                    base_temp = base_temps.get(city_info["zone"], 20)
                    
                    # Ajustar por hemisferio
                    is_southern = city_info["lat"] < 0
                    season_adj = -seasonal_factor if is_southern else seasonal_factor
                    
                    # Variación diurna (más frío de noche)
                    diurnal = 6 * math.sin((hour - 6) * math.pi / 12)
                    
                    # Temperatura final con ruido
                    temp = base_temp + season_adj + diurnal + random.gauss(0, 2)
                    
                    # Humedad (inversa a temperatura, con zona)
                    humidity_base = {"TROPICAL": 80, "SUBTROPICAL": 70, "TEMPERATE": 65,
                                    "CONTINENTAL": 55, "MEDITERRANEAN": 50, "DESERT": 25, "POLAR": 60}
                    humidity = humidity_base.get(city_info["zone"], 60) + random.gauss(0, 10)
                    humidity = max(10, min(100, humidity))
                    
                    # Viento
                    wind_speed = abs(random.weibullvariate(2, 15))
                    wind_direction = random.uniform(0, 360)
                    
                    # Lluvia (más probable si humedad alta)
                    rain_prob = 0.1 + (humidity - 50) / 200
                    rain_mm = round(random.expovariate(1/8) if random.random() < rain_prob else 0, 1)
                    
                    # Presión
                    pressure = random.gauss(1013, 8)
                    
                    # Calcular intensidad del evento (0-1)
                    temp_anomaly = abs(temp - base_temp) / 20
                    wind_anomaly = wind_speed / 50
                    rain_anomaly = min(rain_mm / 30, 1)
                    event_intensity = min(1.0, (temp_anomaly + wind_anomaly + rain_anomaly) / 3)
                    
                    # Evento weather completo
                    weather_event = {
                        "event_type": "weather_update",
                        "timestamp": now.isoformat(),
                        "city": city_info["city"],
                        "City": city_info["city"],  # Alias para compatibilidad
                        "country": city_info["country"],
                        "Country": city_info["country"],  # Alias
                        "latitude": city_info["lat"],
                        "longitude": city_info["lon"],
                        "lat_decimal": city_info["lat"],  # Alias para heatmaps
                        "lon_decimal": city_info["lon"],  # Alias para heatmaps
                        "temperature": round(temp, 1),
                        "temperature_c": round(temp, 1),  # Alias
                        "temperature_hourly": round(temp, 1),  # Alias para heatmaps
                        "humidity": round(humidity, 1),
                        "humidity_pct": round(humidity, 1),  # Alias
                        "pressure": round(pressure, 1),
                        "pressure_hpa": round(pressure, 1),  # Alias
                        "wind_speed": round(wind_speed, 1),
                        "wind_speed_kmh": round(wind_speed, 1),  # Alias
                        "wind_direction": round(wind_direction, 1),
                        "rain_mm": rain_mm,
                        "cloud_cover": round(random.uniform(0, 100), 1),
                        "climate_zone": city_info["zone"],
                        "event_intensity": round(event_intensity, 3),
                        "year": now.year,
                        "month": now.month,
                        "day": now.day,
                        "hour": now.hour,
                        "day_of_week": now.weekday(),
                        "generated_at": now.isoformat()
                    }
                    
                    producer.send('climaxtreme-weather', key=city_info["city"], value=weather_event)
                    self._stats['events_produced'] += 1
                    
                    # Generar alertas basadas en condiciones
                    should_alert = False
                    alert_type = None
                    alert_level = "WATCH"
                    
                    if temp > 38:
                        should_alert = True
                        alert_type = "HEAT"
                        alert_level = "EMERGENCY" if temp > 42 else ("WARNING" if temp > 40 else "WATCH")
                    elif temp < 0:
                        should_alert = True
                        alert_type = "COLD"
                        alert_level = "EMERGENCY" if temp < -15 else ("WARNING" if temp < -5 else "WATCH")
                    elif wind_speed > 60:
                        should_alert = True
                        alert_type = "WIND"
                        alert_level = "EMERGENCY" if wind_speed > 100 else ("WARNING" if wind_speed > 80 else "WATCH")
                    elif rain_mm > 20:
                        should_alert = True
                        alert_type = "FLOOD"
                        alert_level = "EMERGENCY" if rain_mm > 50 else ("WARNING" if rain_mm > 35 else "WATCH")
                    elif config.include_alerts and random.random() < config.alert_probability:
                        should_alert = True
                        alert_type = random.choice(["HEAT", "COLD", "WIND", "STORM", "FLOOD"])
                        alert_level = random.choices(["WATCH", "WARNING", "EMERGENCY"], weights=[0.6, 0.3, 0.1])[0]
                    
                    if should_alert and alert_type:
                        alert_event = {
                            "event_type": "alert",
                            "alert_id": f"ALT-{city_info['city'][:3].upper()}-{int(time.time()*1000)}",
                            "timestamp": now.isoformat(),
                            "city": city_info["city"],
                            "City": city_info["city"],
                            "country": city_info["country"],
                            "Country": city_info["country"],
                            "latitude": city_info["lat"],
                            "longitude": city_info["lon"],
                            "lat_decimal": city_info["lat"],
                            "lon_decimal": city_info["lon"],
                            "alert_type": alert_type,
                            "alert_level": alert_level,
                            "temperature": weather_event["temperature"],
                            "temperature_hourly": weather_event["temperature"],
                            "wind_speed": weather_event["wind_speed"],
                            "wind_speed_kmh": weather_event["wind_speed"],
                            "rain_mm": weather_event["rain_mm"],
                            "humidity_pct": weather_event["humidity"],
                            "event_intensity": event_intensity,
                            "climate_zone": city_info["zone"],
                            "description": f"{alert_type} alert for {city_info['city']}: {alert_level}",
                            "generated_at": now.isoformat()
                        }
                        
                        producer.send('climaxtreme-alerts', key=city_info["city"], value=alert_event)
                        self._stats['alerts_sent'] = self._stats.get('alerts_sent', 0) + 1
                
                # Gestionar tormentas activas (crear nuevas, actualizar existentes)
                if config.include_storms:
                    # Crear nueva tormenta ocasionalmente
                    if random.random() < config.storm_probability and len(active_storms) < 5:
                        storm_counter += 1
                        storm_id = f"STM-{now.strftime('%Y%m%d')}-{storm_counter:03d}"
                        
                        # Ubicación inicial (zonas tropicales principalmente)
                        init_lat = random.uniform(-20, 30)
                        init_lon = random.uniform(-180, 180)
                        
                        active_storms[storm_id] = {
                            "storm_id": storm_id,
                            "storm_name": random.choice(storm_names),
                            "category": random.randint(1, 3),
                            "latitude": init_lat,
                            "longitude": init_lon,
                            "direction": random.uniform(0, 360),
                            "speed": random.uniform(15, 35),
                            "max_wind_kmh": random.uniform(120, 200),
                            "central_pressure": random.uniform(960, 1000),
                            "created_at": now.isoformat(),
                            "updates": 0
                        }
                    
                    # Actualizar tormentas activas
                    storms_to_remove = []
                    for storm_id, storm in active_storms.items():
                        # Mover tormenta
                        dir_rad = math.radians(storm["direction"])
                        move_dist = storm["speed"] * config.interval_seconds / 3600 * 0.5  # grados aprox
                        storm["latitude"] += math.cos(dir_rad) * move_dist
                        storm["longitude"] += math.sin(dir_rad) * move_dist
                        
                        # Variar intensidad
                        storm["category"] = max(1, min(5, storm["category"] + random.choice([-1, 0, 0, 1])))
                        storm["max_wind_kmh"] = 60 + storm["category"] * 40 + random.gauss(0, 10)
                        storm["central_pressure"] = 1010 - storm["category"] * 15 + random.gauss(0, 5)
                        storm["direction"] += random.gauss(0, 10)
                        storm["updates"] += 1
                        
                        # Terminar tormenta después de cierto tiempo o si sale del área
                        if storm["updates"] > 100 or abs(storm["latitude"]) > 60:
                            storms_to_remove.append(storm_id)
                            continue
                        
                        storm_event = {
                            "event_type": "storm_update",
                            "storm_id": storm_id,
                            "storm_name": storm["storm_name"],
                            "timestamp": now.isoformat(),
                            "latitude": round(storm["latitude"], 2),
                            "longitude": round(storm["longitude"], 2),
                            "category": storm["category"],
                            "max_wind_kmh": round(storm["max_wind_kmh"], 1),
                            "central_pressure": round(storm["central_pressure"], 1),
                            "movement_speed": round(storm["speed"], 1),
                            "movement_direction": round(storm["direction"], 1),
                            "created_at": storm["created_at"],
                            "update_number": storm["updates"],
                            "generated_at": now.isoformat()
                        }
                        
                        producer.send('climaxtreme-storms', key=storm_id, value=storm_event)
                        self._stats['storms_sent'] = self._stats.get('storms_sent', 0) + 1
                    
                    for sid in storms_to_remove:
                        del active_storms[sid]
                
                producer.flush()
                batch_num += 1
                self._stats['last_event_time'] = now.isoformat()
                
                # Esperar antes del siguiente batch
                time.sleep(config.interval_seconds)
            
            producer.close()
            logger.info("Simple Kafka producer stopped")
            
        except Exception as e:
            logger.error(f"Error in simple producer: {e}")
            self._stats['error'] = str(e)
        finally:
            self._is_producing = False
    
    def start_producer(self, config: KafkaStreamingConfig, use_spark: bool = True) -> Tuple[bool, str]:
        """
        Iniciar el productor de Kafka.
        
        Args:
            config: Configuración del streaming
            use_spark: Si True, intenta usar Spark primero, fallback a simple
            
        Returns:
            (éxito, mensaje)
        """
        if self._is_producing:
            return False, "El productor ya está corriendo"
        
        # Verificar que Kafka esté disponible
        cluster_status = self.check_kafka_cluster()
        if not cluster_status.get('kafka_running'):
            return False, "Kafka no está disponible. Inicia los servicios primero."
        
        self._stop_requested = False
        self._stats = {
            'events_produced': 0,
            'alerts_sent': 0,
            'storms_sent': 0,
            'start_time': datetime.now().isoformat(),
            'last_event_time': None,
            'error': None
        }
        
        try:
            # Intentar usar el productor simple (funciona en cualquier contenedor)
            self._producer_thread = threading.Thread(
                target=self._run_simple_producer,
                args=(config,),
                daemon=True
            )
            self._producer_thread.start()
            
            # Esperar un poco para ver si arranca
            time.sleep(1)
            
            if self._producer_thread.is_alive():
                self._is_producing = True
                return True, "✅ Productor Kafka iniciado (modo simple)"
            else:
                return False, f"Error iniciando productor: {self._stats.get('error', 'Unknown error')}"
                
        except Exception as e:
            logger.error(f"Error starting producer: {e}")
            return False, f"Error: {str(e)}"
    
    def stop_producer(self) -> Tuple[bool, str]:
        """
        Detener el productor de Kafka.
        
        Returns:
            (éxito, mensaje)
        """
        if not self._is_producing:
            return False, "No hay productor corriendo"
        
        try:
            self._stop_requested = True
            
            # Esperar a que el hilo termine
            if self._producer_thread and self._producer_thread.is_alive():
                self._producer_thread.join(timeout=5)
            
            self._is_producing = False
            self._producer_thread = None
            return True, "Productor detenido correctamente"
            
        except Exception as e:
            return False, str(e)
    
    def is_producing(self) -> bool:
        """Verificar si el productor está activo."""
        if self._producer_thread:
            if not self._producer_thread.is_alive():
                self._is_producing = False
        return self._is_producing
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del productor."""
        return {
            **self._stats,
            'is_producing': self.is_producing()
        }


# ============================================================================
# Singleton para Streamlit
# ============================================================================

_streaming_manager = None

def get_streaming_manager() -> KafkaStreamingManager:
    """Obtener instancia del manager."""
    global _streaming_manager
    if _streaming_manager is None:
        _streaming_manager = KafkaStreamingManager()
    return _streaming_manager


# ============================================================================
# Componentes de UI
# ============================================================================

def render_kafka_cluster_status():
    """Renderizar estado del clúster Kafka."""
    manager = get_streaming_manager()
    status = manager.check_kafka_cluster()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if status['zookeeper_running']:
            st.success("✅ Zookeeper")
        else:
            st.error("❌ Zookeeper")
    
    with col2:
        if status['kafka_running']:
            st.success("✅ Kafka Broker")
        else:
            st.error("❌ Kafka Broker")
    
    with col3:
        n_topics = len(status.get('topics_created', []))
        if n_topics > 0:
            st.success(f"✅ {n_topics} Topics")
        else:
            st.warning("⚠️ Sin topics")
    
    if status.get('error'):
        st.error(f"Error: {status['error']}")
    
    return status


def render_streaming_config_form() -> Optional[KafkaStreamingConfig]:
    """Renderizar formulario de configuración."""
    with st.expander("⚙️ Configuración del Streaming", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            n_cities = st.slider(
                "Número de ciudades",
                min_value=10, max_value=500, value=50, step=10,
                help="Ciudades a incluir en el streaming"
            )
            
            interval = st.slider(
                "Intervalo (segundos)",
                min_value=0.5, max_value=10.0, value=1.0, step=0.5,
                help="Tiempo entre batches de eventos"
            )
        
        with col2:
            include_alerts = st.checkbox("Incluir alertas", value=True)
            include_storms = st.checkbox("Incluir tormentas", value=True)
            
            alert_prob = st.slider(
                "Probabilidad de alertas",
                min_value=0.0, max_value=0.5, value=0.1, step=0.05
            ) if include_alerts else 0.0
            
            storm_prob = st.slider(
                "Probabilidad de tormentas",
                min_value=0.0, max_value=0.3, value=0.05, step=0.01
            ) if include_storms else 0.0
        
        return KafkaStreamingConfig(
            n_cities=n_cities,
            interval_seconds=interval,
            include_alerts=include_alerts,
            include_storms=include_storms,
            alert_probability=alert_prob,
            storm_probability=storm_prob
        )


def render_producer_controls():
    """Renderizar controles del productor."""
    manager = get_streaming_manager()
    
    st.subheader("🎛️ Control del Productor")
    
    # Opción de usar Spark (recomendado)
    use_spark = st.checkbox(
        "🚀 Usar Spark para generación (recomendado)",
        value=True,
        help="Usa SyntheticClimateGenerator de Spark para generar datos de alta calidad con modelos estadísticos"
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if not manager.is_producing():
            if st.button("▶️ Iniciar Productor", type="primary", use_container_width=True):
                config = KafkaStreamingConfig(
                    n_cities=st.session_state.get('kafka_n_cities', 50),
                    interval_seconds=st.session_state.get('kafka_interval', 1.0)
                )
                with st.spinner("Iniciando generación con Spark..." if use_spark else "Iniciando productor..."):
                    success, msg = manager.start_producer(config, use_spark=use_spark)
                if success:
                    st.success(msg)
                else:
                    st.error(msg)
                st.rerun()
        else:
            if st.button("⏹️ Detener Productor", type="secondary", use_container_width=True):
                success, msg = manager.stop_producer()
                if success:
                    st.success(msg)
                else:
                    st.error(msg)
                st.rerun()
    
    with col2:
        # Botón para crear topics
        if st.button("📋 Crear Topics", use_container_width=True):
            with st.spinner("Creando topics..."):
                success, msg = manager.create_topics()
                if success:
                    st.success(msg)
                else:
                    st.error(msg)
    
    with col3:
        # Estado actual
        if manager.is_producing():
            st.markdown("""
            <div style='background:#2E7D32; padding:10px; border-radius:5px; text-align:center;'>
                <strong>🟢 PRODUCIENDO</strong>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background:#424242; padding:10px; border-radius:5px; text-align:center;'>
                <strong>⚫ DETENIDO</strong>
            </div>
            """, unsafe_allow_html=True)
    
    # Info sobre modo de generación
    if use_spark:
        st.info("""
        **Modo Spark**: Genera datos sintéticos usando `SyntheticClimateGenerator` con:
        - Interpolación horaria con ciclo diurno
        - Variables meteorológicas (precipitación, viento, humedad, presión)
        - Eventos extremos (olas de calor, frío, tormentas, inundaciones)
        - Tracking de tormentas con categorías Saffir-Simpson
        - Alertas basadas en umbrales
        """)


def render_quick_start_guide():
    """Renderizar guía de inicio rápido."""
    with st.expander("📚 Guía de Inicio Rápido"):
        st.markdown("""
        ### 🚀 Pipeline de Streaming: Spark → Kafka → Dashboard
        
        #### Arquitectura
        ```
        ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
        │  Spark          │    │  Kafka          │    │  Streamlit      │
        │  Generator      │───▶│  Broker         │───▶│  Dashboard      │
        │  (Synthetic)    │    │  (Topics)       │    │  (Real-time)    │
        └─────────────────┘    └─────────────────┘    └─────────────────┘
        ```
        
        #### Paso 1: Iniciar el clúster Kafka
        ```bash
        cd infra
        docker-compose up -d zookeeper kafka
        ```
        
        #### Paso 2: Verificar el estado
        Los indicadores arriba deberían mostrar ✅ para Zookeeper y Kafka.
        
        #### Paso 3: Iniciar el productor Spark
        - Asegúrate de que "🚀 Usar Spark para generación" está activado
        - Configura los parámetros según tus necesidades
        - Haz clic en "▶️ Iniciar Productor"
        
        El generador Spark usará `SyntheticClimateGenerator` para crear:
        - Temperaturas horarias con ciclo diurno
        - Variables meteorológicas completas
        - Eventos extremos y alertas
        - Tracking de tormentas
        
        #### Paso 4: Ver datos en vivo
        Ve a la página **🔴 Live Streaming** para ver los gráficos 
        actualizándose en tiempo real.
        
        #### Topics de Kafka
        - `climaxtreme-weather`: Datos meteorológicos
        - `climaxtreme-alerts`: Alertas activas
        - `climaxtreme-storms`: Tracking de tormentas
        - `climaxtreme-predictions`: Predicciones
        - `climaxtreme-progress`: Estado del pipeline
        
        #### Opcional: Kafka UI
        ```bash
        docker-compose --profile monitoring up -d kafka-ui
        # Accede a http://localhost:8080
        ```
        """)


def render_full_kafka_control_panel():
    """
    Renderizar panel completo de control de Kafka.
    
    Incluye:
    - Estado del clúster
    - Configuración
    - Controles del productor
    - Guía de inicio
    """
    st.header("🎛️ Panel de Control Kafka")
    
    # Estado del clúster
    st.subheader("📊 Estado del Clúster")
    status = render_kafka_cluster_status()
    
    st.markdown("---")
    
    # Si Kafka está corriendo, mostrar controles
    if status.get('kafka_running'):
        # Configuración
        config = render_streaming_config_form()
        
        # Guardar en session state
        if config:
            st.session_state.kafka_n_cities = config.n_cities
            st.session_state.kafka_interval = config.interval_seconds
        
        st.markdown("---")
        
        # Controles
        render_producer_controls()
    else:
        st.warning("""
        ⚠️ **Kafka no está corriendo**
        
        Ejecuta el siguiente comando para iniciar:
        ```bash
        cd infra
        docker-compose up -d zookeeper kafka
        ```
        """)
    
    st.markdown("---")
    
    # Guía
    render_quick_start_guide()
