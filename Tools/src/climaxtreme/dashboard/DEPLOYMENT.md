# climaXtreme Dashboard - Deployment Guide

## 🚀 Quick Start

### Opción 1: Usar el Nuevo Dashboard (Recomendado)

```bash
# Navegar al directorio del dashboard
cd Tools/src/climaxtreme/dashboard

# Ejecutar script de migración (PowerShell)
.\migrate_to_new.ps1

# O manualmente
Copy-Item app_new.py app.py -Force

# Lanzar
streamlit run app.py
```

### Opción 2: Lanzar Directamente el Nuevo Dashboard

```bash
cd Tools/src/climaxtreme/dashboard
streamlit run app_new.py
```

### Opción 3: Dashboard Antiguo (Para Comparación)

```bash
cd Tools/src/climaxtreme/dashboard
streamlit run app_old_backup.py  # Si ya migraste
# O
streamlit run app.py  # Si no has migrado
```

## 📊 Comparación de Versiones

| Característica | Dashboard Antiguo | Dashboard Nuevo |
|----------------|-------------------|-----------------|
| Archivo principal | `app.py` (1572 líneas) | `app_new.py` (200 líneas) |
| Arquitectura | Monolítico | Multi-página |
| Páginas | 1 archivo con tabs | 7 archivos separados + home |
| Navegación | Tabs anidados | Sidebar con autodescubrimiento |
| Optimización HDFS | Básica | Avanzada |
| Mapas interactivos | No | Sí (scatter geo) |
| Parquets soportados | Intenta todos | Específico por página |
| Mantenibilidad | Baja | Alta |
| Performance | Buena | Excelente |
| Extensibilidad | Difícil | Fácil (agregar página) |

## 🗂️ Estructura de Archivos

```
dashboard/
│
├── app.py                  # Dashboard ACTIVO (actualizar con migrate_to_new.ps1)
├── app_new.py              # Nuevo dashboard (multi-página)
├── app_old_backup.py       # Backup del dashboard antiguo
├── utils.py                # Utilidades compartidas (NUEVO)
│
├── pages/                  # Páginas del nuevo dashboard (NUEVO)
│   ├── 1_📈_Temporal_Analysis.py
│   ├── 2_🌡️_Anomalies.py
│   ├── 3_🍂_Seasonal_Analysis.py
│   ├── 4_⚡_Extreme_Events.py
│   ├── 5_🗺️_Regional_Analysis.py
│   ├── 6_🌐_Continental_Analysis.py
│   └── 7_📊_Statistical_Analysis.py
│
├── migrate_to_new.ps1      # Script de migración
├── README_NEW_DASHBOARD.md # Documentación completa
└── DEPLOYMENT.md           # Este archivo
```

## 🔧 Configuración de Producción

### 1. Variables de Entorno (Opcional)

```bash
# PowerShell
$env:CLIMAXTREME_HDFS_HOST = "climaxtreme-namenode"
$env:CLIMAXTREME_HDFS_PORT = "9000"
$env:CLIMAXTREME_HDFS_PATH = "/data/processed"
```

### 2. Configuración HDFS

El dashboard se configura desde el **sidebar**:

1. Seleccionar "HDFS" como fuente de datos
2. Ingresar:
   - **Host**: `climaxtreme-namenode` (o IP del namenode)
   - **Puerto**: `9000`
   - **Base Path**: `/data/processed`
3. Click en "Test Connection" para verificar

### 3. Lanzamiento en Servidor

```bash
# Con host específico y puerto
streamlit run app.py --server.address 0.0.0.0 --server.port 8501

# Con configuración adicional
streamlit run app.py \
    --server.address 0.0.0.0 \
    --server.port 8501 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection true
```

### 4. Docker Deployment (Futuro)

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY Tools/src/climaxtreme /app/climaxtreme

EXPOSE 8501

CMD ["streamlit", "run", "climaxtreme/dashboard/app.py", \
     "--server.address", "0.0.0.0", \
     "--server.port", "8501", \
     "--server.headless", "true"]
```

## 🔄 Proceso de Migración Detallado

### Pre-requisitos

```bash
# Verificar que tienes los parquets en HDFS
docker exec namenode hdfs dfs -ls /data/processed/

# Verificar que tienes las dependencias
pip install streamlit pandas plotly pyarrow
```

### Pasos

1. **Backup del dashboard actual**
   ```powershell
   cd Tools/src/climaxtreme/dashboard
   Copy-Item app.py app_old_backup.py
   ```

2. **Ejecutar script de migración**
   ```powershell
   .\migrate_to_new.ps1
   ```

3. **Verificar estructura**
   ```powershell
   # Debe existir:
   Get-ChildItem pages/
   Get-Item utils.py
   ```

4. **Probar localmente**
   ```bash
   streamlit run app.py
   ```

5. **Verificar en navegador**
   - Abrir http://localhost:8501
   - Verificar que aparecen 8 páginas en el sidebar
   - Probar conexión HDFS
   - Navegar por cada página

### Rollback

Si algo falla:

```powershell
# Volver al dashboard antiguo
Copy-Item app_old_backup.py app.py -Force

# Relanzar
streamlit run app.py
```

## 🐛 Troubleshooting

### Problema: "No module named 'climaxtreme.dashboard.utils'"

**Solución:**
```bash
# Opción 1: Instalar paquete
cd Tools
pip install -e .

# Opción 2: Agregar al PYTHONPATH
$env:PYTHONPATH = "c:\Users\HP\Desktop\PGVD\Proyecto\climaXtreme\Tools\src"
```

### Problema: "Failed to load X.parquet from HDFS"

**Verificaciones:**
1. ¿Está corriendo el namenode?
   ```bash
   docker ps | grep namenode
   ```

2. ¿Existen los parquets?
   ```bash
   docker exec namenode hdfs dfs -ls /data/processed/
   ```

3. ¿Es accesible desde el host?
   ```bash
   telnet climaxtreme-namenode 9000
   ```

**Solución temporal:**
- Cambiar a "Local Files" en el sidebar
- Copiar parquets localmente:
  ```bash
  docker exec namenode hdfs dfs -get /data/processed/*.parquet ./DATA/processed/
  ```

### Problema: Páginas no aparecen en sidebar

**Causas:**
- Archivos no están en `pages/`
- Falta prefijo numérico en nombres
- Errores de sintaxis en archivos

**Verificación:**
```powershell
# Listar páginas
Get-ChildItem pages/*.py | Select-Object Name

# Deben verse como:
# 1_📈_Temporal_Analysis.py
# 2_🌡️_Anomalies.py
# etc.
```

### Problema: "streamlit: command not found"

**Solución:**
```bash
# Reinstalar streamlit
pip install --upgrade streamlit

# O usar módulo de Python
python -m streamlit run app.py
```

## 📚 Recursos Adicionales

- **Documentación Completa**: `README_NEW_DASHBOARD.md`
- **Esquemas de Parquets**: `../../documentation/PARQUETS.md`
- **Streamlit Docs**: https://docs.streamlit.io
- **Plotly Docs**: https://plotly.com/python/

## ✅ Checklist de Deployment

- [ ] Backup del dashboard antiguo creado
- [ ] Script `migrate_to_new.ps1` ejecutado
- [ ] Directorio `pages/` existe con 7 páginas
- [ ] Archivo `utils.py` presente
- [ ] Parquets disponibles en HDFS o localmente
- [ ] Conexión HDFS testeada desde el dashboard
- [ ] Todas las páginas cargan correctamente
- [ ] Visualizaciones funcionan
- [ ] Filtros interactivos responden
- [ ] Performance es aceptable

## 🎯 Recomendación Final

**Para producción, usar el dashboard nuevo (`app_new.py`)**:
- Mejor performance
- Más mantenible
- Diseñado específicamente para los parquets de Spark
- Arquitectura escalable

**Mantener el antiguo solo como referencia o fallback.**

---

**Última actualización**: 2024
**Autor**: Equipo climaXtreme
