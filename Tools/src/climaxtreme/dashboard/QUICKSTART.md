# 🚀 Quick Start - climaXtreme Dashboard Modernizado

## ⚡ Inicio Rápido (3 pasos)

### 1. Migrar al Nuevo Dashboard

```powershell
# Navegar al directorio
cd c:\Users\HP\Desktop\PGVD\Proyecto\climaXtreme\Tools\src\climaxtreme\dashboard

# Ejecutar migración
.\migrate_to_new.ps1

# O manualmente
Copy-Item app_new.py app.py -Force
```

### 2. Lanzar el Dashboard

```powershell
streamlit run app.py
```

### 3. Configurar en el Navegador

1. Abrir: http://localhost:8501
2. En el sidebar → Seleccionar **"HDFS"**
3. Configurar:
   - **Host**: `climaxtreme-namenode`
   - **Puerto**: `9000`
   - **Base Path**: `/data/processed`
4. Click **"🔍 Test Connection"**
5. ¡Explorar las 7 páginas de análisis!

---

## 📋 Checklist Pre-lanzamiento

- [ ] Docker containers corriendo (`docker ps`)
- [ ] Parquets en HDFS (`docker exec namenode hdfs dfs -ls /data/processed/`)
- [ ] Python 3.9+ instalado
- [ ] Dependencias instaladas (`pip install streamlit pandas plotly pyarrow`)

---

## 🗺️ Navegación del Dashboard

| Página | Icono | Parquet(s) | Qué muestra |
|--------|-------|-----------|-------------|
| **Home** | 🏠 | descriptive_stats | Estado del sistema, datasets disponibles |
| **Temporal Analysis** | 📈 | monthly, yearly | Tendencias mensuales/anuales, series temporales |
| **Anomalies** | 🌡️ | anomalies, climatology | Desviaciones vs norma climatológica |
| **Seasonal Analysis** | 🍂 | seasonal | Patrones estacionales (Winter/Spring/Summer/Fall) |
| **Extreme Events** | ⚡ | extreme_thresholds | Umbrales P10/P90, detección de extremos |
| **Regional Analysis** | 🗺️ | regional | Mapa global con 16 regiones |
| **Continental Analysis** | 🌐 | continental | Vista por 7 continentes |
| **Statistical Analysis** | 📊 | descriptive_stats, correlation_matrix, chi_square_tests | Stats, correlaciones, tests |

---

## 🎯 Casos de Uso Rápidos

### Ver tendencia global de temperatura
1. **Temporal Analysis** → tab "Yearly Trends"
2. Aggregation: **"Global"**
3. Ver gráfica con trend line

### Identificar ciudad con mayor calentamiento
1. **Temporal Analysis** → tab "Time Series Explorer"
2. Seleccionar múltiples ciudades
3. Comparar slopes de las líneas

### Encontrar eventos extremos recientes
1. **Extreme Events**
2. Seleccionar ciudad
3. Ver tabla "Recent Extreme Events"

### Comparar regiones geográficas
1. **Regional Analysis**
2. Slider a año reciente (ej: 2015)
3. Ver mapa + gráfica de barras

### Ver correlaciones entre variables
1. **Statistical Analysis** → tab "Correlations"
2. Ver heatmap
3. Revisar tabla de top correlaciones

---

## 🐛 Solución Rápida de Problemas

### ❌ "Failed to load X.parquet"

**Causa**: Parquet no existe en HDFS o conexión fallida

**Solución**:
```powershell
# Verificar que existen los parquets
docker exec namenode hdfs dfs -ls /data/processed/

# Si no existen, procesarlos
.\scripts\process_full_dataset.ps1

# Si HDFS no responde, usar Local:
# En sidebar → "Local Files"
```

### ❌ "Module not found: climaxtreme"

**Solución**:
```powershell
cd Tools
pip install -e .
```

### ❌ Páginas no aparecen en sidebar

**Verificación**:
```powershell
# Debe listar 7 archivos .py
Get-ChildItem pages/*.py
```

**Solución**: Verificar que el directorio `pages/` existe y contiene los archivos

---

## 📖 Documentación Completa

- **Arquitectura detallada**: `README_NEW_DASHBOARD.md`
- **Guía de deployment**: `DEPLOYMENT.md`
- **Resumen ejecutivo**: `SUMMARY.md`
- **Esquemas de parquets**: `../../documentation/PARQUETS.md`

---

## 💡 Tips de Uso

- 🔄 **Caché**: Los datos se cachean 5 minutos. Para refrescar, reload la página (F5)
- 🎨 **Filtros**: Los filtros son jerárquicos (País → Ciudad)
- 🗺️ **Mapas**: Hover sobre burbujas para ver detalles
- 📊 **Gráficas**: Todas son interactivas (zoom, pan, select)
- 💾 **Export**: Click en 📷 en esquina superior derecha de cada gráfica

---

## 🎉 ¡Listo!

El dashboard modernizado está **100% funcional** y optimizado para los parquets de Spark.

**Disfruta explorando los datos climáticos! 🌡️**
