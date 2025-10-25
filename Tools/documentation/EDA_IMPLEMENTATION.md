# 📊 Análisis Exploratorio de Datos (EDA) - climaXtreme

## Resumen de Implementación

Este documento describe la implementación completa del módulo de **Análisis Exploratorio de Datos (EDA)** en climaXtreme, que genera estadísticas avanzadas procesadas en Spark y visualizadas en el dashboard.

---

## ✅ Componentes Implementados

### 1. **Procesamiento Spark** (3 nuevas funciones)

**Ubicación:** `Tools/src/climaxtreme/preprocessing/spark_processor.py`

#### 1.1 Matriz de Correlación de Pearson

```python
def compute_correlation_matrix(df: DataFrame) -> DataFrame
```

**Líneas:** 757-811

**Descripción:**
- Calcula correlaciones de Pearson entre todas las variables numéricas
- Variables analizadas: `year`, `avg_temperature`, `min_temperature`, `max_temperature`, `temperature_range`
- Genera matriz simétrica completa (incluye triángulo superior e inferior)
- Incluye valor absoluto de correlación para ordenamiento

**Output:** `correlation_matrix.parquet`
- Columnas: `variable_1`, `variable_2`, `correlation`, `abs_correlation`
- ~25 registros (combinaciones pairwise de 5 variables)

**Métricas clave:**
- Rango: -1 (correlación negativa perfecta) a +1 (correlación positiva perfecta)
- |r| > 0.7 = Correlación fuerte
- |r| > 0.4 = Correlación moderada
- |r| < 0.3 = Correlación débil

---

#### 1.2 Estadísticas Descriptivas

```python
def compute_descriptive_statistics(df: DataFrame) -> DataFrame
```

**Líneas:** 813-920

**Descripción:**
- Calcula 11 estadísticas descriptivas para cada variable numérica
- Variables analizadas: `avg_temperature`, `min_temperature`, `max_temperature`, `uncertainty`
- Incluye medidas de tendencia central, dispersión, forma y posición

**Estadísticas calculadas:**
1. **Conteo** (`count`): Número de observaciones válidas
2. **Media** (`mean`): Promedio aritmético
3. **Desviación estándar** (`std_dev`): Dispersión alrededor de la media
4. **Mínimo** (`min`): Valor más bajo
5. **Primer cuartil** (`q1`): Percentil 25
6. **Mediana** (`median`): Percentil 50 (valor central)
7. **Tercer cuartil** (`q3`): Percentil 75
8. **Máximo** (`max`): Valor más alto
9. **Rango intercuartílico** (`iqr`): Q3 - Q1 (medida de dispersión robusta)
10. **Asimetría** (`skewness`): Simetría de la distribución
11. **Curtosis** (`kurtosis`): "Peso" de las colas de la distribución

**Output:** `descriptive_stats.parquet`
- Formato pivotado: cada fila = 1 variable, columnas = estadísticas
- ~4 registros (4 variables × 11 estadísticas cada una)

**Interpretación de Skewness:**
- < -1: Muy sesgada izquierda
- -0.5 a 0.5: Aproximadamente simétrica
- \> 1: Muy sesgada derecha

**Interpretación de Kurtosis:**
- < 0: Platicúrtica (colas ligeras)
- ≈ 0: Mesocúrtica (normal)
- \> 0: Leptocúrtica (colas pesadas)

---

#### 1.3 Pruebas de Independencia Chi-Cuadrado

```python
def compute_chi_square_tests(df: DataFrame) -> DataFrame
def _chi_square_manual(df: DataFrame, var1: str, var2: str) -> Dict[str, float]
```

**Líneas:** 922-1044

**Descripción:**
- Realiza pruebas de independencia χ² para pares de variables categóricas
- Crea categorías de temperatura: `Cold` (<10°C), `Moderate` (10-20°C), `Hot` (>20°C)
- Calcula manualmente el estadístico χ² usando tablas de contingencia
- Computa p-value usando la distribución chi-cuadrado de scipy

**Tests realizados:**

1. **Continente vs Categoría de Temperatura**
   - H₀: La distribución de temperaturas es independiente del continente
   - Variables: 7 continentes × 3 categorías de temperatura

2. **Estación vs Categoría de Temperatura**
   - H₀: Las temperaturas no varían significativamente por estación
   - Variables: 4 estaciones × 3 categorías de temperatura

3. **Período Temporal vs Categoría de Temperatura**
   - H₀: La distribución de temperaturas no ha cambiado entre períodos temprano/tardío
   - Variables: 2 períodos × 3 categorías de temperatura

**Fórmula Chi-Cuadrado:**
```
χ² = Σ [(Observado - Esperado)² / Esperado]
```

**Grados de libertad:**
```
df = (filas - 1) × (columnas - 1)
```

**Output:** `chi_square_tests.parquet`
- Columnas: `test`, `variable_1`, `variable_2`, `chi_square_statistic`, `p_value`, `degrees_of_freedom`, `is_significant`
- ~3 registros (3 tests)

**Criterio de decisión:**
- **p < 0.05**: Rechazar H₀ → Variables **dependientes** (significativo) ✅
- **p ≥ 0.05**: No rechazar H₀ → Variables **independientes** (no significativo) ❌

---

### 2. **Dashboard Visualización** (1 nueva pestaña)

**Ubicación:** `Tools/src/climaxtreme/dashboard/app.py`

#### Pestaña 7: "📊 Exploratory Analysis (EDA)"

```python
def create_eda_tab(df: pd.DataFrame, *, max_points_to_plot: int)
```

**Líneas:** ~1350-1540

**Descripción:**
Tab inteligente que detecta automáticamente qué tipo de archivo EDA se ha cargado y muestra las visualizaciones apropiadas.

**Visualizaciones implementadas:**

#### 2.1 Para `correlation_matrix.parquet`:

**A) Heatmap de Correlación**
- Mapa de calor interactivo usando Plotly
- Escala de colores: `RdBu_r` (rojo = positivo, azul = negativo)
- Valores de correlación superpuestos en cada celda
- Matriz simétrica completa

**B) Top Correlaciones**
- Lista de las 10 correlaciones más fuertes
- División en correlaciones positivas y negativas
- Formato: "variable_1 ↔ variable_2: 0.XXX"

#### 2.2 Para `descriptive_stats.parquet`:

**A) Tabla Estilizada**
- DataFrame con formato numérico (3 decimales)
- Gradiente de color de fondo (YlOrRd)
- Todas las estadísticas visibles en formato tabular

**B) Gráfico de Barras con Error Bars**
- Barras = Media de cada variable
- Error bars = Desviación estándar
- Permite comparar magnitudes y dispersión

**C) Box Plot Representation**
- Box plots interactivos de Plotly
- Muestra los 5 números resumen (min, Q1, median, Q3, max)
- Visualización clara de la distribución

#### 2.3 Para `chi_square_tests.parquet`:

**A) Tabla de Resultados**
- Formato de científico para p-values (ej: 1.23e-05)
- Resaltado verde para tests significativos
- Muestra todos los detalles: χ², p-value, df, significancia

**B) Gráfico de Barras de Estadísticos χ²**
- Altura de barra = valor del estadístico χ²
- Color: Rojo = significativo, Verde = no significativo
- Anotaciones con p-values

**C) Interpretación Textual**
- Mensajes claros para cada test
- ⚠️ Advertencia: Si hay relación significativa
- ✅ Success: Si no hay relación significativa
- Explicación de qué significa cada resultado

---

### 3. **Script de Procesamiento** (Actualizado)

**Ubicación:** `scripts/process_full_dataset.ps1`

**Cambio:** Lista de artifacts actualizada de 8 a **11 archivos**

```powershell
$artifacts = @(
    "monthly.parquet", 
    "yearly.parquet", 
    "anomalies.parquet", 
    "climatology.parquet", 
    "seasonal.parquet", 
    "extreme_thresholds.parquet", 
    "regional.parquet", 
    "continental.parquet",
    "correlation_matrix.parquet",      # NUEVO
    "descriptive_stats.parquet",       # NUEVO
    "chi_square_tests.parquet"         # NUEVO
)
```

---

### 4. **Documentación** (Actualizada)

#### `PARQUETS.md`
- ✅ Actualizado resumen: 8 → **11 archivos**
- ✅ Agregadas 3 nuevas secciones detalladas:
  - 9️⃣ `correlation_matrix.parquet`
  - 🔟 `descriptive_stats.parquet`
  - 1️⃣1️⃣ `chi_square_tests.parquet`
- ✅ Actualizado flujo de procesamiento
- ✅ Actualizado dashboard: 6 → **7 pestañas**

#### `to-do.txt`
- ✅ Tarea marcada como completada:
  ```
  - hacer analisis exploratorio con matriz de pearson, estadistica descriptiva y pruebas chi-cuadrado - done
  ```

---

## 🚀 Cómo Usar

### Paso 1: Procesar los Datos

Ejecuta el script completo de procesamiento:

```powershell
.\scripts\process_full_dataset.ps1
```

Esto generará los 11 archivos Parquet, incluyendo los 3 nuevos de EDA.

### Paso 2: Iniciar el Dashboard

```bash
python -m climaxtreme.cli dashboard
```

### Paso 3: Cargar Archivos EDA

En el dashboard:

1. **Modo:** Selecciona "HDFS (Recommended)"
2. **Configurar HDFS:**
   - Host: `climaxtreme-namenode`
   - Port: `9000`
   - Base Path: `/data/climaxtreme/processed`

3. **Cargar cada archivo EDA:**
   - `correlation_matrix.parquet` → Ver heatmap de correlaciones
   - `descriptive_stats.parquet` → Ver estadísticas descriptivas
   - `chi_square_tests.parquet` → Ver pruebas de independencia

4. **Navegar a Tab 7:** "📊 Exploratory Analysis (EDA)"

---

## 📊 Ejemplos de Visualización

### Heatmap de Correlación

```
                    year  avg_temp  min_temp  max_temp  temp_range
year              1.000     0.723     0.698     0.741       0.156
avg_temp          0.723     1.000     0.987     0.989      -0.234
min_temp          0.698     0.987     1.000     0.952      -0.456
max_temp          0.741     0.989     0.952     1.000       0.234
temp_range        0.156    -0.234    -0.456     0.234       1.000
```

**Interpretación:**
- `avg_temp` ↔ `min_temp`: **0.987** (correlación muy fuerte positiva) ✅
- `avg_temp` ↔ `max_temp`: **0.989** (correlación muy fuerte positiva) ✅
- `year` ↔ `avg_temp`: **0.723** (correlación fuerte positiva - tendencia de calentamiento) 🌡️
- `temp_range` ↔ `min_temp`: **-0.456** (correlación moderada negativa)

---

### Estadísticas Descriptivas

**avg_temperature:**
```
count:      8,623,450
mean:       15.42°C
std_dev:    10.87°C
min:        -42.8°C
q1:         7.3°C
median:     15.8°C
q3:         23.4°C
max:        38.9°C
iqr:        16.1°C
skewness:   -0.123 (aproximadamente simétrica)
kurtosis:   -0.456 (ligeramente platicúrtica)
```

---

### Pruebas Chi-Cuadrado

**Test: Continente vs Categoría de Temperatura**
```
χ² = 145,678.23
p-value = 0.0000 (< 0.05)
df = 12
is_significant = True ✅

→ INTERPRETACIÓN: Existe una relación significativa entre el continente 
  y la distribución de temperaturas. La temperatura NO es independiente 
  de la ubicación geográfica.
```

**Test: Estación vs Categoría de Temperatura**
```
χ² = 298,456.78
p-value = 0.0000 (< 0.05)
df = 6
is_significant = True ✅

→ INTERPRETACIÓN: Las temperaturas varían significativamente por estación,
  como era de esperarse. Strong evidence of seasonal patterns.
```

---

## 🔬 Detalles Técnicos

### Dependencias Agregadas

**Python:**
- `scipy.stats.chi2`: Para cálculo de p-values en pruebas chi-cuadrado
- Ya incluido en requirements.txt estándar de scipy

**PySpark:**
- `pyspark.sql.functions.corr`: Correlación de Pearson
- `pyspark.sql.functions.skewness`: Asimetría
- `pyspark.sql.functions.kurtosis`: Curtosis
- `pyspark.sql.functions.percentile_approx`: Cuartiles

### Performance

**Tiempo de procesamiento estimado (dataset completo):**
- Correlación: ~2-3 minutos
- Descriptive stats: ~3-5 minutos
- Chi-square tests: ~5-7 minutos
- **Total EDA: ~10-15 minutos adicionales**

**Tamaño de archivos:**
- `correlation_matrix.parquet`: <1 MB
- `descriptive_stats.parquet`: <1 MB
- `chi_square_tests.parquet`: <1 MB
- **Total: ~3 MB adicionales**

### Optimizaciones Aplicadas

1. **Correlación:**
   - Uso de `df.stat.corr()` nativo de Spark (optimizado)
   - Solo calcula triángulo superior, luego duplica para simetría

2. **Descriptive stats:**
   - Una sola pasada sobre los datos con `.agg()` múltiple
   - Uso de `percentile_approx()` para cuartiles (más rápido que exacto)

3. **Chi-square:**
   - Cálculo manual usando tablas de contingencia
   - Uso de scipy solo para p-value (no requiere datos completos)

---

## 🎯 Casos de Uso

### 1. Análisis Pre-Modelado

**Objetivo:** Entender relaciones antes de construir modelos ML

**Acciones:**
- Revisar correlaciones para detectar multicolinealidad
- Verificar distribuciones para decidir transformaciones
- Identificar variables dependientes con chi-cuadrado

### 2. Validación de Hipótesis

**Objetivo:** Confirmar suposiciones sobre el dataset

**Acciones:**
- ¿Las temperaturas están aumentando globalmente? → Ver `year` ↔ `avg_temp`
- ¿Hay diferencias continentales? → Ver chi-square continente vs temp
- ¿Las estaciones afectan temperatura? → Ver chi-square estación vs temp

### 3. Detección de Anomalías Estadísticas

**Objetivo:** Encontrar valores atípicos o patrones inusuales

**Acciones:**
- Usar IQR para detectar outliers: valores fuera de [Q1 - 1.5×IQR, Q3 + 1.5×IQR]
- Verificar skewness para detectar distribuciones asimétricas
- Analizar kurtosis para identificar eventos extremos

### 4. Reportes Científicos

**Objetivo:** Generar estadísticas para publicaciones

**Acciones:**
- Exportar tablas de estadísticas descriptivas
- Incluir heatmaps de correlación en papers
- Reportar resultados de tests chi-cuadrado con interpretación

---

## 📝 Notas Importantes

### Arquitectura Respetada ✅

- ✅ **Todo el procesamiento en Spark:** Funciones pesadas en `spark_processor.py`
- ✅ **Dashboard solo visualiza:** No realiza cálculos estadísticos complejos
- ✅ **HDFS como fuente única:** Archivos procesados almacenados en HDFS
- ✅ **Formato Parquet:** Eficiente, columnar, comprimido

### Limitaciones Conocidas

1. **Chi-square manual:**
   - Solo funciona con datasets moderados (~millones de filas)
   - Para datasets muy grandes (>100M), considerar muestreo

2. **Percentiles aproximados:**
   - Uso de `percentile_approx()` en lugar de exactos
   - Error típico: <1% para percentiles

3. **Categorización simple:**
   - Temperatura dividida en 3 categorías (Cold/Moderate/Hot)
   - Podría mejorarse con más categorías o clustering

---

## 🔮 Mejoras Futuras

- [ ] Agregar test de normalidad (Shapiro-Wilk, Kolmogorov-Smirnov)
- [ ] Implementar ANOVA para comparar medias entre grupos
- [ ] Agregar análisis de componentes principales (PCA)
- [ ] Incluir análisis de outliers multivariado (Mahalanobis distance)
- [ ] Generar reporte PDF automático con todos los EDA
- [ ] Agregar comparaciones temporales (¿cambio en correlaciones por década?)

---

## 📚 Referencias

### Código Fuente
- **Spark Processor:** `Tools/src/climaxtreme/preprocessing/spark_processor.py` (líneas 757-1044)
- **Dashboard:** `Tools/src/climaxtreme/dashboard/app.py` (líneas ~1350-1540)
- **Script:** `scripts/process_full_dataset.ps1`

### Documentación Estadística
- **Pearson Correlation:** https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
- **Chi-Square Test:** https://en.wikipedia.org/wiki/Chi-squared_test
- **Descriptive Statistics:** https://en.wikipedia.org/wiki/Descriptive_statistics
- **Skewness & Kurtosis:** https://en.wikipedia.org/wiki/Skewness

### Librerías Utilizadas
- **PySpark SQL Functions:** https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/functions.html
- **SciPy Stats:** https://docs.scipy.org/doc/scipy/reference/stats.html
- **Plotly Heatmap:** https://plotly.com/python/heatmaps/

---

**Implementado:** Octubre 2024  
**Versión:** 1.0  
**Autor:** Equipo climaXtreme
