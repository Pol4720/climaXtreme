# climaXtreme 🌡️

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Climate analysis and extreme event modeling using Hadoop and PySpark for large-scale climate data processing. This project provides a comprehensive toolkit for ingesting, processing, analyzing, and visualizing Berkeley Earth climate data.

## 🌟 Features

- **Data Ingestion**: Automated download and ingestion of Berkeley Earth climate datasets
- **Big Data Processing**: PySpark-based preprocessing for large-scale climate data
- **Advanced Analytics**: Temperature trend analysis, seasonal patterns, and extreme event detection
- **Interactive Dashboard**: Streamlit-based web interface for data exploration and visualization
- **Machine Learning**: Baseline models for climate prediction with ensemble methods
- **Comprehensive Testing**: Full test coverage with unit and integration tests
- **CI/CD Pipeline**: Automated testing, linting, and deployment workflows

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- Java 11 (required for PySpark)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Pol4720/climaXtreme.git
   cd climaXtreme
   ```

2. **Run the setup script**
   ```bash
   chmod +x scripts/setup.sh
   ./scripts/setup.sh
   ```

3. **Activate the virtual environment**
   ```bash
   source venv/bin/activate
   ```

### Manual Installation

If you prefer manual installation:

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Create directory structure
mkdir -p data/{raw,processed,output} logs models
```

## 📊 Usage

### Command Line Interface

climaXtreme provides a comprehensive CLI for all operations:

```bash
# Download climate data
climaxtreme ingest --start-year 2020 --end-year 2023

# Preprocess data using PySpark (local files)
climaxtreme preprocess --input-dir data/raw --output-dir data/processed

# Preprocess data directly from HDFS (CSV or Berkeley Earth .txt)
# Example with the sample CSV uploaded to HDFS (see HDFS section below)
climaxtreme preprocess \
   --input-path hdfs://climaxtreme-namenode:9000/data/climaxtreme/GlobalLandTemperaturesByCity_sample.csv \
   --output-path hdfs://climaxtreme-namenode:9000/data/climaxtreme/processed \
   --format city-csv

# Run analysis
climaxtreme analyze --data-path data/processed --analysis-type both

# Launch interactive dashboard (via Docker - recommended)
docker-compose -f infra/docker-compose.yml up -d dashboard

# Or run locally for development
climaxtreme dashboard --host localhost --port 8501
```

### Python API

```python
from climaxtreme import DataIngestion, SparkPreprocessor, HeatmapAnalyzer

# Data ingestion
ingestion = DataIngestion("data/raw")
files = ingestion.download_berkeley_earth_data(2020, 2023)

# Preprocessing with PySpark
with SparkPreprocessor() as preprocessor:
    results = preprocessor.process_directory("data/raw", "data/processed")

# Generate heatmaps
analyzer = HeatmapAnalyzer()
heatmap_path = analyzer.generate_global_heatmap("data/processed", "data/output")
```

### HDFS (Local) with Docker on Windows

The project supports reading/writing directly to HDFS. For local development on Windows, a lightweight HDFS cluster is provided via Docker.

**📖 Detailed guide**: See [PROCESSING_GUIDE.md](PROCESSING_GUIDE.md) for complete instructions and [HDFS_SETUP_GUIDE.md](HDFS_SETUP_GUIDE.md) for troubleshooting.

#### Quick Start - Process Full Dataset (RECOMMENDED)

**One-command pipeline** that loads, processes, and downloads the complete dataset (500+ MB):

```powershell
# From repo root, in PowerShell
.\scripts\process_full_dataset.ps1
```

This automated script will:
1. ✓ Start HDFS cluster (NameNode + DataNode)
2. ✓ Upload **complete** dataset to HDFS (~500 MB)
3. ✓ Process with PySpark (~8.6M records)
4. ✓ Download results to `DATA/processed/`
5. ✓ Ready for dashboard visualization

**Time**: ~10-25 minutes depending on hardware

#### Manual Steps (Alternative)

**Step 1: Upload Complete Dataset to HDFS**

```powershell
# Upload FULL dataset (500+ MB, ~8.6M records)
.\scripts\hdfs_setup_and_load.ps1 -FullFile

# OR upload a sample for testing (faster)
.\scripts\hdfs_setup_and_load.ps1 -Head 1000000  # 1M rows
```

**Step 2: Process with PySpark**

```powershell
docker exec -it climaxtreme-processor python -m climaxtreme.cli preprocess `
   --input-path hdfs://climaxtreme-namenode:9000/data/climaxtreme/GlobalLandTemperaturesByCity.csv `
   --output-path hdfs://climaxtreme-namenode:9000/data/climaxtreme/processed `
   --format city-csv
```

**Step 3: Launch Dashboard**

```powershell
# Dashboard already running from docker-compose up -d
# Just open: http://localhost:8501

# Or start it separately:
cd infra
docker-compose up -d dashboard
```

Access at: http://localhost:8501
- Select "HDFS (Recommended)" in sidebar
- Configure: Host=namenode, Port=9000, Path=/data/climaxtreme/processed

#### Verification & Monitoring

```powershell
# Check system status
.\scripts\check_status.ps1

# NameNode Web UI (browse HDFS files)
# http://localhost:9870

# List files in HDFS
docker exec climaxtreme-namenode hdfs dfs -ls -h /data/climaxtreme
```

#### What Gets Processed?

The complete processing pipeline generates:
- **monthly.parquet**: Monthly temperature aggregations by city/country
- **yearly.parquet**: Yearly temperature aggregations
- **anomalies.parquet**: Detected temperature anomalies with z-scores

All files include: avg/min/max temperature, record counts, uncertainty metrics

#### Stop HDFS

```powershell
docker compose -f infra\docker-compose.yml down
```

**Prerequisites:**
- Docker Desktop installed and **running** (WSL2 backend recommended)
- ~8GB RAM available for processing
- ~2GB disk space for Docker volumes
- Internet access to pull Hadoop images (first run only)

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# Access dashboard at http://localhost:8501
# Access Jupyter at http://localhost:8888
```

## 📁 Project Structure

```
climaXtreme/
├── src/climaxtreme/           # Main package
│   ├── data/                  # Data ingestion and validation
│   ├── preprocessing/         # PySpark data processing
│   ├── analysis/             # Analysis modules (heatmaps, time series)
│   ├── dashboard/            # Streamlit dashboard
│   ├── ml/                   # Machine learning models
│   └── utils/                # Utilities and configuration
├── tests/                    # Test suite
│   ├── unit/                 # Unit tests
│   └── integration/          # Integration tests
├── docs/                     # Documentation
├── notebooks/                # Jupyter notebooks
├── scripts/                  # Setup and utility scripts
├── configs/                  # Configuration files
├── data/                     # Data directories
│   ├── raw/                  # Raw data
│   ├── processed/            # Processed data
│   └── output/               # Analysis outputs
├── .github/workflows/        # CI/CD workflows
├── requirements.txt          # Python dependencies
├── pyproject.toml           # Package configuration
├── infra/                   # Local HDFS (Docker Compose)
│   └── docker-compose.yml   # NameNode/DataNode
└── scripts/                 # Utility scripts (incl. Windows PowerShell for HDFS)
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/climaxtreme --cov-report=html

# Run specific test categories
pytest tests/unit/           # Unit tests only
pytest tests/integration/    # Integration tests only
```

## 🔧 Development

### Code Quality

The project uses several tools to maintain code quality:

```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/climaxtreme/

# Pre-commit hooks (run automatically on commit)
pre-commit install
```

### Configuration

Configuration is managed through YAML files in the `configs/` directory:

```yaml
# configs/default_config.yml
data_dir: "data"
spark_app_name: "climaXtreme"
dashboard_port: 8501
# ... more settings
```

## 📈 Analysis Capabilities

### Temperature Trend Analysis
- Long-term global temperature trends
- Statistical significance testing
- Seasonal decomposition
- Polynomial trend fitting

### Heatmap Generation
- Global temperature heatmaps
- Seasonal pattern visualization
- Temperature anomaly maps
- Regional comparison heatmaps

### Extreme Event Detection
- Percentile-based threshold detection
- Hot and cold extreme identification
- Temporal distribution analysis
- Frequency trend analysis

### Machine Learning
- Multiple baseline models (Linear, Ridge, Random Forest, Gradient Boosting)
- Ensemble methods with uncertainty quantification
- Time series cross-validation
- Feature importance analysis

## 🌐 Dashboard Features

The Streamlit dashboard provides:

- **Data Overview**: Dataset statistics and quality metrics
- **Temperature Trends**: Interactive trend visualization with statistical analysis
- **Heatmaps**: Dynamic temperature and anomaly heatmaps
- **Seasonal Analysis**: Monthly climatology and seasonal comparisons
- **Extreme Events**: Interactive extreme event detection and visualization

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest`)
5. Run code quality checks (`black`, `flake8`, `mypy`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Berkeley Earth** for providing high-quality climate data
- **Apache Spark** for big data processing capabilities
- **Streamlit** for the interactive dashboard framework
- **scikit-learn** for machine learning algorithms

## 📞 Support

For questions, issues, or contributions:

- Open an issue on [GitHub Issues](https://github.com/Pol4720/climaXtreme/issues)
- Check the documentation in the `docs/` directory
- Review the example notebooks in `notebooks/`

---

**climaXtreme** - Empowering climate research through big data analytics 🌍
