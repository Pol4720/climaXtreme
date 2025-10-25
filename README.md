# climaXtreme 🌡️

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Climate analysis and extreme event modeling using Hadoop and PySpark for large-scale climate data processing. This project provides a comprehensive toolkit for ingesting, processing, analyzing, and visualizing Berkeley Earth climate data with Big Data technologies.

## 🌟 Features

- **Big Data Processing**: PySpark-based preprocessing with HDFS for large-scale climate data (~8.6M records)
- **Advanced Analytics**: Temperature trend analysis, seasonal patterns, extreme event detection, and exploratory data analysis (EDA)
- **Interactive Dashboard**: Streamlit-based web interface running in Docker with real-time HDFS data access
- **Machine Learning**: Baseline models for climate prediction with ensemble methods
- **Comprehensive EDA**: Correlation matrices, descriptive statistics, and chi-square independence tests
- **Geographic Visualizations**: Interactive maps by region and continent with 16 regions and 7 continents
- **Production Ready**: Full Docker deployment with HDFS, Spark processor, and dashboard

## 🚀 Quick Start

### Prerequisites

- **Docker Desktop** installed and running (required)
- ~8GB RAM available
- ~2GB disk space for Docker volumes

### Setup and Execution

**📖 For complete setup and execution instructions, see:**

## **➡️ [HDFS_SETUP_GUIDE.md](HDFS_SETUP_GUIDE.md)** ⬅️

The guide includes:
- ✅ Quick start (2 commands to run everything)
- ✅ Step-by-step instructions for Windows and Linux/macOS
- ✅ Full pipeline automation scripts
- ✅ Dashboard configuration
- ✅ Troubleshooting and common issues
- ✅ Development workflows

### Quick Command Reference

**Windows (PowerShell):**
```powershell
# 1. Start all containers
cd infra
docker-compose up -d

# 2. Process complete dataset
cd ..
.\scripts\windows\process_full_dataset.ps1 -SkipDownload

# 3. Open dashboard: http://localhost:8501
```

**Linux/macOS (Bash):**
```bash
# 1. Start all containers
cd infra
docker-compose up -d

# 2. Process complete dataset
cd ..
bash scripts/linux/process_full_dataset.sh --skip-download

# 3. Open dashboard: http://localhost:8501
```

## 📊 System Architecture

```
Docker Network (hdfs)
├── namenode:9870 (HDFS Web UI)
├── namenode:9000 (HDFS API)
├── datanode (Storage)
├── processor (PySpark)
└── dashboard:8501 (Streamlit)
```

## 📁 Generated Datasets

The processing pipeline generates 11 Parquet files (~150 MB total):

**Climate Aggregations (8 files):**
- `monthly.parquet` - Monthly temperature trends
- `yearly.parquet` - Yearly aggregations with trend lines
- `anomalies.parquet` - Temperature anomaly detection
- `climatology.parquet` - Long-term climatology
- `seasonal.parquet` - Seasonal patterns
- `extreme_thresholds.parquet` - Extreme event thresholds
- `regional.parquet` - Analysis by 16 geographic regions
- `continental.parquet` - Analysis by 7 continents

**Exploratory Data Analysis (3 files):**
- `correlation_matrix.parquet` - Pearson correlation matrix
- `descriptive_stats.parquet` - 11 statistical measures per variable
- `chi_square_tests.parquet` - Independence tests (continent, season, period vs temperature)

## 🎨 Dashboard Features

- **Temperature Trends**: Interactive trend visualization with statistical analysis
- **Heatmaps**: Dynamic temperature and anomaly heatmaps
- **Seasonal Analysis**: Monthly climatology and seasonal comparisons
- **Extreme Events**: Interactive extreme event detection
- **Regional Analysis**: Analysis by region with interactive world map 🗺️
- **Continental Analysis**: Analysis by continent with global bubble map 🌍
- **Exploratory Analysis (EDA)**: Correlations, descriptive stats, and chi-square tests 📊

## 📁 Project Structure

```
climaXtreme/
├── Tools/src/climaxtreme/     # Main package
│   ├── data/                   # Data ingestion and validation
│   ├── preprocessing/          # PySpark data processing
│   ├── analysis/               # Analysis modules (heatmaps, time series)
│   ├── dashboard/              # Streamlit dashboard
│   ├── ml/                     # Machine learning models
│   └── utils/                  # Utilities and configuration
├── infra/                      # Docker infrastructure
│   ├── docker-compose.yml      # HDFS + Processor + Dashboard
│   └── Dockerfile.processor    # Processor image
├── scripts/                    # Utility scripts
│   ├── windows/                # PowerShell scripts for Windows
│   └── linux/                  # Bash scripts for Linux/macOS
├── DATA/                       # Data directories
│   ├── GlobalLandTemperaturesByCity.csv  # Raw dataset
│   └── processed/              # Processed Parquet files
└── docs/                       # Additional documentation
    ├── HDFS_SETUP_GUIDE.md     # Complete setup guide (START HERE)
    ├── PARQUETS.md             # Parquet file schemas
    ├── EDA_IMPLEMENTATION.md   # EDA documentation
    └── scripts/README.md       # Script documentation
```

## 📚 Documentation

- **[HDFS_SETUP_GUIDE.md](HDFS_SETUP_GUIDE.md)** - Complete setup and execution guide (Windows & Linux)
- **[scripts/README.md](scripts/README.md)** - Detailed script documentation
- **[PARQUETS.md](PARQUETS.md)** - Parquet file structures and schemas
- **[EDA_IMPLEMENTATION.md](EDA_IMPLEMENTATION.md)** - Exploratory Data Analysis guide

## 🔧 Development

For development workflows (modifying code, rebuilding containers, etc.), see the "Development" section in [HDFS_SETUP_GUIDE.md](HDFS_SETUP_GUIDE.md).

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/climaxtreme --cov-report=html
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Berkeley Earth** for providing high-quality climate data
- **Apache Spark** for big data processing capabilities
- **Apache Hadoop** for distributed storage (HDFS)
- **Streamlit** for the interactive dashboard framework

## 📞 Support

For questions, issues, or contributions:

- Open an issue on [GitHub Issues](https://github.com/Pol4720/climaXtreme/issues)
- Check the [HDFS_SETUP_GUIDE.md](HDFS_SETUP_GUIDE.md) for setup help
- Review the [scripts/README.md](scripts/README.md) for script documentation

---

**climaXtreme** - Empowering climate research through big data analytics 🌍
