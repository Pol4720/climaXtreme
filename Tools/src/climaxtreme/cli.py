"""
Command-line interface for climaXtreme.
"""

from typing import Optional
import click
from pathlib import Path

from .data import DataIngestion
from .preprocessing import SparkPreprocessor
from .analysis import HeatmapAnalyzer, TimeSeriesAnalyzer
from .dashboard.app import run_dashboard
try:
    from .utils.config import default_dataset_dir as _default_dataset_dir
    _DEFAULT_DATA_DIR = _default_dataset_dir()
except Exception:
    from pathlib import Path as _Path
    _DEFAULT_DATA_DIR = _Path("DATA")


@click.group()
@click.version_option()
def main() -> None:
    """climaXtreme: Climate analysis and extreme event modeling."""
    pass


@main.command()
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("data/raw"),
    help="Directory to store downloaded data",
)
@click.option(
    "--start-year", type=int, default=2020, help="Start year for data download"
)
@click.option("--end-year", type=int, default=2023, help="End year for data download")
def ingest(output_dir: Path, start_year: int, end_year: int) -> None:
    """Download and ingest Berkeley Earth climate data."""
    click.echo(f"Ingesting data from {start_year} to {end_year}...")
    
    ingestion = DataIngestion(str(output_dir))
    ingestion.download_berkeley_earth_data(start_year, end_year)
    
    click.echo(f"Data successfully downloaded to {output_dir}")


@main.command()
@click.option(
    "--input-dir",
    type=click.Path(exists=True, path_type=Path),
    default=Path("data/raw"),
    help="Directory containing raw data (local). Ignored if --input-path is provided.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("data/processed"),
    help="Directory to store processed data (local). Ignored if --output-path is provided.",
)
@click.option(
    "--input-path",
    type=str,
    default=None,
    help="Input path (local or hdfs://). Can be file, directory, or glob.",
)
@click.option(
    "--output-path",
    type=str,
    default=None,
    help="Output base path (local or hdfs://) for Parquet outputs.",
)
@click.option(
    "--format",
    type=click.Choice(["auto", "berkeley-txt", "city-csv"]),
    default="auto",
    help="Input data format. auto=detect by extension",
)
def preprocess(input_dir: Path, output_dir: Path, input_path: Optional[str], output_path: Optional[str], format: str) -> None:
    """Preprocess climate data using PySpark (local FS or HDFS)."""
    click.echo("Starting data preprocessing...")

    preprocessor = SparkPreprocessor()
    if input_path is not None:
        base_out = output_path if output_path is not None else str(output_dir)
        artifacts = preprocessor.process_path(input_path, base_out, fmt=format)
        click.echo(f"Data preprocessing completed. Outputs: {artifacts}")
    else:
        preprocessor.process_directory(str(input_dir), str(output_dir))
        click.echo(f"Data preprocessing completed. Output: {output_dir}")


@main.command()
@click.option(
    "--data-path",
    type=click.Path(exists=True, path_type=Path),
    default=Path("data/processed"),
    help="Path to processed data",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("data/output"),
    help="Directory to store analysis results",
)
@click.option(
    "--analysis-type",
    type=click.Choice(["heatmap", "timeseries", "both"]),
    default="both",
    help="Type of analysis to perform",
)
def analyze(data_path: Path, output_dir: Path, analysis_type: str) -> None:
    """Run climate data analysis."""
    click.echo(f"Running {analysis_type} analysis...")
    
    if analysis_type in ["heatmap", "both"]:
        heatmap_analyzer = HeatmapAnalyzer()
        heatmap_analyzer.generate_global_heatmap(str(data_path), str(output_dir))
        click.echo("Heatmap analysis completed")
    
    if analysis_type in ["timeseries", "both"]:
        ts_analyzer = TimeSeriesAnalyzer()
        ts_analyzer.analyze_temperature_trends(str(data_path), str(output_dir))
        click.echo("Time series analysis completed")
    
    click.echo(f"Analysis results saved to {output_dir}")


@main.command()
@click.option(
    "--host", default="localhost", help="Host to run the dashboard on"
)
@click.option("--port", default=8501, help="Port to run the dashboard on")
@click.option(
    "--data-dir",
    type=click.Path(exists=True, path_type=Path),
    default=_DEFAULT_DATA_DIR,
    help="Directory containing climate data (defaults to repo-root/DATA)",
)
def dashboard(host: str, port: int, data_dir: Path) -> None:
    """Launch the Streamlit dashboard."""
    click.echo(f"Starting dashboard at http://{host}:{port}")
    run_dashboard(host, port, str(data_dir))


if __name__ == "__main__":
    main()