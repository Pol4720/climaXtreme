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
    type=click.Path(path_type=Path),
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
        # Validate input_dir exists only when it's actually used
        if not input_dir.exists():
            raise click.ClickException(f"Input directory '{input_dir}' does not exist.")
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
def dashboard(host: str, port: int) -> None:
    """Launch the Streamlit dashboard with HDFS + Local mode support.
    
    The dashboard will start with both HDFS and Local Files modes available.
    You can select the data source from the sidebar.
    """
    click.echo(f"Starting dashboard at http://{host}:{port}")
    click.echo("Dashboard supports both HDFS and Local Files modes")
    click.echo("Select your data source from the sidebar once the dashboard loads")
    run_dashboard(host, port)


# ============================================================================
# Synthetic Data Generation Commands
# ============================================================================

@main.command("generate-synthetic")
@click.option(
    "--input-path",
    type=str,
    default="DATA/GlobalLandTemperaturesByCity.csv",
    help="Input CSV path (local or hdfs://)",
)
@click.option(
    "--output-path",
    type=str,
    default="DATA/synthetic",
    help="Output base path for synthetic data",
)
@click.option(
    "--seed",
    type=int,
    default=42,
    help="Random seed for reproducibility",
)
@click.option(
    "--no-hourly",
    is_flag=True,
    default=False,
    help="Disable hourly interpolation (keep daily resolution)",
)
@click.option(
    "--no-storms",
    is_flag=True,
    default=False,
    help="Disable storm generation",
)
@click.option(
    "--sample-fraction",
    type=float,
    default=None,
    help="Sample fraction (0-1) for testing with smaller data",
)
def generate_synthetic(
    input_path: str, 
    output_path: str, 
    seed: int, 
    no_hourly: bool,
    no_storms: bool,
    sample_fraction: Optional[float]
) -> None:
    """Generate synthetic climate data from original dataset.
    
    This command reads the original temperature CSV and generates:
    - Hourly temperature interpolation with diurnal cycles
    - Precipitation, wind, humidity, pressure data
    - Extreme weather events (heatwaves, coldsnaps, floods)
    - Storm events with tracking data
    - Weather alerts
    
    Output is written as partitioned Parquet files.
    """
    from .preprocessing.spark import (
        SyntheticClimateGenerator, 
        SyntheticConfig,
        read_city_temperature_csv_path
    )
    from .preprocessing.spark.spark_session_manager import SparkSessionManager
    
    click.echo("=" * 60)
    click.echo("🌪️  climaXtreme Synthetic Data Generator")
    click.echo("=" * 60)
    click.echo(f"📁 Input:  {input_path}")
    click.echo(f"📂 Output: {output_path}")
    click.echo(f"🎲 Seed:   {seed}")
    click.echo(f"⏰ Hourly: {'No' if no_hourly else 'Yes'}")
    click.echo(f"🌀 Storms: {'No' if no_storms else 'Yes'}")
    if sample_fraction:
        click.echo(f"📊 Sample: {sample_fraction * 100:.1f}%")
    click.echo("=" * 60)
    
    # Initialize Spark
    click.echo("\n🔧 Initializing Spark session...")
    session_manager = SparkSessionManager("climaXtreme-SyntheticGenerator")
    spark = session_manager.get_spark_session()
    
    try:
        # Read input data
        click.echo(f"\n📖 Reading input data from {input_path}...")
        input_df = read_city_temperature_csv_path(spark, input_path)
        
        initial_count = input_df.count()
        click.echo(f"   Loaded {initial_count:,} records")
        
        # Apply sampling if requested
        if sample_fraction:
            input_df = input_df.sample(fraction=sample_fraction, seed=seed)
            sampled_count = input_df.count()
            click.echo(f"   Sampled to {sampled_count:,} records")
        
        # Configure generator
        config = SyntheticConfig(
            seed=seed,
            hourly_interpolation=not no_hourly
        )
        
        # Create generator and run pipeline
        click.echo("\n🏭 Running synthetic generation pipeline...")
        generator = SyntheticClimateGenerator(spark, config)
        
        synthetic_df, storm_tracks = generator.generate_full_synthetic_dataset(
            input_df,
            generate_storms=not no_storms
        )
        
        final_count = synthetic_df.count()
        click.echo(f"   Generated {final_count:,} synthetic records")
        
        # Write outputs
        click.echo(f"\n💾 Writing outputs to {output_path}...")
        
        # Main synthetic data
        main_output = f"{output_path}/synthetic_hourly.parquet"
        generator.write_to_parquet(synthetic_df, main_output)
        click.echo(f"   ✅ Synthetic hourly data: {main_output}")
        
        # Storm tracks
        if storm_tracks is not None and storm_tracks.count() > 0:
            storm_output = f"{output_path}/storm_tracks.parquet"
            generator.write_storm_tracks(storm_tracks, storm_output)
            click.echo(f"   ✅ Storm tracks: {storm_output}")
        
        # Alerts history
        from pyspark.sql import functions as F
        alerts_df = synthetic_df.filter(F.col("alert_active") == True)
        alerts_count = alerts_df.count()
        if alerts_count > 0:
            alerts_output = f"{output_path}/alerts_history.parquet"
            alerts_df.write.mode("overwrite").partitionBy("year", "month").parquet(alerts_output)
            click.echo(f"   ✅ Alerts history ({alerts_count:,} alerts): {alerts_output}")
        
        # Events summary
        events_df = synthetic_df.filter(F.col("event_type") != "NORMAL")
        events_count = events_df.count()
        if events_count > 0:
            events_output = f"{output_path}/event_summary.parquet"
            events_df.write.mode("overwrite").partitionBy("year", "event_type").parquet(events_output)
            click.echo(f"   ✅ Events summary ({events_count:,} events): {events_output}")
        
        click.echo("\n" + "=" * 60)
        click.echo("✅ Synthetic data generation completed successfully!")
        click.echo("=" * 60)
        
        # Show summary statistics
        click.echo("\n📊 Summary Statistics:")
        click.echo(f"   Total records: {final_count:,}")
        click.echo(f"   Alert events:  {alerts_count:,}")
        click.echo(f"   Extreme events: {events_count:,}")
        if storm_tracks is not None:
            click.echo(f"   Storm track points: {storm_tracks.count():,}")
        
    finally:
        session_manager.stop_spark_session()
        click.echo("\n🛑 Spark session stopped")


@main.command("train-intensity-model")
@click.option(
    "--data-path",
    type=str,
    default="DATA/synthetic/synthetic_hourly.parquet",
    help="Path to synthetic data",
)
@click.option(
    "--model-output",
    type=str,
    default="models/intensity_model",
    help="Output path for trained model",
)
@click.option(
    "--sample-size",
    type=int,
    default=100000,
    help="Sample size for training (to fit in memory)",
)
def train_intensity_model(data_path: str, model_output: str, sample_size: int) -> None:
    """Train intensity prediction model on synthetic data.
    
    Uses ensemble of models (Linear, Ridge, RandomForest) to predict
    event intensity from weather features.
    """
    import pandas as pd
    from pathlib import Path
    
    click.echo("=" * 60)
    click.echo("🧠 Training Intensity Prediction Model")
    click.echo("=" * 60)
    
    # Read sample of data
    click.echo(f"\n📖 Loading data from {data_path}...")
    
    try:
        df = pd.read_parquet(data_path)
        click.echo(f"   Loaded {len(df):,} records")
        
        # Sample if needed
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
            click.echo(f"   Sampled to {len(df):,} records")
        
        # Filter to events only for better signal
        events_df = df[df['event_type'] != 'NORMAL'].copy()
        click.echo(f"   Training on {len(events_df):,} event records")
        
        if len(events_df) < 100:
            click.echo("⚠️  Not enough event data for training. Using all data.")
            events_df = df.copy()
        
        # Prepare features
        feature_cols = [
            'hour', 'month', 'lat_decimal', 'lon_decimal',
            'temperature_hourly', 'humidity_pct', 'pressure_hpa',
            'wind_speed_kmh', 'rain_mm', 'anomaly_score'
        ]
        
        # Check available columns
        available_features = [c for c in feature_cols if c in events_df.columns]
        click.echo(f"   Using features: {available_features}")
        
        # Handle missing values
        events_df = events_df.dropna(subset=available_features + ['event_intensity'])
        
        if len(events_df) < 50:
            raise click.ClickException("Not enough valid data for training after dropping NaN values")
        
        X = events_df[available_features]
        y = events_df['event_intensity']
        
        # Train model using ClimatePredictor
        from .ml.predictor import ClimatePredictor
        
        click.echo("\n🏋️ Training ensemble model...")
        
        # We need to create a compatible training dataframe
        train_df = events_df.copy()
        train_df['avg_temperature'] = train_df['temperature_hourly']
        train_df['year'] = train_df.get('year', 2020)
        
        predictor = ClimatePredictor(models=['linear', 'ridge', 'random_forest'])
        
        # Custom train with intensity target
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import Ridge
        from sklearn.metrics import mean_squared_error, r2_score
        import numpy as np
        import joblib
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest (good for this type of data)
        click.echo("   Training RandomForest...")
        rf_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = rf_model.predict(X_test_scaled)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        click.echo(f"\n📈 Model Performance:")
        click.echo(f"   RMSE: {rmse:.4f}")
        click.echo(f"   R²:   {r2:.4f}")
        
        # Feature importance
        click.echo("\n📊 Feature Importance:")
        importance = dict(zip(available_features, rf_model.feature_importances_))
        for feat, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
            click.echo(f"   {feat}: {imp:.4f}")
        
        # Save model
        output_path = Path(model_output)
        output_path.mkdir(parents=True, exist_ok=True)
        
        model_file = output_path / "intensity_rf_model.joblib"
        scaler_file = output_path / "scaler.joblib"
        metadata_file = output_path / "metadata.joblib"
        
        joblib.dump(rf_model, model_file)
        joblib.dump(scaler, scaler_file)
        joblib.dump({
            'features': available_features,
            'rmse': rmse,
            'r2': r2,
            'feature_importance': importance
        }, metadata_file)
        
        click.echo(f"\n💾 Model saved to {model_output}/")
        click.echo("   ✅ intensity_rf_model.joblib")
        click.echo("   ✅ scaler.joblib")
        click.echo("   ✅ metadata.joblib")
        
        click.echo("\n" + "=" * 60)
        click.echo("✅ Model training completed successfully!")
        click.echo("=" * 60)
        
    except Exception as e:
        raise click.ClickException(f"Training failed: {e}")


@main.command("stream-demo")
@click.option(
    "--duration",
    type=int,
    default=60,
    help="Duration of streaming simulation in seconds",
)
@click.option(
    "--output-path",
    type=str,
    default="DATA/streaming",
    help="Output directory for streaming data",
)
@click.option(
    "--n-cities",
    type=int,
    default=10,
    help="Number of cities to simulate",
)
@click.option(
    "--rate",
    type=int,
    default=10,
    help="Records per second to generate",
)
def stream_demo(duration: int, output_path: str, n_cities: int, rate: int) -> None:
    """Run streaming simulation demo for real-time climate data.
    
    This command simulates real-time weather data generation,
    useful for testing real-time dashboards and alert systems.
    
    The simulator generates:
    - Temperature with Markov chain state transitions
    - Precipitation following gamma distribution
    - Wind speed with Weibull distribution
    - Weather alerts based on thresholds
    
    Data is written to JSON files that can be consumed by
    Spark Structured Streaming.
    """
    from .streaming import StreamingConfig, StreamingSimulator
    
    click.echo("=" * 60)
    click.echo("🌊 climaXtreme Streaming Demo")
    click.echo("=" * 60)
    click.echo(f"⏱️  Duration: {duration} seconds")
    click.echo(f"📂 Output:   {output_path}")
    click.echo(f"🏙️  Cities:   {n_cities}")
    click.echo(f"📊 Rate:     {rate} records/second")
    click.echo("=" * 60)
    
    config = StreamingConfig(
        output_path=output_path,
        n_cities=n_cities,
        records_per_second=rate
    )
    
    simulator = StreamingSimulator(config)
    
    click.echo("\n🚀 Starting streaming simulation...")
    click.echo("   Press Ctrl+C to stop early\n")
    
    try:
        stats = simulator.run_simulation(duration, output_path)
        
        click.echo("\n" + "=" * 60)
        click.echo("📊 Simulation Statistics:")
        click.echo("=" * 60)
        for key, value in stats.items():
            click.echo(f"   {key}: {value}")
        
        click.echo("\n✅ Streaming simulation completed!")
        click.echo(f"   Data written to: {output_path}")
        
    except KeyboardInterrupt:
        click.echo("\n\n⚠️  Simulation interrupted by user")
    except Exception as e:
        raise click.ClickException(f"Streaming demo failed: {e}")


if __name__ == "__main__":
    main()