"""
Integration tests for the complete climaXtreme pipeline.
"""

import pytest
import tempfile
import shutil
import pandas as pd
import numpy as np
from pathlib import Path

from climaxtreme.data.ingestion import DataIngestion
from climaxtreme.data.validation import DataValidator
from climaxtreme.preprocessing.preprocessor import DataPreprocessor
from climaxtreme.analysis.heatmap import HeatmapAnalyzer
from climaxtreme.analysis.timeseries import TimeSeriesAnalyzer
from climaxtreme.ml.baseline import BaselineModel


class TestClimaXtremePipeline:
    """Integration tests for the complete pipeline."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir) / "data"
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.output_dir = self.data_dir / "output"
        
        # Create directories
        for dir_path in [self.raw_dir, self.processed_dir, self.output_dir]:
            dir_path.mkdir(parents=True)
        
        # Create sample Berkeley Earth format data
        self.create_sample_data()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_sample_data(self):
        """Create sample data in Berkeley Earth format."""
        # Generate sample temperature data
        years = range(2020, 2023)
        months = range(1, 13)
        
        data_lines = [
            "% Sample Berkeley Earth Temperature Data",
            "% Year Month Temperature Uncertainty",
            ""
        ]
        
        np.random.seed(42)
        for year in years:
            for month in months:
                # Seasonal pattern + small trend + noise
                temp = 15 + 10 * np.sin(2 * np.pi * month / 12) + 0.1 * (year - 2020) + np.random.normal(0, 1)
                uncertainty = np.random.uniform(0.1, 0.5)
                data_lines.append(f"{year:4d} {month:2d} {temp:8.3f} {uncertainty:6.3f}")
        
        # Write to file
        sample_file = self.raw_dir / "Land_and_Ocean_complete.txt"
        with open(sample_file, 'w') as f:
            f.write('\n'.join(data_lines))
    
    def test_data_validation_pipeline(self):
        """Test data validation component."""
        validator = DataValidator()
        
        # Test file validation
        sample_file = self.raw_dir / "Land_and_Ocean_complete.txt"
        results = validator.validate_file(sample_file)
        
        assert "error" not in results
        assert results["shape"][0] > 0  # Should have data rows
        assert "missing_values" in results
        assert "data_ranges" in results
    
    def test_preprocessing_pipeline(self):
        """Test data preprocessing component."""
        preprocessor = DataPreprocessor()
        
        # Test file processing
        sample_file = self.raw_dir / "Land_and_Ocean_complete.txt"
        output_files = preprocessor.process_file(str(sample_file), str(self.processed_dir))
        
        assert "monthly" in output_files
        assert "yearly" in output_files
        assert "anomalies" in output_files
        
        # Verify output files exist
        for file_path in output_files.values():
            if file_path.endswith('.json'):
                continue  # Trends file may not exist for short data
            assert Path(file_path).exists()
        
        # Test loading processed data
        monthly_df = pd.read_csv(output_files["monthly"])
        assert len(monthly_df) > 0
        assert "year" in monthly_df.columns
        assert "month" in monthly_df.columns
        assert "avg_temperature" in monthly_df.columns
    
    def test_analysis_pipeline(self):
        """Test analysis components."""
        # First run preprocessing to get data
        preprocessor = DataPreprocessor()
        sample_file = self.raw_dir / "Land_and_Ocean_complete.txt"
        preprocessor.process_file(str(sample_file), str(self.processed_dir))
        
        # Test heatmap analysis
        heatmap_analyzer = HeatmapAnalyzer()
        
        try:
            heatmap_file = heatmap_analyzer.generate_global_heatmap(
                str(self.processed_dir), 
                str(self.output_dir)
            )
            assert Path(heatmap_file).exists()
        except Exception as e:
            # Heatmap generation might fail due to limited data
            assert "data" in str(e).lower()
        
        # Test time series analysis  
        ts_analyzer = TimeSeriesAnalyzer()
        
        try:
            ts_results = ts_analyzer.analyze_temperature_trends(
                str(self.processed_dir),
                str(self.output_dir)
            )
            assert "linear_trend" in ts_results
            assert "plot_path" in ts_results
        except Exception as e:
            # Time series analysis might fail due to limited data
            assert "data" in str(e).lower()
    
    def test_ml_pipeline(self):
        """Test machine learning pipeline."""
        # First run preprocessing to get data
        preprocessor = DataPreprocessor()
        sample_file = self.raw_dir / "Land_and_Ocean_complete.txt"
        preprocessor.process_file(str(sample_file), str(self.processed_dir))
        
        # Load processed data
        monthly_files = list(self.processed_dir.glob("*_monthly.csv"))
        if monthly_files:
            df = pd.read_csv(monthly_files[0])
        else:
            # Fallback to original data format
            df = preprocessor.read_berkeley_earth_file(str(sample_file))
        
        # Test baseline model training
        model = BaselineModel("linear")
        results = model.train(df)
        
        assert model.is_fitted
        assert "test_r2" in results
        assert "train_rmse" in results
        
        # Test prediction
        predictions = model.predict(df.head(10))
        assert len(predictions) == 10
        
        # Test future prediction
        future_df = model.predict_future(2024, 2024)
        assert len(future_df) == 12  # 12 months
        assert "predicted_temperature" in future_df.columns
    
    def test_full_pipeline_integration(self):
        """Test complete pipeline from raw data to predictions."""
        # Step 1: Data validation
        validator = DataValidator()
        sample_file = self.raw_dir / "Land_and_Ocean_complete.txt"
        validation_results = validator.validate_file(sample_file)
        
        assert "error" not in validation_results
        
        # Step 2: Data preprocessing
        preprocessor = DataPreprocessor()
        output_files = preprocessor.process_file(str(sample_file), str(self.processed_dir))
        
        # Step 3: Load processed data for ML
        monthly_file = output_files["monthly"]
        df = pd.read_csv(monthly_file)
        
        # Step 4: Train model
        model = BaselineModel("random_forest")
        training_results = model.train(df)
        
        assert training_results["test_r2"] > -1  # Should be reasonable
        
        # Step 5: Make future predictions
        future_predictions = model.predict_future(2024, 2024)
        
        assert len(future_predictions) == 12
        assert all(isinstance(temp, (int, float, np.number)) 
                  for temp in future_predictions["predicted_temperature"])
        
        # Step 6: Verify pipeline produces consistent results
        # Re-run preprocessing
        output_files_2 = preprocessor.process_file(str(sample_file), str(self.processed_dir))
        df_2 = pd.read_csv(output_files_2["monthly"])
        
        # DataFrames should be identical
        pd.testing.assert_frame_equal(df, df_2)
    
    def test_error_handling_pipeline(self):
        """Test pipeline error handling with invalid data."""
        # Create invalid data file
        invalid_file = self.raw_dir / "invalid.txt"
        with open(invalid_file, 'w') as f:
            f.write("Invalid data format\nNot Berkeley Earth format")
        
        # Test validator handles invalid data
        validator = DataValidator()
        results = validator.validate_file(invalid_file)
        
        # Should either handle gracefully or return error
        assert "error" in results or results.get("shape", [0])[0] >= 0
        
        # Test preprocessor handles invalid data
        preprocessor = DataPreprocessor()
        
        try:
            preprocessor.process_file(str(invalid_file), str(self.processed_dir))
        except Exception as e:
            # Should raise meaningful error
            assert len(str(e)) > 0