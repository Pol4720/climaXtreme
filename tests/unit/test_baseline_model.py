"""
Unit tests for baseline ML model.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path

from climaxtreme.ml.baseline import BaselineModel


class TestBaselineModel:
    """Test cases for BaselineModel class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample data
        np.random.seed(42)
        years = np.arange(2000, 2020)
        months = np.arange(1, 13)
        
        data = []
        for year in years:
            for month in months:
                # Simple seasonal pattern + trend + noise
                temp = 15 + 10 * np.sin(2 * np.pi * month / 12) + 0.02 * (year - 2000) + np.random.normal(0, 2)
                data.append({
                    'year': year,
                    'month': month,
                    'temperature': temp
                })
        
        self.sample_data = pd.DataFrame(data)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_init_valid_model(self):
        """Test initialization with valid model type."""
        model = BaselineModel("linear")
        assert model.model_type == "linear"
        assert model.model is not None
        assert not model.is_fitted
    
    def test_init_invalid_model(self):
        """Test initialization with invalid model type."""
        with pytest.raises(ValueError, match="Unsupported model type"):
            BaselineModel("invalid_model")
    
    def test_prepare_features(self):
        """Test feature preparation."""
        model = BaselineModel("linear")
        X, y = model.prepare_features(self.sample_data)
        
        assert X.shape[0] == len(self.sample_data)
        assert X.shape[1] > 0  # Should have multiple features
        assert len(y) == len(self.sample_data)
        assert len(model.feature_names) == X.shape[1]
    
    def test_prepare_features_missing_columns(self):
        """Test feature preparation with missing required columns."""
        model = BaselineModel("linear")
        
        # Data without required columns
        invalid_data = pd.DataFrame({'invalid_col': [1, 2, 3]})
        
        with pytest.raises(ValueError, match="Required column"):
            model.prepare_features(invalid_data)
    
    def test_train_linear_model(self):
        """Test training linear model."""
        model = BaselineModel("linear")
        results = model.train(self.sample_data)
        
        assert model.is_fitted
        assert "train_rmse" in results
        assert "test_rmse" in results
        assert "train_r2" in results
        assert "test_r2" in results
        assert results["test_r2"] > 0  # Should have some predictive power
    
    def test_train_random_forest_model(self):
        """Test training random forest model."""
        model = BaselineModel("random_forest")
        results = model.train(self.sample_data)
        
        assert model.is_fitted
        assert "feature_importance" in results
        assert results["test_r2"] > 0
    
    def test_predict_without_training(self):
        """Test prediction without training."""
        model = BaselineModel("linear")
        
        with pytest.raises(ValueError, match="Model must be trained"):
            model.predict(self.sample_data)
    
    def test_predict_after_training(self):
        """Test prediction after training."""
        model = BaselineModel("linear")
        model.train(self.sample_data)
        
        predictions = model.predict(self.sample_data.head(10))
        
        assert len(predictions) == 10
        assert all(isinstance(p, (int, float, np.number)) for p in predictions)
    
    def test_predict_future(self):
        """Test future prediction."""
        model = BaselineModel("linear")
        model.train(self.sample_data)
        
        future_df = model.predict_future(2021, 2022)
        
        assert len(future_df) == 24  # 2 years * 12 months
        assert 'year' in future_df.columns
        assert 'month' in future_df.columns
        assert 'predicted_temperature' in future_df.columns
        assert future_df['year'].min() == 2021
        assert future_df['year'].max() == 2022
    
    def test_save_and_load_model(self):
        """Test model saving and loading."""
        model = BaselineModel("linear")
        model.train(self.sample_data)
        
        # Save model
        model_path = Path(self.temp_dir) / "test_model.joblib"
        model.save_model(str(model_path))
        
        assert model_path.exists()
        
        # Load model
        new_model = BaselineModel("linear")
        new_model.load_model(str(model_path))
        
        assert new_model.is_fitted
        assert new_model.model_type == "linear"
        assert new_model.feature_names == model.feature_names
        
        # Test predictions are identical
        test_data = self.sample_data.head(5)
        pred1 = model.predict(test_data)
        pred2 = new_model.predict(test_data)
        
        np.testing.assert_array_almost_equal(pred1, pred2)
    
    def test_save_model_without_training(self):
        """Test saving model without training."""
        model = BaselineModel("linear")
        
        with pytest.raises(ValueError, match="Model must be trained"):
            model.save_model("test.joblib")
    
    def test_hyperparameter_tuning_ridge(self):
        """Test hyperparameter tuning for Ridge model."""
        model = BaselineModel("ridge")
        results = model.hyperparameter_tuning(self.sample_data)
        
        assert "best_params" in results
        assert "best_score" in results
        assert "alpha" in results["best_params"]
    
    def test_hyperparameter_tuning_unsupported_model(self):
        """Test hyperparameter tuning for unsupported model."""
        model = BaselineModel("linear")  # No default param grid
        results = model.hyperparameter_tuning(self.sample_data)
        
        assert results == {}  # Should return empty dict
    
    def test_get_temperature_column(self):
        """Test temperature column detection."""
        model = BaselineModel("linear")
        
        # Test with standard column name
        assert model._get_temperature_column(self.sample_data) == "temperature"
        
        # Test with alternative column name
        alt_data = self.sample_data.rename(columns={'temperature': 'avg_temperature'})
        assert model._get_temperature_column(alt_data) == "avg_temperature"
        
        # Test with no temperature column
        no_temp_data = self.sample_data.drop(columns=['temperature'])
        assert model._get_temperature_column(no_temp_data) is None