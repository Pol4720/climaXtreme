"""
Unit tests for configuration management.
"""

import pytest
import tempfile
import shutil
from pathlib import Path

from climaxtreme.utils.config import Config, load_config


class TestConfig:
    """Test cases for Config class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_default_config(self):
        """Test default configuration values."""
        config = Config()
        
        assert config.data_dir == "data"
        assert config.raw_data_dir == "data/raw"
        assert config.processed_data_dir == "data/processed"
        assert config.spark_app_name == "climaXtreme"
        assert config.anomaly_threshold_percentile == 95.0
        assert config.test_size == 0.2
        assert config.dashboard_port == 8501
    
    def test_custom_config(self):
        """Test configuration with custom values."""
        config = Config(
            data_dir="custom_data",
            spark_app_name="custom_app",
            dashboard_port=9999
        )
        
        assert config.data_dir == "custom_data"
        assert config.spark_app_name == "custom_app"
        assert config.dashboard_port == 9999
        # Default values should still be present
        assert config.test_size == 0.2
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = Config(data_dir="test_data")
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict["data_dir"] == "test_data"
        assert "spark_app_name" in config_dict
    
    def test_from_dict(self):
        """Test creation from dictionary."""
        config_dict = {
            "data_dir": "dict_data",
            "spark_app_name": "dict_app",
            "dashboard_port": 7777
        }
        
        config = Config.from_dict(config_dict)
        
        assert config.data_dir == "dict_data"
        assert config.spark_app_name == "dict_app"
        assert config.dashboard_port == 7777
    
    def test_save_and_load(self):
        """Test saving and loading configuration."""
        config = Config(
            data_dir="save_test",
            spark_app_name="save_app",
            dashboard_port=8888
        )
        
        config_path = Path(self.temp_dir) / "test_config.yml"
        
        # Save config
        config.save(str(config_path))
        assert config_path.exists()
        
        # Load config
        loaded_config = Config.load(str(config_path))
        
        assert loaded_config.data_dir == "save_test"
        assert loaded_config.spark_app_name == "save_app"
        assert loaded_config.dashboard_port == 8888
        # Default values should be preserved
        assert loaded_config.test_size == 0.2


class TestLoadConfig:
    """Test cases for load_config function."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_load_config_file_exists(self):
        """Test loading config when file exists."""
        config_path = Path(self.temp_dir) / "config.yml"
        
        # Create config file
        test_config = Config(data_dir="file_test")
        test_config.save(str(config_path))
        
        # Load config
        loaded_config = load_config(str(config_path))
        
        assert loaded_config.data_dir == "file_test"
    
    def test_load_config_file_not_exists(self):
        """Test loading config when file doesn't exist."""
        non_existent_path = Path(self.temp_dir) / "nonexistent.yml"
        
        # Should return default config
        config = load_config(str(non_existent_path))
        
        assert isinstance(config, Config)
        assert config.data_dir == "data"  # Default value
    
    def test_load_config_no_path(self):
        """Test loading config without specifying path."""
        # Should return default config
        config = load_config()
        
        assert isinstance(config, Config)
        assert config.data_dir == "data"  # Default value