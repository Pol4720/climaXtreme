"""
Configuration management for climaXtreme.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from dataclasses import dataclass, asdict


def get_repo_root() -> Path:
    """Return the absolute path to the repository root.

    This is computed relative to this file's location (Tools/src/climaxtreme/utils/).
    The repo root is four levels up from here.
    """
    return Path(__file__).resolve().parents[4]


def default_dataset_dir() -> Path:
    """Default path to the dataset directory at repo root (DATA/)."""
    return get_repo_root() / "DATA"


@dataclass
class Config:
    """Configuration settings for climaXtreme."""
    
    # Data paths
    # Default dataset directory points to repo-root/DATA by default
    data_dir: str = str(default_dataset_dir())
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    output_data_dir: str = "data/output"
    models_dir: str = "models"
    
    # Berkeley Earth settings
    berkeley_earth_base_url: str = "https://berkeley-earth-temperature.s3.amazonaws.com/Global/"
    default_start_year: int = 2020
    default_end_year: int = 2023
    
    # Spark settings
    spark_app_name: str = "climaXtreme"
    spark_master: str = "local[*]"
    spark_sql_adaptive_enabled: bool = True
    spark_sql_adaptive_coalescePartitions_enabled: bool = True
    
    # Analysis settings
    anomaly_threshold_percentile: float = 95.0
    temperature_bounds_min: float = -100.0
    temperature_bounds_max: float = 60.0
    
    # ML settings
    test_size: float = 0.2
    cv_folds: int = 5
    random_state: int = 42
    
    # Dashboard settings
    dashboard_host: str = "localhost"
    dashboard_port: int = 8501
    
    # Visualization settings
    figure_dpi: int = 300
    figure_format: str = "png"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def save(self, config_path: str) -> None:
        """Save configuration to YAML file."""
        config_dict = self.to_dict()
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    @classmethod
    def load(cls, config_path: str) -> 'Config':
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls.from_dict(config_dict)


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from file or create default.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Config object
    """
    if config_path is None:
        # Look for config in common locations
        possible_paths = [
            "climaxtreme_config.yml",
            "climaxtreme_config.yaml", 
            "config.yml",
            "config.yaml",
            os.path.expanduser("~/.climaxtreme/config.yml"),
            "/etc/climaxtreme/config.yml"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                config_path = path
                break
    
    if config_path and os.path.exists(config_path):
        return Config.load(config_path)
    else:
        # Return default configuration
        return Config()