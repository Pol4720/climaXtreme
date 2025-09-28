"""
Logging configuration for climaXtreme.
"""

import logging
import logging.config
from pathlib import Path
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_dir: str = "logs"
) -> None:
    """
    Set up logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file name
        log_dir: Directory for log files
    """
    
    # Create logs directory if it doesn't exist
    if log_file:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        log_filepath = log_path / log_file
    else:
        log_filepath = None
    
    # Logging configuration
    config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'detailed': {
                'format': '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': level,
                'formatter': 'standard',
                'stream': 'ext://sys.stdout'
            }
        },
        'loggers': {
            'climaxtreme': {
                'level': level,
                'handlers': ['console'],
                'propagate': False
            },
            'pyspark': {
                'level': 'WARN',
                'handlers': ['console'],
                'propagate': False
            },
            'py4j': {
                'level': 'WARN',
                'handlers': ['console'], 
                'propagate': False
            }
        },
        'root': {
            'level': level,
            'handlers': ['console']
        }
    }
    
    # Add file handler if log file specified
    if log_filepath:
        config['handlers']['file'] = {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': level,
            'formatter': 'detailed',
            'filename': str(log_filepath),
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5
        }
        
        # Add file handler to loggers
        config['loggers']['climaxtreme']['handlers'].append('file')
        config['root']['handlers'].append('file')
    
    # Apply configuration
    logging.config.dictConfig(config)
    
    # Log setup completion
    logger = logging.getLogger('climaxtreme.utils.logging_config')
    logger.info(f"Logging configured with level: {level}")
    if log_filepath:
        logger.info(f"Log file: {log_filepath}")