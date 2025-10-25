"""
HDFS utilities for reading data directly from Hadoop.
Eliminates the need to download files locally.
"""

import pandas as pd
import logging
from typing import Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)


class HDFSReader:
    """
    Read Parquet files directly from HDFS without downloading to local filesystem.
    Follows Big Data best practices: single source of truth in HDFS.
    """
    
    def __init__(self, namenode_host: str = "climaxtreme-namenode", namenode_port: int = 9000):
        """
        Initialize HDFS reader.
        
        Args:
            namenode_host: HDFS namenode hostname
            namenode_port: HDFS namenode port
        """
        self.namenode_host = namenode_host
        self.namenode_port = namenode_port
        self.hdfs_url = f"hdfs://{namenode_host}:{namenode_port}"
        
    def read_parquet(self, hdfs_path: str) -> pd.DataFrame:
        """
        Read Parquet file directly from HDFS.
        
        Args:
            hdfs_path: Path in HDFS (e.g., /data/climaxtreme/processed/monthly.parquet)
            
        Returns:
            pandas DataFrame
        """
        try:
            # Construct full HDFS URL
            if not hdfs_path.startswith('hdfs://'):
                full_path = f"{self.hdfs_url}{hdfs_path}"
            else:
                full_path = hdfs_path
            
            logger.info(f"Reading Parquet from HDFS: {full_path}")
            
            # PyArrow can read directly from HDFS
            import pyarrow.parquet as pq
            import pyarrow.fs as pafs
            
            # Create HDFS filesystem
            hdfs = pafs.HadoopFileSystem(
                host=self.namenode_host,
                port=self.namenode_port
            )
            
            # Read parquet
            table = pq.read_table(hdfs_path.lstrip('/'), filesystem=hdfs)
            df = table.to_pandas()
            
            logger.info(f"Successfully read {len(df):,} rows from HDFS")
            return df
            
        except Exception as e:
            logger.error(f"Error reading from HDFS: {e}")
            logger.info("Falling back to local filesystem if available")
            raise
    
    def list_files(self, hdfs_directory: str) -> list:
        """
        List files in HDFS directory.
        
        Args:
            hdfs_directory: Directory path in HDFS
            
        Returns:
            List of file paths
        """
        try:
            import pyarrow.fs as pafs
            
            hdfs = pafs.HadoopFileSystem(
                host=self.namenode_host,
                port=self.namenode_port
            )
            
            file_info = hdfs.get_file_info(pafs.FileSelector(hdfs_directory.lstrip('/')))
            files = [info.path for info in file_info if info.type == pafs.FileType.File]
            
            logger.info(f"Found {len(files)} files in {hdfs_directory}")
            return files
            
        except Exception as e:
            logger.error(f"Error listing HDFS directory: {e}")
            return []
    
    def file_exists(self, hdfs_path: str) -> bool:
        """
        Check if file exists in HDFS.
        
        Args:
            hdfs_path: Path in HDFS
            
        Returns:
            True if file exists
        """
        try:
            import pyarrow.fs as pafs
            
            hdfs = pafs.HadoopFileSystem(
                host=self.namenode_host,
                port=self.namenode_port
            )
            
            info = hdfs.get_file_info(hdfs_path.lstrip('/'))
            return info.type != pafs.FileType.NotFound
            
        except Exception as e:
            logger.error(f"Error checking HDFS file: {e}")
            return False


def read_from_hdfs_or_local(
    hdfs_path: str,
    local_path: Optional[str] = None,
    namenode_host: str = "climaxtreme-namenode",
    namenode_port: int = 9000
) -> pd.DataFrame:
    """
    Try to read from HDFS first, fall back to local if HDFS is not available.
    
    Args:
        hdfs_path: Path in HDFS
        local_path: Optional local fallback path
        namenode_host: HDFS namenode hostname
        namenode_port: HDFS namenode port
        
    Returns:
        pandas DataFrame
    """
    # Try HDFS first (Big Data approach)
    try:
        reader = HDFSReader(namenode_host, namenode_port)
        return reader.read_parquet(hdfs_path)
    except Exception as e:
        logger.warning(f"Could not read from HDFS: {e}")
        
        # Fall back to local if provided
        if local_path and Path(local_path).exists():
            logger.info(f"Falling back to local file: {local_path}")
            return pd.read_parquet(local_path)
        else:
            raise FileNotFoundError(
                f"Could not read from HDFS ({hdfs_path}) "
                f"and no valid local fallback provided ({local_path})"
            )
