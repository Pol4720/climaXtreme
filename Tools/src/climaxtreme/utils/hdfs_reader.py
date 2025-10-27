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
        Read Parquet file directly from HDFS using WebHDFS API.
        
        Args:
            hdfs_path: Path in HDFS (e.g., /data/climaxtreme/processed/monthly.parquet)
            
        Returns:
            pandas DataFrame
        """
        try:
            logger.info(f"Reading Parquet from HDFS: {hdfs_path}")
            
            # Use WebHDFS to read the parquet file
            import pyarrow.parquet as pq
            import requests
            import io
            
            # For Spark-written parquet (directories with part files),
            # we need to find all part files and read them
            webhdfs_url = f"http://{self.namenode_host}:9870/webhdfs/v1{hdfs_path}"
            params = {'op': 'LISTSTATUS'}
            
            response = requests.get(webhdfs_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            file_statuses = data.get('FileStatuses', {}).get('FileStatus', [])
            
            # Find all parquet part files (excluding _SUCCESS)
            part_files = [f['pathSuffix'] for f in file_statuses 
                         if f['pathSuffix'].endswith('.parquet')]
            
            if not part_files:
                raise ValueError(f"No parquet part files found in {hdfs_path}")
            
            # Read all part files and concatenate
            dfs = []
            for part_file in part_files:
                part_path = f"{hdfs_path}/{part_file}"
                part_url = f"http://{self.namenode_host}:9870/webhdfs/v1{part_path}?op=OPEN"
                
                logger.debug(f"Reading part file: {part_file}")
                part_response = requests.get(part_url, timeout=30)
                part_response.raise_for_status()
                
                # Read parquet from bytes
                parquet_bytes = io.BytesIO(part_response.content)
                df_part = pq.read_table(parquet_bytes).to_pandas()
                dfs.append(df_part)
            
            # Concatenate all parts
            df = pd.concat(dfs, ignore_index=True)
            
            logger.info(f"Successfully read {len(df):,} rows from HDFS")
            return df
            
        except Exception as e:
            logger.error(f"Error reading from HDFS: {e}")
            raise
    
    def list_files(self, hdfs_directory: str) -> list:
        """
        List files and directories in HDFS directory.
        
        Args:
            hdfs_directory: Directory path in HDFS
            
        Returns:
            List of file/directory paths (includes Parquet directories)
        """
        try:
            # Use WebHDFS API for listing (more compatible)
            import requests
            
            # WebHDFS runs on port 9870 (HTTP) by default on namenode
            webhdfs_url = f"http://{self.namenode_host}:9870/webhdfs/v1{hdfs_directory}"
            params = {'op': 'LISTSTATUS'}
            
            response = requests.get(webhdfs_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            file_statuses = data.get('FileStatuses', {}).get('FileStatus', [])
            
            # Get paths of all items (files and directories)
            files = [f"{hdfs_directory}/{item['pathSuffix']}" 
                    for item in file_statuses 
                    if item['pathSuffix']]  # Exclude empty names
            
            logger.info(f"Found {len(files)} items in {hdfs_directory}")
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
            import requests
            
            # Use WebHDFS API
            webhdfs_url = f"http://{self.namenode_host}:9870/webhdfs/v1{hdfs_path}"
            params = {'op': 'GETFILESTATUS'}
            
            response = requests.get(webhdfs_url, params=params, timeout=10)
            return response.status_code == 200
            
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
