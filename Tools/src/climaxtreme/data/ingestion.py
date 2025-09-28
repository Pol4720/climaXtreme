"""
Data ingestion module for Berkeley Earth climate data.
"""

import os
import logging
from pathlib import Path
from typing import List, Optional
from urllib.parse import urljoin
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


logger = logging.getLogger(__name__)


class DataIngestion:
    """
    Handles downloading and ingesting Berkeley Earth climate data.
    
    Berkeley Earth provides gridded climate data including temperature
    and precipitation records from weather stations worldwide.
    """
    
    BASE_URL = "https://berkeley-earth-temperature.s3.amazonaws.com/Global/"
    
    # Common Berkeley Earth data files
    DATA_FILES = {
        "land_temperature": "Land_and_Ocean_complete.txt",
        "land_only_temperature": "Land_complete.txt",
        "ocean_temperature": "Ocean_complete.txt",
        "gridded_temperature": "Land_and_Ocean_LatLong1.nc",
        "station_list": "station_list.txt"
    }
    
    def __init__(self, output_dir: str) -> None:
        """
        Initialize the data ingestion client.
        
        Args:
            output_dir: Directory to store downloaded data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure HTTP session with retries
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
    
    def download_file(self, url: str, filename: str) -> bool:
        """
        Download a file from the given URL.
        
        Args:
            url: URL to download from
            filename: Local filename to save to
            
        Returns:
            True if download successful, False otherwise
        """
        filepath = self.output_dir / filename
        
        if filepath.exists():
            logger.info(f"File {filename} already exists, skipping download")
            return True
        
        try:
            logger.info(f"Downloading {filename} from {url}")
            response = self.session.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            logger.info(f"Successfully downloaded {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading {filename}: {e}")
            if filepath.exists():
                filepath.unlink()  # Remove partial file
            return False
    
    def download_berkeley_earth_data(
        self, 
        start_year: Optional[int] = None, 
        end_year: Optional[int] = None
    ) -> List[str]:
        """
        Download Berkeley Earth climate data files.
        
        Args:
            start_year: Start year for data (if applicable)
            end_year: End year for data (if applicable)
            
        Returns:
            List of successfully downloaded filenames
        """
        downloaded_files = []
        
        for data_type, filename in self.DATA_FILES.items():
            url = urljoin(self.BASE_URL, filename)
            
            if self.download_file(url, filename):
                downloaded_files.append(filename)
        
        # Download monthly data files if year range specified
        if start_year and end_year:
            downloaded_files.extend(
                self._download_monthly_data(start_year, end_year)
            )
        
        logger.info(f"Downloaded {len(downloaded_files)} files successfully")
        return downloaded_files
    
    def _download_monthly_data(self, start_year: int, end_year: int) -> List[str]:
        """
        Download monthly gridded data files for specified year range.
        
        Args:
            start_year: Start year
            end_year: End year
            
        Returns:
            List of successfully downloaded monthly files
        """
        downloaded_files = []
        
        for year in range(start_year, end_year + 1):
            # Berkeley Earth monthly files follow pattern: YYYY.nc
            filename = f"{year}.nc"
            url = urljoin(self.BASE_URL + "Gridded/", filename)
            
            if self.download_file(url, filename):
                downloaded_files.append(filename)
        
        return downloaded_files
    
    def list_downloaded_files(self) -> List[Path]:
        """
        List all downloaded data files.
        
        Returns:
            List of Path objects for downloaded files
        """
        if not self.output_dir.exists():
            return []
        
        return [
            f for f in self.output_dir.iterdir() 
            if f.is_file() and not f.name.startswith('.')
        ]
    
    def get_file_info(self, filename: str) -> dict:
        """
        Get information about a downloaded file.
        
        Args:
            filename: Name of the file
            
        Returns:
            Dictionary with file information
        """
        filepath = self.output_dir / filename
        
        if not filepath.exists():
            return {"exists": False}
        
        stat = filepath.stat()
        return {
            "exists": True,
            "size_bytes": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "modified": stat.st_mtime,
            "path": str(filepath)
        }