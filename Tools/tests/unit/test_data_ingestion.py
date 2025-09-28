"""
Unit tests for data ingestion module.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from climaxtreme.data.ingestion import DataIngestion


class TestDataIngestion:
    """Test cases for DataIngestion class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.ingestion = DataIngestion(self.temp_dir)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_init(self):
        """Test DataIngestion initialization."""
        assert self.ingestion.output_dir == Path(self.temp_dir)
        assert self.ingestion.output_dir.exists()
        assert hasattr(self.ingestion, 'session')
    
    @patch('requests.Session.get')
    def test_download_file_success(self, mock_get):
        """Test successful file download."""
        # Mock response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.iter_content.return_value = [b'test data chunk 1', b'test data chunk 2']
        mock_get.return_value = mock_response
        
        # Test download
        result = self.ingestion.download_file("http://example.com/test.txt", "test.txt")
        
        assert result is True
        test_file = Path(self.temp_dir) / "test.txt"
        assert test_file.exists()
        
        # Verify content
        with open(test_file, 'rb') as f:
            content = f.read()
        assert content == b'test data chunk 1test data chunk 2'
    
    @patch('requests.Session.get')
    def test_download_file_existing(self, mock_get):
        """Test download when file already exists."""
        # Create existing file
        test_file = Path(self.temp_dir) / "existing.txt"
        test_file.write_text("existing content")
        
        # Test download
        result = self.ingestion.download_file("http://example.com/existing.txt", "existing.txt")
        
        assert result is True
        mock_get.assert_not_called()  # Should not attempt download
        
        # Verify original content preserved
        assert test_file.read_text() == "existing content"
    
    @patch('requests.Session.get')
    def test_download_file_failure(self, mock_get):
        """Test download failure handling."""
        # Mock failed response
        mock_get.side_effect = Exception("Network error")
        
        # Test download
        result = self.ingestion.download_file("http://example.com/fail.txt", "fail.txt")
        
        assert result is False
        test_file = Path(self.temp_dir) / "fail.txt"
        assert not test_file.exists()
    
    @patch.object(DataIngestion, 'download_file')
    def test_download_berkeley_earth_data(self, mock_download):
        """Test Berkeley Earth data download."""
        mock_download.return_value = True
        
        # Test download
        result = self.ingestion.download_berkeley_earth_data()
        
        # Verify all expected files were requested
        expected_files = list(DataIngestion.DATA_FILES.values())
        assert mock_download.call_count == len(expected_files)
        assert len(result) == len(expected_files)
    
    @patch.object(DataIngestion, 'download_file')
    def test_download_berkeley_earth_data_with_years(self, mock_download):
        """Test Berkeley Earth data download with year range."""
        mock_download.return_value = True
        
        # Test download with year range
        result = self.ingestion.download_berkeley_earth_data(2020, 2022)
        
        # Should download base files + monthly files for 2020-2022
        expected_calls = len(DataIngestion.DATA_FILES) + 3  # 3 years
        assert mock_download.call_count == expected_calls
    
    def test_list_downloaded_files_empty(self):
        """Test listing files when directory is empty."""
        files = self.ingestion.list_downloaded_files()
        assert files == []
    
    def test_list_downloaded_files(self):
        """Test listing downloaded files."""
        # Create test files
        test_files = ["file1.txt", "file2.nc", ".hidden"]
        for filename in test_files:
            (Path(self.temp_dir) / filename).touch()
        
        files = self.ingestion.list_downloaded_files()
        
        # Should exclude hidden files
        assert len(files) == 2
        file_names = [f.name for f in files]
        assert "file1.txt" in file_names
        assert "file2.nc" in file_names
        assert ".hidden" not in file_names
    
    def test_get_file_info_existing(self):
        """Test getting info for existing file."""
        # Create test file
        test_file = Path(self.temp_dir) / "test.txt"
        test_content = "test content"
        test_file.write_text(test_content)
        
        info = self.ingestion.get_file_info("test.txt")
        
        assert info["exists"] is True
        assert info["size_bytes"] == len(test_content)
        assert info["size_mb"] > 0
        assert "modified" in info
        assert info["path"] == str(test_file)
    
    def test_get_file_info_nonexistent(self):
        """Test getting info for non-existent file."""
        info = self.ingestion.get_file_info("nonexistent.txt")
        
        assert info["exists"] is False
        assert len(info) == 1