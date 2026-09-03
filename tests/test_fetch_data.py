import pytest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from unittest.mock import patch, MagicMock
from fetch_data import main


class TestFetchData:
    """Tests for the data fetching script."""

    @patch('fetch_data.urllib.request.urlretrieve')
    def test_main_downloads_file(self, mock_urlretrieve, tmp_path):
        """Verify main() calls urlretrieve with the correct URL and destination."""
        with patch('fetch_data.os.path.dirname', return_value=str(tmp_path)):
            with patch('fetch_data.os.path.abspath', return_value=str(tmp_path)):
                main()

        mock_urlretrieve.assert_called_once()
        call_args = mock_urlretrieve.call_args
        assert 'Telco-Customer-Churn.csv' in call_args[0][0]

    @patch('fetch_data.urllib.request.urlretrieve', side_effect=Exception("Network error"))
    def test_main_handles_download_failure(self, mock_urlretrieve, capsys):
        """Verify main() prints a failure message on network errors instead of crashing."""
        main()
        captured = capsys.readouterr()
        assert "Failed to download" in captured.out

    @patch('fetch_data.urllib.request.urlretrieve')
    def test_main_creates_data_directory(self, mock_urlretrieve, tmp_path):
        """Verify main() creates the data/ directory if it does not exist."""
        with patch('fetch_data.os.makedirs') as mock_makedirs:
            main()
            mock_makedirs.assert_called_once()
            call_args = mock_makedirs.call_args
            assert call_args[1].get('exist_ok') is True
