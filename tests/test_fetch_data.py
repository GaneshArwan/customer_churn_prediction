import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from fetch_data import main


def _mock_https_response(url="https://raw.githubusercontent.com/IBM/telco.csv", payload=b"a,b\n1,2\n"):
    resp = MagicMock()
    resp.geturl.return_value = url
    resp.read.return_value = payload
    cm = MagicMock()
    cm.__enter__.return_value = resp
    cm.__exit__.return_value = False
    return cm


class TestFetchData:
    """Tests for the data fetching script."""

    @patch('fetch_data.urllib.request.urlopen')
    def test_main_downloads_file(self, mock_urlopen, tmp_path, monkeypatch):
        """Verify main() fetches HTTPS URL and writes dataset to data/ directory."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        monkeypatch.setattr('fetch_data.os.path.abspath', lambda p: str(src_dir / "fetch_data.py"))
        mock_urlopen.return_value = _mock_https_response()

        main()

        mock_urlopen.assert_called_once()
        dest_file = tmp_path / "data" / "telco_customer_churn.csv"
        assert dest_file.exists()
        assert dest_file.read_bytes() == b"a,b\n1,2\n"

    @patch('fetch_data.urllib.request.urlopen', side_effect=Exception("Network error"))
    def test_main_handles_download_failure(self, mock_urlopen, tmp_path, monkeypatch, capsys):
        """Verify main() prints a failure message on network errors inside isolated tmp_path."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        monkeypatch.setattr('fetch_data.os.path.abspath', lambda p: str(src_dir / "fetch_data.py"))

        main()
        captured = capsys.readouterr()
        assert "Failed to download" in captured.out

    @patch('fetch_data.urllib.request.urlopen')
    def test_main_rejects_non_https_redirect(self, mock_urlopen, tmp_path, monkeypatch, capsys):
        """Verify non-HTTPS redirect target is rejected and file is not written."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        monkeypatch.setattr('fetch_data.os.path.abspath', lambda p: str(src_dir / "fetch_data.py"))
        mock_urlopen.return_value = _mock_https_response(url="http://insecure.example.com/file.csv")

        main()
        captured = capsys.readouterr()
        assert "Refusing non-HTTPS redirect target" in captured.out
        assert not (tmp_path / "data" / "telco_customer_churn.csv").exists()

    @patch('fetch_data.urllib.request.urlopen')
    def test_main_creates_data_directory(self, mock_urlopen, tmp_path, monkeypatch):
        """Verify main() creates the data/ directory if it does not exist."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        monkeypatch.setattr('fetch_data.os.path.abspath', lambda p: str(src_dir / "fetch_data.py"))
        mock_urlopen.return_value = _mock_https_response()

        main()
        assert (tmp_path / "data").is_dir()
