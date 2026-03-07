import unittest
from unittest.mock import patch, MagicMock
import subprocess
import sys
import setup_and_run

class TestSetupAndRun(unittest.TestCase):
    @patch('setup_and_run.subprocess.check_call')
    def test_install_success(self, mock_check_call):
        package = "test-package"
        setup_and_run.install(package)
        mock_check_call.assert_called_once_with([sys.executable, "-m", "pip", "install", package])

    @patch('setup_and_run.subprocess.check_call')
    def test_install_failure(self, mock_check_call):
        mock_check_call.side_effect = subprocess.CalledProcessError(1, 'pip install')
        with self.assertRaises(subprocess.CalledProcessError):
            setup_and_run.install("test-package")

    @patch('setup_and_run.subprocess.check_call')
    @patch('setup_and_run.install')
    @patch('builtins.__import__')
    def test_main(self, mock_import, mock_install, mock_check_call):
        # Mocking __import__ to fail for the first package and succeed for others
        def side_effect(name, *args, **kwargs):
            if name == setup_and_run.REQUIRED[0].split("-")[0]:
                raise ImportError
            return MagicMock()

        mock_import.side_effect = side_effect

        setup_and_run.main()

        # Verify install was called for the first package
        mock_install.assert_called_once_with(setup_and_run.REQUIRED[0])

        # Verify subprocess.check_call was called to run analyze_face.py
        mock_check_call.assert_called_once_with([sys.executable, "analyze_face.py"])
