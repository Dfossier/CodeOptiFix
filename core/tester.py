# core/tester.py
import unittest
from pathlib import Path
import tempfile
import os
from typing import Dict

class CodeTester:
    def __init__(self, test_dir: str):
        self.test_dir = Path(test_dir)

    def run_tests(self, original_files: Dict[str, str], proposed_files: Dict[str, str]) -> bool:
        with tempfile.TemporaryDirectory() as tmp_dir:
            for fname, code in original_files.items():
                (Path(tmp_dir) / fname).write_text(code)
            suite = unittest.TestLoader().discover(tmp_dir)
            result = unittest.TextTestRunner().run(suite)
            if not result.wasSuccessful():
                return False
            for fname, code in proposed_files.items():
                (Path(tmp_dir) / fname).write_text(code)
            result = unittest.TextTestRunner().run(suite)
            return result.wasSuccessful()