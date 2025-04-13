"""
Tests Validator

Validates that code passes tests.
"""
import subprocess
from pathlib import Path
from typing import Tuple, Optional, List

from code_updater import ValidationRule

class TestsValidator(ValidationRule):
    """Validates that code passes tests."""
    
    def get_name(self) -> str:
        """Return the name of this validation rule."""
        return "TestsValidator"
    
    def get_description(self) -> str:
        """Return a description of this validation rule."""
        return "Validates that code passes the project's test suite."
    
    def validate(self, file_path: Path, code: str) -> Tuple[bool, Optional[str]]:
        """
        Validate the code by running tests.
        
        Args:
            file_path: Path to the file being validated
            code: The code to validate
            
        Returns:
            Tuple of (success, error_message)
        """
        # Determine the test command based on the file path
        module_path = self._get_module_path(file_path)
        
        # Only run tests if we have a module path
        if not module_path:
            return True, None
        
        # Try to find and run tests for this module
        test_file = self._find_test_file(file_path)
        
        if not test_file:
            # If no test file found, just return success
            return True, None
        
        try:
            # Run the test
            result = subprocess.run(
                ["python", "-m", "unittest", test_file],
                capture_output=True,
                text=True,
                check=False
            )
            
            # Check if tests passed
            if result.returncode == 0:
                return True, None
            else:
                # Extract the first few lines of the error
                error_lines = result.stderr.splitlines() or result.stdout.splitlines()
                error_summary = "\n".join(error_lines[:5])
                return False, f"Tests failed for {module_path}:\n{error_summary}"
                
        except Exception as e:
            return False, f"Error running tests for {module_path}: {str(e)}"
    
    def _get_module_path(self, file_path: Path) -> Optional[str]:
        """Get the module path for a file."""
        rel_path = file_path.relative_to(Path.cwd())
        module_path = str(rel_path).replace("/", ".").replace("\\", ".").replace(".py", "")
        return module_path
    
    def _find_test_file(self, file_path: Path) -> Optional[str]:
        """Find the test file for a given file."""
        # Common test file patterns
        patterns = [
            Path("tests") / f"test_{file_path.name}",
            Path("tests") / file_path.parent.name / f"test_{file_path.name}",
            file_path.parent / f"test_{file_path.name}",
            Path("tests") / f"{file_path.stem}_test.py",
            Path("test") / f"{file_path.stem}_test.py"
        ]
        
        # Check each pattern
        for pattern in patterns:
            if pattern.exists():
                return str(pattern)
        
        return None