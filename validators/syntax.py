"""
Syntax Validator

Validates that code has correct syntax.
"""
import ast
from pathlib import Path
from typing import Tuple, Optional

from code_updater import ValidationRule

class SyntaxValidator(ValidationRule):
    """Validates that code has correct syntax."""
    
    def get_name(self) -> str:
        """Return the name of this validation rule."""
        return "SyntaxValidator"
    
    def get_description(self) -> str:
        """Return a description of this validation rule."""
        return "Validates that code has correct syntax using Python's ast module."
    
    def validate(self, file_path: Path, code: str) -> Tuple[bool, Optional[str]]:
        """
        Validate the code using Python's built-in syntax checking.
        
        Args:
            file_path: Path to the file being validated
            code: The code to validate
            
        Returns:
            Tuple of (success, error_message)
        """
        try:
            # Check syntax by attempting to parse the code with ast
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, f"Syntax error in {file_path.name} at line {e.lineno}, column {e.offset}: {e.msg}"
        except Exception as e:
            return False, f"Error checking syntax of {file_path.name}: {str(e)}"