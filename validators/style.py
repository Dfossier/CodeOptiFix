"""
Style Validator

Validates that code follows style guidelines.
"""
import re
from pathlib import Path
from typing import Tuple, Optional, List, Dict

from code_updater import ValidationRule

class StyleValidator(ValidationRule):
    """Validates that code follows style guidelines."""
    
    def get_name(self) -> str:
        """Return the name of this validation rule."""
        return "StyleValidator"
    
    def get_description(self) -> str:
        """Return a description of this validation rule."""
        return "Validates that code follows style guidelines like line length and import style."
    
    def validate(self, file_path: Path, code: str) -> Tuple[bool, Optional[str]]:
        """
        Validate the code style.
        
        Args:
            file_path: Path to the file being validated
            code: The code to validate
            
        Returns:
            Tuple of (success, error_message)
        """
        # Define style checks
        checks = [
            self._check_line_length,
            self._check_import_style,
            self._check_function_style
        ]
        
        # Run all checks
        issues = []
        for check in checks:
            success, message = check(code)
            if not success:
                issues.append(message)
        
        # Return success if no issues found
        if not issues:
            return True, None
        else:
            return False, f"Style issues in {file_path.name}: {'; '.join(issues)}"
    
    def _check_line_length(self, code: str) -> Tuple[bool, Optional[str]]:
        """Check that lines are not too long."""
        lines = code.splitlines()
        long_lines = []
        
        for i, line in enumerate(lines):
            if len(line) > 100:  # 100 characters is our limit
                long_lines.append(i + 1)
                
                # Limit the number of reported lines
                if len(long_lines) >= 5:
                    break
        
        if long_lines:
            return False, f"Lines too long (>100 chars) at lines: {', '.join(map(str, long_lines))}"
        
        return True, None
    
    def _check_import_style(self, code: str) -> Tuple[bool, Optional[str]]:
        """Check import style."""
        # Check for imports not at the top of the file
        lines = code.splitlines()
        in_imports = False
        found_non_import = False
        late_imports = []
        
        for i, line in enumerate(lines):
            # Skip comments and empty lines
            if line.strip().startswith("#") or not line.strip():
                continue
                
            # Check if this is an import statement
            if line.strip().startswith(("import ", "from ")):
                in_imports = True
                if found_non_import:
                    late_imports.append(i + 1)
            elif in_imports and not line.strip().startswith(("import ", "from ")):
                found_non_import = True
        
        if late_imports:
            return False, f"Imports should be at the top of the file. Late imports at lines: {', '.join(map(str, late_imports))}"
        
        return True, None
    
    def _check_function_style(self, code: str) -> Tuple[bool, Optional[str]]:
        """Check function style."""
        # Check for functions without docstrings
        function_pattern = r"def\s+(\w+)\s*\(.*\).*:"
        functions = re.finditer(function_pattern, code)
        
        functions_without_docstrings = []
        
        for match in functions:
            func_name = match.group(1)
            func_pos = match.end()
            
            # Check if there's a docstring after the function definition
            docstring_pattern = r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\''
            
            # Look for a docstring in the next 5 lines
            next_5_lines = code[func_pos:func_pos + 200]  # A generous limit
            
            if not re.search(docstring_pattern, next_5_lines):
                functions_without_docstrings.append(func_name)
                
                # Limit the number of reported functions
                if len(functions_without_docstrings) >= 5:
                    break
        
        if functions_without_docstrings:
            return False, f"Functions missing docstrings: {', '.join(functions_without_docstrings)}"
        
        return True, None