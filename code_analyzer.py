"""
Code Analyzer module for the Self-Improving AI Assistant Update Generator.

Parses existing codebase to understand structure and dependencies.
Identifies target functions/modules for improvement.
"""
import ast
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
import importlib
import inspect

import utils
from utils import CodeAnalysisError, setup_logging

logger = setup_logging(__name__)

class CodeAnalyzer:
    """Analyzes Python code to provide context for improvements."""
    
    def __init__(self, base_path: Optional[Path] = None):
        """Initialize the analyzer with an optional base path."""
        self.base_path = base_path or Path.cwd()
        self.logger = logger
        
    def should_skip_file(self, file_path: str) -> bool:
        """Determine if a file should be skipped during analysis."""
        # Skip files in virtual environments and other generated code
        skip_patterns = [
            '.venv/', 
            'site-packages/',
            '.git/',
            '__pycache__/',
            '.pytest_cache/'
        ]
        return any(pattern in file_path for pattern in skip_patterns)
    
    def analyze_module(self, module_path: str) -> Dict[str, Any]:
        """
        Analyze a Python module and return its structure and context.
        
        Args:
            module_path: Path to the module to analyze (relative to base_path)
            
        Returns:
            Dict containing module analysis information
        """
        # Skip analysis for virtual environment files
        if self.should_skip_file(module_path):
            self.logger.debug(f"Skipping virtual environment module: {module_path}")
            return {
                "module_path": str(module_path),
                "content": "",
                "imports": [],
                "classes": [],
                "functions": [],
                "dependencies": []
            }
            
        try:
            full_path = self.base_path / module_path
            if not full_path.exists():
                raise CodeAnalysisError(f"Module not found: {full_path}")
            
            self.logger.info(f"Analyzing module: {module_path}")
            
            # Read file content
            content = utils.read_file(full_path)
            
            # Parse AST
            try:
                tree = ast.parse(content)
            except SyntaxError as e:
                raise CodeAnalysisError(f"Syntax error in {module_path}: {str(e)}")
            
            # Extract module info
            imports = self._extract_imports(tree)
            classes = self._extract_classes(tree)
            functions = self._extract_functions(tree)
            
            return {
                "module_path": str(module_path),
                "content": content,
                "imports": imports,
                "classes": classes,
                "functions": functions,
                "dependencies": self._analyze_dependencies(imports)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing module {module_path}: {str(e)}")
            raise CodeAnalysisError(f"Failed to analyze module {module_path}: {str(e)}")
    
    def analyze_function(self, module_path: str, function_name: str) -> Dict[str, Any]:
        """
        Analyze a specific function within a module.
        
        Args:
            module_path: Path to the module
            function_name: Name of the function to analyze
            
        Returns:
            Dict containing function analysis information
        """
        module_info = self.analyze_module(module_path)
        
        # Find the target function
        target_func = None
        for func in module_info["functions"]:
            if func["name"] == function_name:
                target_func = func
                break
        
        if not target_func:
            raise CodeAnalysisError(
                f"Function '{function_name}' not found in module '{module_path}'"
            )
        
        # Enhance function info with call graph and complexity metrics
        target_func["complexity"] = self._analyze_complexity(target_func["ast_node"])
        target_func["calls"] = self._extract_function_calls(target_func["ast_node"])
        
        return {
            "module_info": module_info,
            "function_info": target_func
        }
    
    def _extract_imports(self, tree: ast.Module) -> List[Dict[str, Any]]:
        """Extract import statements from an AST."""
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports.append({
                        "type": "import",
                        "name": name.name,
                        "asname": name.asname
                    })
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for name in node.names:
                    imports.append({
                        "type": "import_from",
                        "module": module,
                        "name": name.name,
                        "asname": name.asname
                    })
        
        return imports
    
    def _extract_classes(self, tree: ast.Module) -> List[Dict[str, Any]]:
        """Extract class definitions from an AST."""
        classes = []
        
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                methods = []
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        methods.append({
                            "name": item.name,
                            "args": [arg.arg for arg in item.args.args],
                            "doc": ast.get_docstring(item)
                        })
                
                classes.append({
                    "name": node.name,
                    "doc": ast.get_docstring(node),
                    "methods": methods,
                    "bases": [self._get_name(base) for base in node.bases],
                    "ast_node": node
                })
        
        return classes
    
    def _extract_functions(self, tree: ast.Module) -> List[Dict[str, Any]]:
        """Extract function definitions from an AST."""
        functions = []
        
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                functions.append({
                    "name": node.name,
                    "args": [arg.arg for arg in node.args.args],
                    "doc": ast.get_docstring(node),
                    "ast_node": node
                })
        
        return functions
    
    def _analyze_dependencies(self, imports: List[Dict[str, Any]]) -> List[str]:
        """Analyze module dependencies based on imports."""
        dependencies = []
        
        for imp in imports:
            if imp["type"] == "import":
                dependencies.append(imp["name"])
            elif imp["type"] == "import_from":
                dependencies.append(imp["module"])
        
        return list(set(dependencies))
    
    def _analyze_complexity(self, node: ast.AST) -> Dict[str, int]:
        """Analyze code complexity metrics for a function."""
        visitor = ComplexityVisitor()
        visitor.visit(node)
        
        return {
            "cyclomatic_complexity": visitor.complexity,
            "line_count": visitor.line_count,
            "return_count": visitor.return_count,
            "branch_count": visitor.branch_count
        }
    
    def _extract_function_calls(self, node: ast.AST) -> List[str]:
        """Extract function calls made within a function."""
        visitor = FunctionCallVisitor()
        visitor.visit(node)
        return visitor.calls
    
    def _get_name(self, node: ast.AST) -> str:
        """Helper to get the string name from an AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        return str(node)


class ComplexityVisitor(ast.NodeVisitor):
    """AST visitor that calculates code complexity metrics."""
    
    def __init__(self):
        self.complexity = 1  # Start at 1
        self.line_count = 0
        self.return_count = 0
        self.branch_count = 0
    
    def visit_If(self, node: ast.If) -> None:
        self.complexity += 1
        self.branch_count += 1
        self.generic_visit(node)
    
    def visit_For(self, node: ast.For) -> None:
        self.complexity += 1
        self.branch_count += 1
        self.generic_visit(node)
    
    def visit_While(self, node: ast.While) -> None:
        self.complexity += 1
        self.branch_count += 1
        self.generic_visit(node)
    
    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        self.complexity += len(node.values) - 1
        self.generic_visit(node)
    
    def visit_Return(self, node: ast.Return) -> None:
        self.return_count += 1
        self.generic_visit(node)
    
    def generic_visit(self, node: ast.AST) -> None:
        if hasattr(node, 'lineno'):
            self.line_count = max(self.line_count, node.lineno)
        super().generic_visit(node)


class FunctionCallVisitor(ast.NodeVisitor):
    """AST visitor that extracts function calls."""
    
    def __init__(self):
        self.calls = []
    
    def visit_Call(self, node: ast.Call) -> None:
        func_name = ""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = f"{self._get_attribute_name(node.func)}"
        
        if func_name:
            self.calls.append(func_name)
        
        self.generic_visit(node)
    
    def _get_attribute_name(self, node: ast.Attribute) -> str:
        """Recursively get the full attribute name."""
        if isinstance(node.value, ast.Name):
            return f"{node.value.id}.{node.attr}"
        elif isinstance(node.value, ast.Attribute):
            return f"{self._get_attribute_name(node.value)}.{node.attr}"
        return node.attr