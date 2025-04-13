"""
Pattern Registry for Code State Observatory.

Indexes and quantifies code patterns present in the codebase, providing
insights into the actual state of the code for transformation planning.
"""
import ast
import os
import re
from pathlib import Path
from typing import Dict, List, Set, Any, Optional, DefaultDict, Counter, Union, Tuple
from collections import defaultdict
import json
import logging
import libcst as cst
from datetime import datetime
import hashlib

from utils import setup_logging

logger = setup_logging(__name__)

class PatternVisitor(cst.CSTVisitor):
    """Visitor that identifies and counts various code patterns."""
    
    def __init__(self):
        """Initialize pattern visitor."""
        self.patterns = defaultdict(int)
        self.pattern_locations = defaultdict(list)
        self.current_function = []
        self.current_class = []
    
    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        """Track class definitions."""
        self.current_class.append(node.name.value)
        self.patterns["class_definitions"] += 1
        
    def leave_ClassDef(self, original_node: cst.ClassDef) -> None:
        """Exit class context."""
        self.current_class.pop()
    
    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        """Track function definitions."""
        self.current_function.append(node.name.value)
        self.patterns["function_definitions"] += 1
        
        # Check for decorators
        if node.decorators:
            self.patterns["decorated_functions"] += 1
            for decorator in node.decorators:
                if isinstance(decorator.decorator, cst.Name):
                    decorator_name = decorator.decorator.value
                    self.patterns[f"decorator_{decorator_name}"] += 1
        
        # Check for docstrings
        body = node.body.body
        if body and isinstance(body[0], cst.SimpleStatementLine):
            stmt = body[0].body[0]
            if isinstance(stmt, cst.Expr) and isinstance(stmt.value, cst.SimpleString):
                self.patterns["functions_with_docstrings"] += 1
            else:
                self.patterns["functions_without_docstrings"] += 1
        else:
            self.patterns["functions_without_docstrings"] += 1
            
    def leave_FunctionDef(self, original_node: cst.FunctionDef) -> None:
        """Exit function context."""
        self.current_function.pop()
    
    def visit_Call(self, node: cst.Call) -> None:
        """Track function calls."""
        # Track print calls
        if isinstance(node.func, cst.Name) and node.func.value == "print":
            self.patterns["print_calls"] += 1
            
            # Record location
            scope = ".".join(self.current_class + self.current_function) or "<module>"
            self.pattern_locations["print_calls"].append({
                "scope": scope,
                "args_count": len(node.args)
            })
            
        # Track logging calls
        elif (isinstance(node.func, cst.Attribute) and 
              isinstance(node.func.value, cst.Name) and 
              node.func.value.value == "logger"):
            
            log_level = node.func.attr.value if hasattr(node.func.attr, "value") else "unknown"
            self.patterns[f"logger_{log_level}_calls"] += 1
            self.patterns["total_logger_calls"] += 1
            
            # Check for structured logging (extra parameter)
            has_extra = any(kw.keyword and kw.keyword.value == "extra" for kw in node.keywords)
            if has_extra:
                self.patterns["structured_logger_calls"] += 1
            else:
                self.patterns["simple_logger_calls"] += 1
                
            # Record location
            scope = ".".join(self.current_class + self.current_function) or "<module>"
            self.pattern_locations[f"logger_{log_level}_calls"].append({
                "scope": scope,
                "structured": has_extra
            })
            
    def visit_If(self, node: cst.If) -> None:
        """Track conditional statements."""
        self.patterns["if_statements"] += 1
        
        # Check if there's logging in this if statement
        has_logging = False
        for stmt in node.body.body:
            if isinstance(stmt, cst.SimpleStatementLine):
                for expr in stmt.body:
                    if isinstance(expr, cst.Expr) and isinstance(expr.value, cst.Call):
                        if (isinstance(expr.value.func, cst.Attribute) and
                            isinstance(expr.value.func.value, cst.Name) and
                            expr.value.func.value.value == "logger"):
                            has_logging = True
                            break
            if has_logging:
                break
                
        if has_logging:
            self.patterns["if_with_logging"] += 1
        else:
            self.patterns["if_without_logging"] += 1
            
        # Record location
        scope = ".".join(self.current_class + self.current_function) or "<module>"
        self.pattern_locations["if_statements"].append({
            "scope": scope,
            "has_logging": has_logging
        })
            
    def visit_Try(self, node: cst.Try) -> None:
        """Track try-except blocks."""
        self.patterns["try_blocks"] += 1
        
        # Check for structured logging in except handlers
        for handler in node.handlers:
            has_structured_logging = False
            for stmt in handler.body.body:
                if isinstance(stmt, cst.SimpleStatementLine):
                    for expr in stmt.body:
                        if isinstance(expr, cst.Expr) and isinstance(expr.value, cst.Call):
                            if (isinstance(expr.value.func, cst.Attribute) and
                                isinstance(expr.value.func.value, cst.Name) and
                                expr.value.func.value.value == "logger"):
                                
                                # Check if it has extra parameter (structured)
                                for kw in expr.value.keywords:
                                    if kw.keyword and kw.keyword.value == "extra":
                                        has_structured_logging = True
                                        break
            
            if has_structured_logging:
                self.patterns["except_with_structured_logging"] += 1
            else:
                self.patterns["except_without_structured_logging"] += 1
        
        # Record location
        scope = ".".join(self.current_class + self.current_function) or "<module>"
        self.pattern_locations["try_blocks"].append({
            "scope": scope,
            "handlers_count": len(node.handlers)
        })

class PatternRegistry:
    """Registry for code patterns found in the codebase."""
    
    def __init__(self, base_path: Optional[Path] = None, storage_path: Optional[Path] = None):
        """Initialize the pattern registry."""
        self.base_path = base_path or Path.cwd()
        self.storage_path = storage_path or (self.base_path / "observatory" / "data")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.patterns: Dict[str, Dict[str, Any]] = defaultdict(lambda: defaultdict(int))
        self.pattern_locations: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
        self.file_hashes: Dict[str, str] = {}
        
        self.logger = logger
    
    def scan_codebase(self, file_paths: Optional[List[Path]] = None) -> Dict[str, Any]:
        """
        Scan the codebase for patterns.
        
        Args:
            file_paths: Optional list of file paths to scan. If None, scans all Python files.
            
        Returns:
            Dict containing pattern statistics
        """
        self.logger.info("Scanning codebase for patterns...")
        
        # Reset patterns
        self.patterns = defaultdict(lambda: defaultdict(int))
        self.pattern_locations = defaultdict(lambda: defaultdict(list))
        self.file_hashes = {}
        
        # Get file list if not provided
        if not file_paths:
            file_paths = list(self.base_path.glob("**/*.py"))
            self.logger.info(f"Found {len(file_paths)} Python files to scan")
        
        # Track overall statistics
        total_files = len(file_paths)
        processed_files = 0
        error_files = 0
        
        # Process each file
        for file_path in file_paths:
            rel_path = file_path.relative_to(self.base_path)
            
            # Skip files that should be ignored
            if self._should_skip_file(str(file_path)):
                continue
                
            try:
                # Generate file hash for change detection
                file_hash = self._hash_file(file_path)
                self.file_hashes[str(rel_path)] = file_hash
                
                # Read and parse the file
                with open(file_path, "r", encoding="utf-8") as f:
                    code = f.read()
                
                # Parse with libcst
                try:
                    tree = cst.parse_module(code)
                    
                    # Visit the tree to collect patterns
                    visitor = PatternVisitor()
                    tree.visit(visitor)
                    
                    # Store patterns for this file
                    self.patterns[str(rel_path)] = dict(visitor.patterns)
                    self.pattern_locations[str(rel_path)] = dict(visitor.pattern_locations)
                    
                    processed_files += 1
                    
                except Exception as e:
                    self.logger.warning(f"Error parsing {rel_path} with libcst: {str(e)}")
                    error_files += 1
                    
            except Exception as e:
                self.logger.error(f"Error processing file {rel_path}: {str(e)}")
                error_files += 1
        
        # Calculate aggregated statistics
        aggregated_stats = self._aggregate_patterns()
        
        # Save to storage
        self._save_patterns()
        
        self.logger.info(f"Codebase scan complete. Processed {processed_files} files with {error_files} errors.")
        return aggregated_stats
    
    def _should_skip_file(self, file_path: str) -> bool:
        """Determine if a file should be skipped during analysis."""
        skip_patterns = [
            '.venv/', 
            'site-packages/',
            '.git/',
            '__pycache__/',
            '.pytest_cache/'
        ]
        return any(pattern in file_path for pattern in skip_patterns)
    
    def _hash_file(self, file_path: Path) -> str:
        """Generate a hash of the file contents."""
        with open(file_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def _aggregate_patterns(self) -> Dict[str, Any]:
        """Aggregate patterns across all files."""
        aggregated = defaultdict(int)
        
        for file_path, patterns in self.patterns.items():
            for pattern, count in patterns.items():
                aggregated[pattern] += count
        
        # Add file count
        aggregated["total_files"] = len(self.patterns)
        
        return dict(aggregated)
    
    def _save_patterns(self) -> None:
        """Save patterns to disk."""
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save aggregated patterns
        aggregated = self._aggregate_patterns()
        
        output_file = self.storage_path / f"patterns_{timestamp}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": timestamp,
                "patterns": aggregated,
                "files": {
                    path: {
                        "patterns": patterns,
                        "hash": self.file_hashes.get(path, "")
                    }
                    for path, patterns in self.patterns.items()
                }
            }, f, indent=2)
            
        self.logger.info(f"Saved pattern data to {output_file}")
        
        # Save latest patterns for easy access
        latest_file = self.storage_path / "patterns_latest.json"
        with open(latest_file, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": timestamp,
                "patterns": aggregated,
                "files": {
                    path: {
                        "patterns": patterns,
                        "hash": self.file_hashes.get(path, "")
                    }
                    for path, patterns in self.patterns.items()
                }
            }, f, indent=2)
    
    def get_pattern_stats(self) -> Dict[str, Any]:
        """Get the latest pattern statistics."""
        latest_file = self.storage_path / "patterns_latest.json"
        
        if latest_file.exists():
            try:
                with open(latest_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return data.get("patterns", {})
            except Exception as e:
                self.logger.error(f"Error loading latest patterns: {str(e)}")
                return {}
        else:
            # No data yet, scan the codebase
            return self.scan_codebase()
    
    def get_pattern_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get historical pattern data.
        
        Args:
            limit: Maximum number of historical entries to return
            
        Returns:
            List of pattern statistics over time
        """
        pattern_files = sorted(self.storage_path.glob("patterns_*.json"), reverse=True)
        pattern_files = [f for f in pattern_files if f.name != "patterns_latest.json"]
        
        history = []
        for file_path in pattern_files[:limit]:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    history.append({
                        "timestamp": data.get("timestamp", ""),
                        "patterns": data.get("patterns", {})
                    })
            except Exception as e:
                self.logger.error(f"Error loading pattern history from {file_path}: {str(e)}")
        
        return history
    
    def get_patterns_for_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get patterns for a specific file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dict of patterns found in the file
        """
        if isinstance(file_path, Path):
            file_path = str(file_path.relative_to(self.base_path))
            
        latest_file = self.storage_path / "patterns_latest.json"
        
        if latest_file.exists():
            try:
                with open(latest_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                files_data = data.get("files", {})
                return files_data.get(file_path, {}).get("patterns", {})
                
            except Exception as e:
                self.logger.error(f"Error loading patterns for file {file_path}: {str(e)}")
                return {}
        else:
            return {}
    
    def has_pattern(self, pattern_name: str, file_path: Optional[Union[str, Path]] = None) -> bool:
        """
        Check if a specific pattern exists.
        
        Args:
            pattern_name: Name of the pattern to check
            file_path: Optional path to limit the check to a specific file
            
        Returns:
            True if the pattern exists, False otherwise
        """
        if file_path:
            # Check in a specific file
            file_patterns = self.get_patterns_for_file(file_path)
            return pattern_name in file_patterns and file_patterns[pattern_name] > 0
        else:
            # Check across the entire codebase
            aggregated = self.get_pattern_stats()
            return pattern_name in aggregated and aggregated[pattern_name] > 0
    
    def get_pattern_count(self, pattern_name: str, file_path: Optional[Union[str, Path]] = None) -> int:
        """
        Get the count of a specific pattern.
        
        Args:
            pattern_name: Name of the pattern to count
            file_path: Optional path to limit the count to a specific file
            
        Returns:
            Count of the pattern
        """
        if file_path:
            # Count in a specific file
            file_patterns = self.get_patterns_for_file(file_path)
            return file_patterns.get(pattern_name, 0)
        else:
            # Count across the entire codebase
            aggregated = self.get_pattern_stats()
            return aggregated.get(pattern_name, 0)
    
    def get_best_files_for_pattern(self, pattern_name: str, limit: int = 5) -> List[Tuple[str, int]]:
        """
        Find files with the highest count of a specific pattern.
        
        Args:
            pattern_name: Name of the pattern to look for
            limit: Maximum number of files to return
            
        Returns:
            List of (file_path, count) tuples
        """
        latest_file = self.storage_path / "patterns_latest.json"
        
        if latest_file.exists():
            try:
                with open(latest_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                files_data = data.get("files", {})
                pattern_counts = []
                
                for file_path, file_data in files_data.items():
                    patterns = file_data.get("patterns", {})
                    count = patterns.get(pattern_name, 0)
                    if count > 0:
                        pattern_counts.append((file_path, count))
                
                # Sort by count descending
                pattern_counts.sort(key=lambda x: x[1], reverse=True)
                return pattern_counts[:limit]
                
            except Exception as e:
                self.logger.error(f"Error finding best files for pattern {pattern_name}: {str(e)}")
                return []
        else:
            return []