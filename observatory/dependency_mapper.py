"""
Dependency Mapper for Code State Observatory.

Enhanced version of DependencyGraph that tracks semantic relationships
between code elements across the codebase.
"""
import os
import re
from pathlib import Path
from typing import Dict, List, Set, Any, Optional, Tuple, DefaultDict, Union
from collections import defaultdict
import json
import logging
import networkx as nx
import libcst as cst
from datetime import datetime

from utils import setup_logging

logger = setup_logging(__name__)

class SemanticDependencyVisitor(cst.CSTVisitor):
    """CST visitor that extracts semantic dependencies from Python code."""
    
    def __init__(self, file_key: str):
        """Initialize with the file key for node naming."""
        self.file_key = file_key
        self.current_scope = []  # Stack of current class/function names
        self.nodes = {}  # Node ID to attributes
        self.edges = []  # (source, target) pairs
        
        # Track imports and their aliases
        self.imports = {}  # Maps alias to module
        
        # Track all defined names
        self.defined_names = set()
        
        # Track semantic information
        self.api_usage = defaultdict(list)  # Track API usage patterns
        self.error_handling = defaultdict(list)  # Track error handling patterns
        self.logging_patterns = defaultdict(list)  # Track logging patterns
    
    def visit_Import(self, node: cst.Import) -> None:
        """Process import statements."""
        for name in node.names:
            try:
                module_name = name.name.value
                alias = name.asname.name.value if name.asname and hasattr(name.asname, 'name') else module_name
                
                # Add to imports map
                self.imports[alias] = module_name
                
                # Add import node and edge
                import_key = f"{self.file_key}::import::{module_name}"
                self.nodes[import_key] = {
                    "type": "import", 
                    "name": module_name,
                    "semantic_type": "dependency"
                }
                self.edges.append((self.file_key, import_key))
                
                # Track API usage
                self.api_usage["imports"].append({
                    "module": module_name,
                    "alias": alias
                })
                
            except Exception:
                # Skip this import if we encounter errors
                continue
    
    def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
        """Process from ... import ... statements."""
        if node.module:
            # Safely extract module name based on node structure
            try:
                # Handle different module attribute structures
                if hasattr(node.module, 'names'):
                    # Regular case where node.module is a Name sequence
                    module_name = ".".join(name.value for name in node.module.names)
                elif hasattr(node.module, 'value'):
                    # Case where node.module is a simple Name
                    module_name = node.module.value
                else:
                    # Fallback
                    module_name = str(node.module)
            except Exception:
                # If we can't determine the module name, use a placeholder
                module_name = "unknown_module"
            
            for name in node.names:
                try:
                    import_name = name.name.value
                    alias = name.asname.name.value if name.asname else import_name
                    
                    # Add to imports map
                    self.imports[alias] = f"{module_name}.{import_name}"
                    
                    # Add import node and edge
                    import_key = f"{self.file_key}::import::{module_name}.{import_name}"
                    self.nodes[import_key] = {
                        "type": "import_from", 
                        "name": import_name, 
                        "module": module_name,
                        "semantic_type": "dependency"
                    }
                    self.edges.append((self.file_key, import_key))
                    
                    # Track API usage
                    self.api_usage["imports"].append({
                        "module": module_name,
                        "name": import_name,
                        "alias": alias
                    })
                    
                    # Special tracking for logging imports
                    if module_name == "logging" or import_name == "logging":
                        self.logging_patterns["imports"].append({
                            "module": module_name,
                            "name": import_name,
                            "alias": alias
                        })
                    
                except Exception:
                    # Skip this import name if we encounter errors
                    continue
    
    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        """Process class definitions."""
        class_name = node.name.value
        self.current_scope.append(class_name)
        
        # Add class node
        class_key = f"{self.file_key}::{'.'.join(self.current_scope)}"
        self.nodes[class_key] = {
            "type": "class", 
            "name": class_name,
            "semantic_type": "definition"
        }
        self.edges.append((self.file_key, class_key))
        
        # Track defined name
        self.defined_names.add(class_name)
        
        # Process base classes
        for base in node.bases:
            if isinstance(base.value, cst.Name):
                base_name = base.value.value
                if base_name in self.imports:
                    target_key = f"{self.file_key}::import::{self.imports[base_name]}"
                    self.edges.append((class_key, target_key))
                    
                    # Track semantic relationship (ensure string)
                    self.nodes[class_key]["extends"] = str(base_name)
        
    def leave_ClassDef(self, original_node: cst.ClassDef) -> None:
        """Pop the class name from the scope stack."""
        self.current_scope.pop()
    
    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        """Process function definitions."""
        func_name = node.name.value
        self.current_scope.append(func_name)
        
        # Add function node
        func_key = f"{self.file_key}::{'.'.join(self.current_scope)}"
        parent_key = f"{self.file_key}::{'.'.join(self.current_scope[:-1])}" if self.current_scope[:-1] else self.file_key
        
        # Determine if this is a method in a class
        is_method = len(self.current_scope) > 1
        
        self.nodes[func_key] = {
            "type": "function", 
            "name": func_name,
            "semantic_type": "definition",
            "is_method": is_method
        }
        self.edges.append((parent_key, func_key))
        
        # Track defined name
        self.defined_names.add(func_name)
    
    def leave_FunctionDef(self, original_node: cst.FunctionDef) -> None:
        """Pop the function name from the scope stack."""
        self.current_scope.pop()
    
    def visit_Call(self, node: cst.Call) -> None:
        """Process function/method calls."""
        try:
            # Get current scope key
            current_key = f"{self.file_key}::{'.'.join(self.current_scope)}" if self.current_scope else self.file_key
            
            # Handle simple function calls
            if isinstance(node.func, cst.Name):
                call_name = node.func.value
                
                # Check if it's an imported name
                if call_name in self.imports:
                    target_key = f"{self.file_key}::import::{self.imports[call_name]}"
                    self.edges.append((current_key, target_key))
                    
                    # Track API usage
                    self.api_usage["calls"].append({
                        "name": call_name,
                        "imported": True,
                        "scope": ".".join(self.current_scope) or "<module>"
                    })
                    
                # Check if it's a name defined in this file
                elif call_name in self.defined_names:
                    # Find the closest matching definition
                    for scope in range(len(self.current_scope), -1, -1):
                        prefix = ".".join(self.current_scope[:scope])
                        potential_key = f"{self.file_key}::{prefix}.{call_name}" if prefix else f"{self.file_key}::{call_name}"
                        if potential_key in self.nodes:
                            self.edges.append((current_key, potential_key))
                            
                            # Track internal function calls
                            self.api_usage["internal_calls"].append({
                                "name": call_name,
                                "scope": ".".join(self.current_scope) or "<module>"
                            })
                            
                            break
                
                # Special handling for print statements
                if call_name == "print":
                    scope = ".".join(self.current_scope) or "<module>"
                    self.logging_patterns["print_calls"].append({
                        "scope": scope,
                        "args_count": len(node.args)
                    })
                
            # Handle attribute calls (like module.function())
            elif isinstance(node.func, cst.Attribute):
                if isinstance(node.func.value, cst.Name):
                    obj_name = node.func.value.value
                    attr_name = node.func.attr.value
                    
                    # Track the call
                    self.api_usage["attribute_calls"].append({
                        "object": obj_name,
                        "attribute": attr_name,
                        "scope": ".".join(self.current_scope) or "<module>"
                    })
                    
                    # Special handling for logger calls
                    if obj_name == "logger":
                        log_level = attr_name
                        scope = ".".join(self.current_scope) or "<module>"
                        
                        # Check for extra parameter (structured logging)
                        has_extra = any(kw.keyword and kw.keyword.value == "extra" for kw in node.keywords)
                        
                        self.logging_patterns["logger_calls"].append({
                            "level": log_level,
                            "scope": scope,
                            "structured": has_extra,
                            "args_count": len(node.args)
                        })
        except Exception:
            # Skip this call if we encounter errors
            pass
    
    def visit_Try(self, node: cst.Try) -> None:
        """Process try/except blocks to track error handling patterns."""
        scope = ".".join(self.current_scope) or "<module>"
        
        # Track the types of exceptions being caught
        exception_types = []
        has_bare_except = False
        has_logging = False
        
        for handler in node.handlers:
            # Check exception type
            if handler.type is None:
                has_bare_except = True
                exception_type = "Exception"  # Bare except catches all exceptions
            elif isinstance(handler.type, cst.Name):
                exception_type = handler.type.value
            else:
                exception_type = "complex"  # Complex exception expression
                
            exception_types.append(exception_type)
            
            # Check for logging in the handler
            for stmt in handler.body.body:
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
        
        # Record the error handling pattern
        self.error_handling["try_except"].append({
            "scope": scope,
            "exception_types": exception_types,
            "has_bare_except": has_bare_except,
            "has_logging": has_logging
        })

class DependencyMapper:
    """Enhanced dependency mapper that tracks semantic relationships."""
    
    def __init__(self, base_path: Optional[Path] = None, storage_path: Optional[Path] = None):
        """Initialize the dependency mapper."""
        self.base_path = base_path or Path.cwd()
        self.storage_path = storage_path or (self.base_path / "observatory" / "data")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.graph = nx.DiGraph()
        self.semantic_data = {
            "api_usage": defaultdict(list),
            "error_handling": defaultdict(list),
            "logging_patterns": defaultdict(list)
        }
        
        self.logger = logger
    
    def build_from_files(self, file_paths: List[Path]) -> None:
        """Build the dependency graph from a list of files."""
        # Filter out files that should be skipped
        valid_files = [f for f in file_paths if not self._should_skip_file(f)]
        self.logger.info(f"Building dependency graph from {len(valid_files)} files (filtered from {len(file_paths)} total)")
        
        # Reset the graph and semantic data
        self.graph = nx.DiGraph()
        self.semantic_data = {
            "api_usage": defaultdict(list),
            "error_handling": defaultdict(list),
            "logging_patterns": defaultdict(list)
        }
        
        # Process each file
        for file_path in valid_files:
            try:
                self._analyze_file(file_path)
            except Exception as e:
                self.logger.warning(f"Error analyzing {file_path}: {str(e)}")
        
        self.logger.info(f"Dependency graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        
        # Save the dependency data
        self._save_dependencies()
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Determine if a file should be skipped during analysis."""
        path_str = str(file_path)
        # Skip files in virtual environments and other generated code
        skip_patterns = [
            '.venv/', 
            'site-packages/',
            '.git/',
            '__pycache__/',
            '.pytest_cache/'
        ]
        return any(pattern in path_str for pattern in skip_patterns)
    
    def _analyze_file(self, file_path: Path) -> None:
        """Analyze a file to extract dependencies and semantic relationships."""
        if not file_path.suffix == ".py":
            return

        rel_path = str(file_path.relative_to(self.base_path))
        
        try:
            # Parse the file with libcst
            with open(file_path, "r", encoding="utf-8") as f:
                code = f.read()
            
            module = cst.parse_module(code)
            
            # Add file node to the graph
            self.graph.add_node(rel_path, type="file", path=str(file_path))
            
            # Extract semantic dependencies
            visitor = SemanticDependencyVisitor(rel_path)
            module.visit(visitor)
            
            # Add extracted nodes and edges to the graph
            for node_id, attrs in visitor.nodes.items():
                # Log initial attributes for debugging
                self.logger.debug(f"Adding node {node_id} with attributes: {attrs}")
                # Sanitize attributes
                serializable_attrs = {}
                for key, value in attrs.items():
                    if isinstance(value, cst.CSTNode):
                        self.logger.warning(f"Found CSTNode in node {node_id} attribute {key}: {value}")
                        serializable_attrs[key] = str(value)
                    else:
                        serializable_attrs[key] = value
                self.graph.add_node(node_id, **serializable_attrs)
            
            for source, target in visitor.edges:
                self.graph.add_edge(source, target)
                
            # Collect semantic data (already serializable)
            self.semantic_data["api_usage"][rel_path] = dict(visitor.api_usage)
            self.semantic_data["error_handling"][rel_path] = dict(visitor.error_handling)
            self.semantic_data["logging_patterns"][rel_path] = dict(visitor.logging_patterns)
                
        except Exception as e:
            self.logger.warning(f"Error analyzing dependencies in {file_path}: {str(e)}")
    
    def _sanitize_data(self, data: Any) -> Any:
        """Recursively sanitize data to ensure JSON serializability."""
        if isinstance(data, dict):
            return {key: self._sanitize_data(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._sanitize_data(item) for item in data]
        elif isinstance(data, cst.CSTNode):
            self.logger.warning(f"Found CSTNode during serialization: {data}")
            return str(data)
        return data
    
    def _save_dependencies(self) -> None:
        """Save dependency data to disk with thorough sanitization."""
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Rebuild graph data with sanitization
        graph_data = {
            "nodes": [],
            "edges": [
                {"source": source, "target": target}
                for source, target in self.graph.edges
            ]
        }
        
        # Sanitize each node's attributes explicitly
        for node in self.graph.nodes:
            attrs = dict(self.graph.nodes[node])  # Get a fresh copy of attributes
            sanitized_attrs = {}
            for key, value in attrs.items():
                if isinstance(value, cst.CSTNode):
                    self.logger.warning(f"Found CSTNode in node {node} attribute {key} during save: {value}")
                    sanitized_attrs[key] = str(value)
                else:
                    sanitized_attrs[key] = self._sanitize_data(value)
                # Log unexpected attributes
                if key not in {"type", "name", "semantic_type", "module", "extends", "is_method", "path"}:
                    self.logger.debug(f"Unexpected attribute in node {node}: {key} = {value}")
            graph_data["nodes"].append({"id": node, **sanitized_attrs})
        
        # Combine with semantic data
        output_data = {
            "timestamp": timestamp,
            "graph": graph_data,
            "semantic_data": self._sanitize_data(dict(self.semantic_data))
        }
        
        # Save to file
        output_file = self.storage_path / f"dependencies_{timestamp}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)
            self.logger.info(f"Saved dependency data to {output_file}")
        
        # Save latest data for easy access
        latest_file = self.storage_path / "dependencies_latest.json"
        with open(latest_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)
    
    def get_dependents(self, source_file: Path, element_name: Optional[str] = None) -> Set[Tuple[Path, str]]:
        """
        Get elements that depend on a source file or element.
        
        Args:
            source_file: Path to the source file
            element_name: Optional name of a specific element within the file
            
        Returns:
            Set of (file_path, element_name) pairs that depend on the source
        """
        source_key = str(source_file.relative_to(self.base_path))
        if element_name:
            source_key = f"{source_key}::{element_name}"
        
        dependents = set()
        
        # Direct dependents
        for node in self.graph.successors(source_key):
            file_part, *element_part = node.split("::")
            element = element_part[0] if element_part else None
            dependents.add((self.base_path / file_part, element))
        
        return dependents
    
    def get_module_dependencies(self, module_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get detailed dependency information for a module.
        
        Args:
            module_path: Path to the module
            
        Returns:
            Dict containing dependency information
        """
        if isinstance(module_path, Path):
            module_path = str(module_path.relative_to(self.base_path))
        
        # Load the latest dependency data
        latest_file = self.storage_path / "dependencies_latest.json"
        if not latest_file.exists():
            return {}
            
        try:
            with open(latest_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            # Extract semantic data for this module
            api_usage = data["semantic_data"]["api_usage"].get(module_path, {})
            error_handling = data["semantic_data"]["error_handling"].get(module_path, {})
            logging_patterns = data["semantic_data"]["logging_patterns"].get(module_path, {})
            
            # Find nodes and edges for this module
            nodes = [
                node for node in data["graph"]["nodes"] 
                if node["id"] == module_path or node["id"].startswith(f"{module_path}::")
            ]
            
            edges = [
                edge for edge in data["graph"]["edges"]
                if edge["source"] == module_path or edge["source"].startswith(f"{module_path}::")
                or edge["target"] == module_path or edge["target"].startswith(f"{module_path}::")
            ]
            
            return {
                "api_usage": api_usage,
                "error_handling": error_handling,
                "logging_patterns": logging_patterns,
                "nodes": nodes,
                "edges": edges
            }
            
        except Exception as e:
            self.logger.error(f"Error loading dependency data for {module_path}: {str(e)}")
            return {}
    
    def check_module_compatibility(self, module_path: str, transformation_type: str) -> Dict[str, Any]:
        """
        Determine if a module is compatible with a specific transformation type.
        
        Args:
            module_path: Path to the module
            transformation_type: Type of transformation to check
            
        Returns:
            Dict with compatibility assessment
        """
        # Load dependency data for the module
        dependencies = self.get_module_dependencies(module_path)
        
        if not dependencies:
            return {
                "compatible": False,
                "reason": "No dependency data available"
            }
        
        # Check compatibility based on transformation type
        if transformation_type == "replace_print_with_logging":
            # Check if module has print statements
            has_print = False
            if "logging_patterns" in dependencies:
                print_calls = dependencies["logging_patterns"].get("print_calls", [])
                has_print = len(print_calls) > 0
            
            if has_print:
                return {
                    "compatible": True,
                    "reason": f"Module has {len(print_calls)} print statements to replace"
                }
            else:
                # Check if it already has logging
                has_logging = False
                if "logging_patterns" in dependencies:
                    logger_calls = dependencies["logging_patterns"].get("logger_calls", [])
                    has_logging = len(logger_calls) > 0
                
                if has_logging:
                    return {
                        "compatible": True,
                        "reason": "Module has logging calls that could be enhanced",
                        "alternative": "enhance_logging"
                    }
                else:
                    return {
                        "compatible": False,
                        "reason": "Module has no print statements to replace",
                        "alternative": "add_basic_logging"
                    }
                    
        elif transformation_type == "add_exception_handling":
            # Check if module has try/except blocks
            has_try_except = False
            if "error_handling" in dependencies:
                try_except = dependencies["error_handling"].get("try_except", [])
                has_try_except = len(try_except) > 0
            
            # Also check for functions without exception handling
            has_functions = False
            for node in dependencies.get("nodes", []):
                if node.get("type") == "function":
                    has_functions = True
                    break
            
            if has_functions and not has_try_except:
                return {
                    "compatible": True,
                    "reason": "Module has functions without exception handling"
                }
            elif has_functions and has_try_except:
                return {
                    "compatible": True,
                    "reason": "Module has functions with some exception handling that could be improved"
                }
            else:
                return {
                    "compatible": False,
                    "reason": "Module has no functions to add exception handling to"
                }
                
        elif transformation_type in ["add_structured_logging", "add_structured_logging_conditional", "add_structured_logging_error"]:
            # Check if module has logging that could be enhanced
            if "logging_patterns" in dependencies:
                logger_calls = dependencies["logging_patterns"].get("logger_calls", [])
                structured_logging = [call for call in logger_calls if call.get("structured", False)]
                simple_logging = [call for call in logger_calls if not call.get("structured", False)]
                
                if len(simple_logging) > 0:
                    return {
                        "compatible": True,
                        "reason": f"Module has {len(simple_logging)} simple logging calls that could be enhanced"
                    }
                elif len(structured_logging) > 0:
                    return {
                        "compatible": False,
                        "reason": f"Module already has {len(structured_logging)} structured logging calls",
                        "alternative": "enhance_structured_logging"
                    }
                else:
                    return {
                        "compatible": False,
                        "reason": "Module has no logging calls to enhance",
                        "alternative": "add_basic_logging"
                    }
        
        # Default case
        return {
            "compatible": False,
            "reason": f"No compatibility check defined for transformation type: {transformation_type}"
        }
    
    def find_best_candidates_for_transformation(self, transformation_type: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Find the best candidate modules for a specific transformation.
        
        Args:
            transformation_type: Type of transformation to check
            limit: Maximum number of candidates to return
            
        Returns:
            List of candidate modules with compatibility assessments
        """
        latest_file = self.storage_path / "dependencies_latest.json"
        if not latest_file.exists():
            return []
            
        try:
            with open(latest_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            # Get all module paths
            module_paths = []
            for node in data["graph"]["nodes"]:
                if node.get("type") == "file":
                    module_paths.append(node["id"])
            
            # Check compatibility for each module
            candidates = []
            for module_path in module_paths:
                compatibility = self.check_module_compatibility(module_path, transformation_type)
                if compatibility.get("compatible", False):
                    candidates.append({
                        "module_path": module_path,
                        "compatibility": compatibility
                    })
            
            # Sort by compatibility reason
            candidates.sort(key=lambda x: x["compatibility"].get("reason", ""), reverse=True)
            return candidates[:limit]
            
        except Exception as e:
            self.logger.error(f"Error finding candidates for {transformation_type}: {str(e)}")
            return []