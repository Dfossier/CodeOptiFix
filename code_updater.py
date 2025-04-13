"""
Code Updater module for the Self-Improving AI Assistant.

Provides safe, AST-based code transformation capabilities for self-improvement.
Uses libcst for precise code modifications with dependency tracking to ensure
all related parts of the codebase are consistently updated.
"""
import os
import sys
import re
import json
import shutil
import tempfile
import subprocess
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Tuple, Union, Callable
from enum import Enum
import importlib
import asyncio
from datetime import datetime
import uuid
import libcst as cst
from libcst.metadata import MetadataWrapper, PositionProvider
import networkx as nx
import config
import utils
from utils import setup_logging, CodeUpdateError
from interfaces import ImprovementGoal, CodeCandidate
logger = setup_logging(__name__)

class UpdateStatus(Enum):
    """Status of a code update."""
    PENDING = 'pending'
    APPLIED = 'applied'
    VALIDATED = 'validated'
    SYNTAX_ERROR = 'syntax_error'
    STATIC_CHECK_FAILED = 'static_check_failed'
    TEST_FAILED = 'test_failed'
    GOAL_NOT_MET = 'goal_not_met'
    ROLLED_BACK = 'rolled_back'

class CodeTransformation:
    """Represents a specific code transformation."""
    def __init__(self, file_path: Union[str, Path], transformation_type: str, description: str, original_code: Optional[str]=None, transformed_code: Optional[str]=None, metadata: Optional[Dict[str, Any]]=None):
        """Initialize a code transformation."""
        self.id = str(uuid.uuid4())
        self.file_path = Path(file_path) if isinstance(file_path, str) else file_path
        self.transformation_type = transformation_type
        self.description = description
        self.original_code = original_code
        self.transformed_code = transformed_code
        self.metadata = metadata or {}
        self.status = UpdateStatus.PENDING
        self.error_message = None
        self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'id': self.id,
            'file_path': str(self.file_path),
            'transformation_type': self.transformation_type,
            'description': self.description,
            'status': self.status.value,
            'error_message': self.error_message,
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }

    def set_status(self, status: UpdateStatus, error_message: Optional[str]=None) -> None:
        """Update the status of this transformation."""
        self.status = status
        self.error_message = error_message
        logger.info(f'Transformation {self.id[:8]} status set to {status.value}')
        if error_message:
            logger.warning(f'Error: {error_message}')

class DependencyGraph:
    """Tracks dependencies between code elements across files."""
    def __init__(self, base_path: Optional[Path]=None):
        """Initialize the dependency graph."""
        self.base_path = base_path or Path.cwd()
        self.graph = nx.DiGraph()
        self.logger = logger

    def build_from_files(self, file_paths: List[Path]) -> None:
        """Build the dependency graph from a list of files."""
        valid_files = [f for f in file_paths if not self._should_skip_file(f)]
        self.logger.info(f'Building dependency graph from {len(valid_files)} files (filtered from {len(file_paths)} total)')
        for file_path in valid_files:
            try:
                self._analyze_file(file_path)
            except Exception as e:
                self.logger.warning(f'Error analyzing {file_path}: {str(e)}')
        self.logger.info(f'Dependency graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges')

    def _should_skip_file(self, file_path: Path) -> bool:
        """Determine if a file should be skipped during analysis."""
        path_str = str(file_path)
        skip_patterns = ['.venv/', 'site-packages/', '.git/', '__pycache__/', '.pytest_cache/']
        return any(pattern in path_str for pattern in skip_patterns)

    def get_dependents(self, source_file: Path, element_name: Optional[str]=None) -> Set[Tuple[Path, str]]:
        """Get elements that depend on a source file or element."""
        source_key = str(source_file.relative_to(self.base_path))
        if element_name:
            source_key = f'{source_key}::{element_name}'
        dependents = set()
        for node in self.graph.successors(source_key):
            file_part, *element_part = node.split('::')
            element = element_part[0] if element_part else None
            dependents.add((self.base_path / file_part, element))
        return dependents

    def _analyze_file(self, file_path: Path) -> None:
        """Analyze a file to extract dependencies."""
        if not file_path.suffix == '.py':
            return
        rel_path = str(file_path.relative_to(self.base_path))
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            module = cst.parse_module(code)
            self.graph.add_node(rel_path, type='file', path=str(file_path))
            visitor = DependencyVisitor(rel_path)
            module.visit(visitor)
            for node, attrs in visitor.nodes.items():
                self.graph.add_node(node, **attrs)
            for source, target in visitor.edges:
                self.graph.add_edge(source, target)
        except Exception as e:
            self.logger.warning(f'Error analyzing dependencies in {file_path}: {str(e)}')

    def visualize(self, output_path: Path=None) -> None:
        """Generate a visualization of the dependency graph."""
        try:
            import matplotlib.pyplot as plt
            output_path = output_path or self.base_path / 'dependency_graph.png'
            plt.figure(figsize=(12, 10))
            pos = nx.spring_layout(self.graph)
            nx.draw(self.graph, pos, with_labels=True, node_color='skyblue', node_size=1500, font_size=8, font_weight='bold', alpha=0.8)
            plt.savefig(output_path)
            self.logger.info(f'Dependency graph visualization saved to {output_path}')
        except ImportError:
            self.logger.warning('Cannot visualize dependency graph: matplotlib not installed')
        except Exception as e:
            self.logger.warning(f'Error visualizing dependency graph: {str(e)}')

class DependencyVisitor(cst.CSTVisitor):
    """CST visitor that extracts dependencies from Python code."""
    def __init__(self, file_key: str):
        """Initialize with the file key for node naming."""
        self.file_key = file_key
        self.current_scope = []
        self.nodes = {}
        self.edges = []
        self.imports = {}
        self.defined_names = set()

    def visit_Import(self, node: cst.Import) -> None:
        """Process import statements."""
        for name in node.names:
            try:
                module_name = name.name.value
                alias = name.asname.name.value if name.asname and hasattr(name.asname, 'name') else module_name
                self.imports[alias] = module_name
                import_key = f'{self.file_key}::import::{module_name}'
                self.nodes[import_key] = {'type': 'import', 'name': module_name}
                self.edges.append((self.file_key, import_key))
            except Exception:
                continue

    def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
        """Process from ... import ... statements."""
        if node.module:
            try:
                if hasattr(node.module, 'names'):
                    module_name = '.'.join((name.value for name in node.module.names))
                elif hasattr(node.module, 'value'):
                    module_name = node.module.value
                else:
                    module_name = str(node.module)
            except Exception:
                module_name = 'unknown_module'
            for name in node.names:
                try:
                    import_name = name.name.value
                    alias = name.asname.name.value if name.asname else import_name
                    self.imports[alias] = f'{module_name}.{import_name}'
                    import_key = f'{self.file_key}::import::{module_name}.{import_name}'
                    self.nodes[import_key] = {'type': 'import_from', 'name': import_name, 'module': module_name}
                    self.edges.append((self.file_key, import_key))
                except Exception:
                    continue

    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        """Process class definitions."""
        class_name = node.name.value
        self.current_scope.append(class_name)
        class_key = f'{self.file_key}::{'.'.join(self.current_scope)}'
        self.nodes[class_key] = {'type': 'class', 'name': class_name}
        self.edges.append((self.file_key, class_key))
        self.defined_names.add(class_name)
        for base in node.bases:
            if isinstance(base.value, cst.Name):
                base_name = base.value.value
                if base_name in self.imports:
                    target_key = f'{self.file_key}::import::{self.imports[base_name]}'
                    self.edges.append((class_key, target_key))

    def leave_ClassDef(self, original_node: cst.ClassDef) -> None:
        """Pop the class name from the scope stack."""
        self.current_scope.pop()

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        """Process function definitions."""
        func_name = node.name.value
        self.current_scope.append(func_name)
        func_key = f'{self.file_key}::{'.'.join(self.current_scope)}'
        parent_key = f'{self.file_key}::{'.'.join(self.current_scope[:-1])}' if self.current_scope[:-1] else self.file_key
        self.nodes[func_key] = {'type': 'function', 'name': func_name}
        self.edges.append((parent_key, func_key))
        self.defined_names.add(func_name)

    def leave_FunctionDef(self, original_node: cst.FunctionDef) -> None:
        """Pop the function name from the scope stack."""
        self.current_scope.pop()

    def visit_Call(self, node: cst.Call) -> None:
        """Process function/method calls."""
        try:
            if isinstance(node.func, cst.Name):
                call_name = node.func.value
                current_key = f'{self.file_key}::{'.'.join(self.current_scope)}' if self.current_scope else self.file_key
                if call_name in self.imports:
                    target_key = f'{self.file_key}::import::{self.imports[call_name]}'
                    self.edges.append((current_key, target_key))
                elif call_name in self.defined_names:
                    for scope in range(len(self.current_scope), -1, -1):
                        prefix = '.'.join(self.current_scope[:scope])
                        potential_key = f'{self.file_key}::{prefix}.{call_name}' if prefix else f'{self.file_key}::{call_name}'
                        if potential_key in self.nodes:
                            self.edges.append((current_key, potential_key))
                            break
            elif isinstance(node.func, cst.Attribute):
                pass
        except Exception:
            pass

    def visit_Attribute(self, node: cst.Attribute) -> None:
        """Process attribute access."""
        pass

class TransformationVisitor(cst.CSTTransformer):
    """Base class for code transformations using libcst."""
    def __init__(self, transformation_data: Dict[str, Any]=None):
        """Initialize with optional transformation data."""
        self.transformation_data = transformation_data or {}
        self.changes_made = False
        self.decision_log = []
        self.logger = logger

    def log_decision(self, node_type: str, decision: str, reason: str) -> None:
        """Log information about transformation decisions for diagnostics."""
        self.decision_log.append({'node_type': node_type, 'decision': decision, 'reason': reason})
        self.logger.debug(f'Transformation decision: {node_type} was {decision} because {reason}')

class CodeUpdater:
    """Core class for safely updating code using AST-based transformations."""
    def __init__(self, base_path: Optional[Path]=None, transforms_dir: Optional[Path]=None, sandbox_dir: Optional[Path]=None, static_checkers: Optional[List[str]]=None, test_command: Optional[str]=None):
        """Initialize the CodeUpdater."""
        self.base_path = base_path or Path.cwd()
        self.transforms_dir = transforms_dir or self.base_path / 'transforms'
        self.sandbox_dir = sandbox_dir or self.base_path / 'sandbox'
        self.static_checkers = static_checkers or ['mypy', 'flake8']
        self.test_command = test_command or 'pytest'
        self.logger = logger
        self.dependency_graph = DependencyGraph(self.base_path)
        self.transformations = []
        self.transforms_dir.mkdir(exist_ok=True)
        self.transformation_factories = {
            'replace_print_with_logging': self._create_print_to_logging_transformer,
            'add_exception_handling': self._create_exception_handling_transformer,
            'extract_function': self._create_function_extraction_transformer,
            'split_file': self._create_file_split_transformer,
            'add_structured_logging': self._create_structured_logging_transformer,
            'add_structured_logging_conditional': self._create_structured_logging_transformer,
            'add_structured_logging_error': self._create_structured_logging_transformer,
            'optimize_string_formatting': self._create_string_format_transformer
        }

    def _load_code(self, file_path: Path) -> cst.Module:
        """Load and parse Python code from a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            return cst.parse_module(code)
        except Exception as e:
            self.logger.error(f'Error loading code from {file_path}: {str(e)}')
            raise CodeUpdateError(f'Failed to load code from {file_path}: {str(e)}')

    def _save_code(self, file_path: Path, module: cst.Module) -> None:
        """Save modified code back to a file."""
        try:
            code = module.code
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(code)
            self.logger.info(f'Updated code saved to {file_path}')
        except Exception as e:
            self.logger.error(f'Error saving code to {file_path}: {str(e)}')
            raise CodeUpdateError(f'Failed to save code to {file_path}: {str(e)}')

    async def update_codebase(self, goals: List[ImprovementGoal]) -> List[Dict[str, Any]]:
        """Update the codebase based on improvement goals."""
        results = []
        try:
            python_files = list(self.base_path.glob('**/*.py'))
            self.logger.info(f'Found {len(python_files)} Python files in codebase')
            self.dependency_graph.build_from_files(python_files)
            for goal in goals:
                self.logger.info(f'Processing goal: {goal.description}')
                try:
                    transformations = await self._generate_plan(goal)
                    if not transformations:
                        self.logger.warning(f'No transformations generated for goal: {goal.description}')
                        results.append({'goal': goal.to_dict(), 'status': 'no_transformations', 'message': 'No transformations were generated'})
                        continue
                    status, message, applied = self._apply_transformations(transformations)
                    results.append({'goal': goal.to_dict(), 'status': status, 'message': message, 'transformations': [t.to_dict() for t in applied]})
                except Exception as e:
                    self.logger.error(f"Error processing goal '{goal.description}': {str(e)}")
                    results.append({'goal': goal.to_dict(), 'status': 'error', 'message': f'Error: {str(e)}'})
            return results
        except Exception as e:
            self.logger.error(f'Error updating codebase: {str(e)}')
            raise CodeUpdateError(f'Failed to update codebase: {str(e)}')

    async def _generate_plan(self, goal: ImprovementGoal) -> List[CodeTransformation]:
        """Generate a transformation plan for an improvement goal."""
        transformations = []
        transformation_type = self._determine_transformation_type(goal)
        if not transformation_type:
            self.logger.warning(f'No transformation type determined for goal: {goal.description}')
            return []
        target_file = self.base_path / goal.target_module
        if not target_file.exists():
            raise CodeUpdateError(f'Target module not found: {goal.target_module}')
        transformation = CodeTransformation(file_path=target_file, transformation_type=transformation_type, description=goal.description)
        transformation.metadata.update({'goal_id': id(goal), 'target_function': goal.target_function, 'priority': goal.priority})
        with open(target_file, 'r', encoding='utf-8') as f:
            original_code = f.read()
            transformation.original_code = original_code
        transformations.append(transformation)
        if transformation_type == 'split_file':
            pass
        dependents = self.dependency_graph.get_dependents(target_file, goal.target_function)
        for dep_file, element in dependents:
            if any((t.file_path == dep_file for t in transformations)):
                continue
            dependent_transformation = CodeTransformation(file_path=dep_file, transformation_type=f'update_for_{transformation_type}', description=f'Update {dep_file.name} for changes in {target_file.name}')
            dependent_transformation.metadata.update({'primary_transformation_id': transformation.id, 'dependent_element': element})
            with open(dep_file, 'r', encoding='utf-8') as f:
                dependent_transformation.original_code = f.read()
            transformations.append(dependent_transformation)
        return transformations

    def _determine_transformation_type(self, goal: ImprovementGoal) -> Optional[str]:
        """Determine which transformation type to use based on the goal description."""
        description = goal.description.lower()
        if 'string concatenation' in description or 'string formatting' in description:
            return 'optimize_string_formatting'
        elif 'print' in description and ('log' in description or 'logging' in description):
            return 'replace_print_with_logging'
        elif 'structured logging' in description:
            if 'conditional logic' in description:
                return 'add_structured_logging_conditional'
            elif 'error handling' in description:
                return 'add_structured_logging_error'
            else:
                return 'add_structured_logging'
        elif 'exception' in description or 'error handling' in description or 'try' in description:
            return 'add_exception_handling'
        elif 'extract' in description or 'split' in description or 'refactor' in description:
            if 'file' in description or 'module' in description:
                return 'split_file'
            else:
                return 'extract_function'
        return None

    def _apply_transformations(self, transformations: List[CodeTransformation]) -> Tuple[str, str, List[CodeTransformation]]:
        """Apply a list of transformations to the codebase."""
        self.logger.info(f'Applying {len(transformations)} transformations')
        with tempfile.TemporaryDirectory() as sandbox_dir:
            sandbox_path = Path(sandbox_dir)
            self._create_sandbox(sandbox_path)
            applied_transformations = []
            for transformation in transformations:
                try:
                    sandbox_file = sandbox_path / transformation.file_path.relative_to(self.base_path)
                    transformer = self._create_transformer(transformation)
                    if not transformer:
                        transformation.set_status(UpdateStatus.ROLLED_BACK, f'No transformer available for type: {transformation.transformation_type}')
                        continue
                    module = self._load_code(sandbox_file)
                    self.logger.info(f'Applying transformer {transformer.__class__.__name__} to {transformation.file_path}')
                    if not hasattr(transformer, 'nodes_visited'):
                        transformer.nodes_visited = {
                            'total': 0,
                            'print_calls': 0,
                            'logger_calls': 0,
                            'if_statements': 0,
                            'except_handlers': 0,
                            'string_concatenations': 0,
                            'potential_targets': 0
                        }

                    class DiagnosticVisitor(cst.CSTVisitor):
                        def __init__(self, transformer):
                            self.transformer = transformer

                        def on_visit(self, node):
                            self.transformer.nodes_visited['total'] += 1
                            if isinstance(node, cst.Call):
                                if isinstance(node.func, cst.Name) and node.func.value == 'print':
                                    self.transformer.nodes_visited['print_calls'] += 1
                                    self.transformer.nodes_visited['potential_targets'] += 1
                                elif isinstance(node.func, cst.Attribute) and isinstance(node.func.value, cst.Name) and (node.func.value.value == 'logger'):
                                    self.transformer.nodes_visited['logger_calls'] += 1
                                    self.transformer.nodes_visited['potential_targets'] += 1
                            elif isinstance(node, cst.If):
                                self.transformer.nodes_visited['if_statements'] += 1
                                if isinstance(self.transformer, StructuredLoggingConditionalTransformer):
                                    self.transformer.nodes_visited['potential_targets'] += 1
                            elif isinstance(node, cst.ExceptHandler):
                                self.transformer.nodes_visited['except_handlers'] += 1
                                if isinstance(self.transformer, (StructuredLoggingErrorTransformer, ExceptionHandlingTransformer)):
                                    self.transformer.nodes_visited['potential_targets'] += 1
                            elif isinstance(node, cst.BinaryOperation) and isinstance(node.operator, cst.Add):
                                if isinstance(node.left, cst.SimpleString) or isinstance(node.right, cst.SimpleString):
                                    self.transformer.nodes_visited['string_concatenations'] += 1
                                    if isinstance(self.transformer, StringFormatTransformer):
                                        self.transformer.nodes_visited['potential_targets'] += 1
                            return True

                    module.visit(DiagnosticVisitor(transformer))
                    self.logger.info(f'Transformer stats: visited {transformer.nodes_visited["total"]} nodes, found {transformer.nodes_visited["potential_targets"]} potential targets (prints: {transformer.nodes_visited["print_calls"]}, logger calls: {transformer.nodes_visited["logger_calls"]}, if statements: {transformer.nodes_visited["if_statements"]}, except handlers: {transformer.nodes_visited["except_handlers"]}, string concatenations: {transformer.nodes_visited["string_concatenations"]})')
                    transformed_module = module.visit(transformer)
                    if hasattr(transformer, 'decision_log') and transformer.decision_log:
                        decision_counts = {}
                        for decision in transformer.decision_log:
                            key = f'{decision["node_type"]}:{decision["decision"]}'
                            decision_counts[key] = decision_counts.get(key, 0) + 1
                        decision_summary = ', '.join([f'{count} {key}' for key, count in decision_counts.items()])
                        self.logger.info(f'Transformer decisions: {decision_summary}')
                        detailed_decisions = []
                        for i, decision in enumerate(transformer.decision_log):
                            if i >= 10:
                                detailed_decisions.append('... more decisions omitted ...')
                                break
                            detailed_decisions.append(f'{decision["node_type"]} {decision["decision"]}: {decision["reason"]}')
                        for decision in detailed_decisions:
                            self.logger.debug(f'Decision detail: {decision}')
                    if not transformer.changes_made:
                        self.logger.warning(f'No changes made by transformer for {transformation.file_path} despite finding {transformer.nodes_visited["potential_targets"]} potential targets')
                        reason = 'No suitable targets found'
                        if transformer.nodes_visited['potential_targets'] > 0:
                            if isinstance(transformer, PrintToLoggingTransformer):
                                if transformer.nodes_visited['print_calls'] == 0:
                                    reason = 'No print statements found to convert'
                                else:
                                    reason = 'Print statements were found but may already have logger equivalents'
                            elif isinstance(transformer, StructuredLoggingErrorTransformer):
                                reason = 'Found except handlers but they may already have structured logging'
                            elif isinstance(transformer, StructuredLoggingConditionalTransformer):
                                reason = 'Found if statements but they may not be suitable for logging'
                            elif isinstance(transformer, StringFormatTransformer):
                                if transformer.nodes_visited['string_concatenations'] == 0:
                                    reason = 'No string concatenations found to optimize'
                                else:
                                    reason = 'String concatenations were found but may not be suitable for optimization'
                        transformation.set_status(UpdateStatus.ROLLED_BACK, f'No changes were made by the transformer: {reason}')
                        continue
                    transformation.transformed_code = transformed_module.code
                    self._save_code(sandbox_file, transformed_module)
                    transformation.set_status(UpdateStatus.APPLIED)
                    applied_transformations.append(transformation)
                except Exception as e:
                    self.logger.error(f'Error applying transformation to {transformation.file_path}: {str(e)}')
                    transformation.set_status(UpdateStatus.ROLLED_BACK, f'Error applying transformation: {str(e)}')
            if not applied_transformations:
                return ('no_changes', 'No changes were applied', [])
            validation_status, validation_message = self._validate(sandbox_path, applied_transformations)
            if validation_status != UpdateStatus.VALIDATED:
                for t in applied_transformations:
                    t.set_status(validation_status, validation_message)
                return ('validation_failed', validation_message, applied_transformations)
            for transformation in applied_transformations:
                if transformation.status == UpdateStatus.VALIDATED:
                    with open(transformation.file_path, 'w', encoding='utf-8') as f:
                        f.write(transformation.transformed_code)
                    if transformation.transformation_type == 'split_file' and 'new_files' in transformation.metadata:
                        self.logger.info(f'Creating new files for split file {transformation.file_path}')
                        target_dir = transformation.file_path.parent
                        package_name = transformation.file_path.stem
                        package_dir = target_dir / package_name
                        package_dir.mkdir(exist_ok=True)
                        for file_name, content in transformation.metadata['new_files'].items():
                            new_file_path = package_dir / file_name
                            with open(new_file_path, 'w', encoding='utf-8') as f:
                                f.write(content)
                            self.logger.info(f'Created new file: {new_file_path}')
            return ('success', f'Successfully applied {len(applied_transformations)} transformations', applied_transformations)

    def _create_sandbox(self, sandbox_path: Path) -> None:
        """Create a sandbox copy of the codebase for testing.""" 
        self.logger.info(f'Creating sandbox at {sandbox_path}')
        for src_path in self.base_path.glob('**/*.py'):
            if not any((p in str(src_path) for p in ['.git', '__pycache__', 'venv', 'env'])):
                rel_path = src_path.relative_to(self.base_path)
                dst_path = sandbox_path / rel_path
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, dst_path)
        for file_name in ['setup.py', 'requirements.txt', 'pyproject.toml', 'pytest.ini']:
            src_path = self.base_path / file_name
            if src_path.exists():
                dst_path = sandbox_path / file_name
                shutil.copy2(src_path, dst_path)
        self.logger.info('Sandbox created successfully')

    def _validate(self, sandbox_path: Path, transformations: List[CodeTransformation]) -> Tuple[UpdateStatus, str]:
        """Validate the transformed code in the sandbox."""
        self.logger.info('Validating transformed code')
        for transformation in transformations:
            sandbox_file = sandbox_path / transformation.file_path.relative_to(self.base_path)
            try:
                with open(sandbox_file, 'r', encoding='utf-8') as f:
                    code = f.read()
                ast.parse(code)
            except SyntaxError as e:
                return (UpdateStatus.SYNTAX_ERROR, f'Syntax error in {transformation.file_path}: {str(e)}')
        for checker in self.static_checkers:
            try:
                cmd = [checker]
                for transformation in transformations:
                    rel_path = transformation.file_path.relative_to(self.base_path)
                    sandbox_file = sandbox_path / rel_path
                    cmd.append(str(sandbox_file))
                self.logger.info(f'Running static checker: {' '.join(cmd)}')
                result = subprocess.run(cmd, cwd=sandbox_path, capture_output=True, text=True, check=False)
                if result.returncode != 0:
                    self.logger.warning(f'{checker} issues found:\n{result.stdout}\n{result.stderr}')
                    return (UpdateStatus.STATIC_CHECK_FAILED, f'{checker} found issues with the code')
            except Exception as e:
                self.logger.warning(f'Error running {checker}: {str(e)}')
        if self.test_command:
            try:
                cmd = self.test_command.split()
                self.logger.info(f'Running tests: {' '.join(cmd)}')
                result = subprocess.run(cmd, cwd=sandbox_path, capture_output=True, text=True, check=False)
                if result.returncode != 0:
                    self.logger.warning(f'Tests failed:\n{result.stdout}\n{result.stderr}')
                    return (UpdateStatus.TEST_FAILED, 'Tests failed after applying transformations')
            except Exception as e:
                self.logger.warning(f'Error running tests: {str(e)}')
                return (UpdateStatus.TEST_FAILED, f'Error running tests: {str(e)}')
        for transformation in transformations:
            if transformation.transformation_type == 'replace_print_with_logging':
                sandbox_file = sandbox_path / transformation.file_path.relative_to(self.base_path)
                with open(sandbox_file, 'r', encoding='utf-8') as f:
                    code = f.read()
                if re.search('print\\s*\\(', code):
                    return (UpdateStatus.GOAL_NOT_MET, 'Some print statements were not replaced with logging')
            elif transformation.transformation_type == 'optimize_string_formatting':
                sandbox_file = sandbox_path / transformation.file_path.relative_to(self.base_path)
                with open(sandbox_file, 'r', encoding='utf-8') as f:
                    code = f.read()
                if re.search(r'[\'\"]\s*\+\s*[\'\"]', code):
                    return (UpdateStatus.GOAL_NOT_MET, 'Some string concatenations were not optimized')
        for transformation in transformations:
            transformation.set_status(UpdateStatus.VALIDATED)
        return (UpdateStatus.VALIDATED, 'All validations passed')

    def _create_transformer(self, transformation: CodeTransformation) -> Optional[TransformationVisitor]:
        """Create a transformer for a specific transformation."""
        transformation_type = transformation.transformation_type
        if transformation_type in self.transformation_factories:
            factory = self.transformation_factories[transformation_type]
            return factory(transformation)
        if transformation_type.startswith('update_for_'):
            base_type = transformation_type[len('update_for_'):]
            if base_type in self.transformation_factories:
                factory = self.transformation_factories[base_type]
                return factory(transformation, is_dependent=True)
        self.logger.warning(f'No transformer available for type: {transformation_type}')
        return None

    def _create_print_to_logging_transformer(self, transformation: CodeTransformation, is_dependent: bool=False) -> TransformationVisitor:
        """Create a transformer that replaces print statements with logging."""
        return PrintToLoggingTransformer(transformation.metadata)

    def _create_exception_handling_transformer(self, transformation: CodeTransformation, is_dependent: bool=False) -> TransformationVisitor:
        """Create a transformer that adds exception handling."""
        return ExceptionHandlingTransformer(transformation.metadata)

    def _create_function_extraction_transformer(self, transformation: CodeTransformation, is_dependent: bool=False) -> TransformationVisitor:
        """Create a transformer that extracts code into separate functions."""
        return FunctionExtractionTransformer(transformation.metadata)

    def _create_file_split_transformer(self, transformation: CodeTransformation, is_dependent: bool=False) -> TransformationVisitor:
        """Create a transformer that splits a file into multiple files."""
        return FileSplitTransformer(transformation.metadata)

    def _create_structured_logging_transformer(self, transformation: CodeTransformation, is_dependent: bool=False) -> TransformationVisitor:
        """Create a transformer that adds structured logging."""
        if 'conditional' in transformation.transformation_type:
            return StructuredLoggingConditionalTransformer(transformation.metadata)
        elif 'error' in transformation.transformation_type:
            return StructuredLoggingErrorTransformer(transformation.metadata)
        else:
            return StructuredLoggingTransformer(transformation.metadata)

    def _create_string_format_transformer(self, transformation: CodeTransformation, is_dependent: bool=False) -> TransformationVisitor:
        """Create a transformer that optimizes string concatenations."""
        return StringFormatTransformer(transformation.metadata)

class PrintToLoggingTransformer(TransformationVisitor):
    """Transforms print statements to logging.info/error/debug calls and enhances existing logging."""
    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        """Track the current function name for context."""
        if not hasattr(self, 'current_function'):
            self.current_function = []
        self.current_function.append(node.name.value)

    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
        """Clear the current function name when leaving the function."""
        if hasattr(self, 'current_function') and self.current_function and (self.current_function[-1] == original_node.name.value):
            self.current_function.pop()
        return updated_node

    def leave_Call(self, original_node: cst.Call, updated_node: cst.Call) -> cst.Call:
        """Replace print calls with logging calls or enhance existing logging."""
        try:
            if isinstance(original_node.func, cst.Name) and original_node.func.value == 'print':
                self.log_decision('print', 'analyzing', 'found print statement')
                log_level = 'info'
                log_content = 'unknown'
                if original_node.args:
                    arg_value = None
                    if isinstance(original_node.args[0].value, cst.SimpleString):
                        arg_value = original_node.args[0].value.value
                        log_content = arg_value[:30] + '...' if len(arg_value) > 30 else arg_value
                    if arg_value and any((kw in arg_value.lower() for kw in ['error', 'exception', 'fail', 'failed'])):
                        log_level = 'error'
                        self.log_decision('print', 'classified', f"identified as error log: '{log_content}'")
                    elif arg_value and any((kw in arg_value.lower() for kw in ['debug', 'trace'])):
                        log_level = 'debug'
                        self.log_decision('print', 'classified', f"identified as debug log: '{log_content}'")
                    elif arg_value and any((kw in arg_value.lower() for kw in ['warn', 'warning'])):
                        log_level = 'warning'
                        self.log_decision('print', 'classified', f"identified as warning log: '{log_content}'")
                    else:
                        self.log_decision('print', 'classified', f"default to info log: '{log_content}'")
                func_context = None
                if hasattr(self, 'current_function') and self.current_function:
                    func_context = self.current_function[-1]
                    self.log_decision('print', 'context', f"adding function context: '{func_context}'")
                else:
                    self.log_decision('print', 'context', 'no function context available')
                self.changes_made = True
                self.log_decision('print', 'transformed', f'converted to logger.{log_level}')
                if func_context:
                    return cst.Call(
                        func=cst.Attribute(value=cst.Name('logger'), attr=cst.Name(log_level)),
                        args=original_node.args,
                        keywords=[
                            cst.Arg(
                                keyword=cst.Name('extra'),
                                value=cst.Dict([
                                    cst.DictElement(
                                        key=cst.SimpleString("'function'"),
                                        value=cst.SimpleString(f"'{func_context}'")
                                    )
                                ])
                            )
                        ] + list(original_node.keywords)
                    )
                else:
                    return cst.Call(
                        func=cst.Attribute(value=cst.Name('logger'), attr=cst.Name(log_level)),
                        args=original_node.args,
                        keywords=original_node.keywords
                    )
            elif isinstance(original_node.func, cst.Attribute) and isinstance(original_node.func.value, cst.Name) and (original_node.func.value.value == 'logger'):
                self.log_decision('logger', 'analyzing', 'found existing logger call')
                log_level = original_node.func.attr.value if hasattr(original_node.func.attr, 'value') else 'unknown'
                self.log_decision('logger', 'info', f'log level: {log_level}')
                if not any((kw.keyword and kw.keyword.value == 'extra' for kw in original_node.keywords)):
                    self.log_decision('logger', 'enhancing', "no 'extra' parameter found, will add context")
                    func_context = None
                    if hasattr(self, 'current_function') and self.current_function:
                        func_context = self.current_function[-1]
                        self.log_decision('logger', 'context', f"adding function context: '{func_context}'")
                    else:
                        self.log_decision('logger', 'context', 'no function context available')
                    if func_context:
                        self.changes_made = True
                        self.log_decision('logger', 'transformed', 'added extra context parameter')
                        return updated_node.with_changes(
                            keywords=list(updated_node.keywords) + [
                                cst.Arg(
                                    keyword=cst.Name('extra'),
                                    value=cst.Dict([
                                        cst.DictElement(
                                            key=cst.SimpleString("'function'"),
                                            value=cst.SimpleString(f"'{func_context}'")
                                        ),
                                        cst.DictElement(
                                            key=cst.SimpleString("'context'"),
                                            value=cst.SimpleString("'error_handling'")
                                        )
                                    ])
                                )
                            ]
                        )
                else:
                    self.log_decision('logger', 'skipped', "already has 'extra' parameter with context")
            return updated_node
        except Exception as e:
            self.log_decision('error', 'exception', f'Error in PrintToLoggingTransformer: {str(e)}')
            self.logger.warning(f'Error in PrintToLoggingTransformer: {str(e)}')
            return updated_node

    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
        """Ensure the module has the necessary logging imports."""
        if not self.changes_made:
            return updated_node
        has_logging_import = False
        has_setup_logging_import = False
        for statement in original_node.body:
            if isinstance(statement, cst.SimpleStatementLine):
                for stmt in statement.body:
                    if isinstance(stmt, cst.Import):
                        for name in stmt.names:
                            if name.name.value == 'logging':
                                has_logging_import = True
                    elif isinstance(stmt, cst.ImportFrom):
                        if stmt.module and stmt.module.value == 'utils':
                            for name in stmt.names:
                                if name.name.value == 'setup_logging':
                                    has_setup_logging_import = True
        imports_to_add = []
        if not has_logging_import:
            imports_to_add.append(cst.SimpleStatementLine(body=[cst.Import(names=[cst.ImportAlias(name=cst.Name('logging'))])]))
        if not has_setup_logging_import:
            try:
                import_from = cst.ImportFrom(module=cst.Name('utils'), names=[cst.ImportAlias(name=cst.Name('setup_logging'))])
                imports_to_add.append(cst.SimpleStatementLine(body=[import_from]))
            except AttributeError:
                self.logger.warning('Using fallback for ImportFrom construction')
                import_str = 'from utils import setup_logging'
                module = cst.parse_module(import_str)
                if module.body:
                    imports_to_add.append(module.body[0])
            except Exception as e:
                self.logger.warning(f'Failed to create import statement: {str(e)}')
        has_logger_init = False
        for statement in original_node.body:
            if isinstance(statement, cst.SimpleStatementLine):
                for stmt in statement.body:
                    if isinstance(stmt, cst.Assign) and len(stmt.targets) == 1:
                        target = stmt.targets[0].target
                        if isinstance(target, cst.Name) and target.value == 'logger':
                            has_logger_init = True
        if not has_logger_init:
            imports_to_add.append(cst.SimpleStatementLine(body=[
                cst.Assign(
                    targets=[cst.AssignTarget(target=cst.Name('logger'))],
                    value=cst.Call(
                        func=cst.Name('setup_logging'),
                        args=[cst.Arg(value=cst.Name('__name__'))]
                    )
                )
            ]))
        if imports_to_add:
            inserted_imports = False
            new_body = []
            for statement in updated_node.body:
                if not inserted_imports and isinstance(statement, cst.SimpleStatementLine) and any((isinstance(stmt, (cst.Import, cst.ImportFrom)) for stmt in statement.body)):
                    new_body.append(statement)
                    new_body.extend(imports_to_add)
                    inserted_imports = True
                else:
                    new_body.append(statement)
            if not inserted_imports:
                new_body = imports_to_add + new_body
            return updated_node.with_changes(body=new_body)
        return updated_node

class StructuredLoggingTransformer(TransformationVisitor):
    """Base class for adding structured logging to code."""
    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        """Track the current function name for context."""
        if not hasattr(self, 'current_function'):
            self.current_function = []
        self.current_function.append(node.name.value)

    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
        """Clear the current function name when leaving the function."""
        if hasattr(self, 'current_function') and self.current_function and (self.current_function[-1] == original_node.name.value):
            self.current_function.pop()
        return updated_node

    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
        """Ensure the module has the necessary imports."""
        if not self.changes_made:
            return updated_node
        imports_to_add = []
        has_logging_import = False
        has_setup_logging_import = False
        has_uuid_import = False
        for statement in original_node.body:
            if isinstance(statement, cst.SimpleStatementLine):
                for stmt in statement.body:
                    if isinstance(stmt, cst.Import):
                        for name in stmt.names:
                            if name.name.value == 'logging':
                                has_logging_import = True
                            elif name.name.value == 'uuid':
                                has_uuid_import = True
                    elif isinstance(stmt, cst.ImportFrom):
                        try:
                            if stmt.module and hasattr(stmt.module, 'value'):
                                if stmt.module.value == 'utils':
                                    for name in stmt.names:
                                        if name.name.value == 'setup_logging':
                                            has_setup_logging_import = True
                                elif stmt.module.value == 'uuid':
                                    for name in stmt.names:
                                        if name.name.value == 'uuid4':
                                            has_uuid_import = True
                        except AttributeError:
                            continue
        try:
            if not has_logging_import:
                imports_to_add.append(cst.parse_module('import logging').body[0])
            if not has_setup_logging_import:
                imports_to_add.append(cst.parse_module('from utils import setup_logging').body[0])
            if not has_uuid_import:
                imports_to_add.append(cst.parse_module('from uuid import uuid4').body[0])
            has_logger_init = False
            for statement in original_node.body:
                if isinstance(statement, cst.SimpleStatementLine):
                    for stmt in statement.body:
                        if isinstance(stmt, cst.Assign) and len(stmt.targets) == 1:
                            target = stmt.targets[0].target
                            if isinstance(target, cst.Name) and target.value == 'logger':
                                has_logger_init = True
            if not has_logger_init:
                imports_to_add.append(cst.parse_module('logger = setup_logging(__name__)').body[0])
            if imports_to_add:
                inserted_imports = False
                new_body = []
                for statement in updated_node.body:
                    if not inserted_imports and isinstance(statement, cst.SimpleStatementLine) and any((isinstance(stmt, (cst.Import, cst.ImportFrom)) for stmt in statement.body)):
                        new_body.append(statement)
                        new_body.extend(imports_to_add)
                        inserted_imports = True
                    else:
                        new_body.append(statement)
                if not inserted_imports:
                    new_body = imports_to_add + new_body
                return updated_node.with_changes(body=new_body)
        except Exception as e:
            self.logger.warning(f'Error adding imports: {str(e)}')
        return updated_node

class StructuredLoggingConditionalTransformer(StructuredLoggingTransformer):
    """Adds structured logging to conditional logic."""
    def leave_If(self, original_node: cst.If, updated_node: cst.If) -> cst.If:
        """Add structured logging around conditional logic."""
        try:
            if isinstance(original_node.test, cst.Compare) and isinstance(original_node.test.left, cst.Name) and (original_node.test.left.value == '__name__'):
                return updated_node
            has_structured_logging = False
            for statement in original_node.body.body:
                if isinstance(statement, cst.SimpleStatementLine):
                    for expr in statement.body:
                        if isinstance(expr, cst.Expr) and isinstance(expr.value, cst.Call):
                            if isinstance(expr.value.func, cst.Attribute) and isinstance(expr.value.func.value, cst.Name) and (expr.value.func.value.value == 'logger') and any((kw.keyword and kw.keyword.value == 'extra' for kw in expr.value.keywords)):
                                has_structured_logging = True
            if has_structured_logging:
                return updated_node
            func_context = 'unknown_function'
            if hasattr(self, 'current_function') and self.current_function:
                func_context = self.current_function[-1]
            log_code = f"""
if {cst.module(original_node.test).code}:
    logger.debug(f"Conditional branch taken", extra={{'function': '{func_context}', 'branch': 'true', 'condition_id': str(uuid4())[:8]}})
    {cst.module(cst.IndentedBlock(body=list(original_node.body.body))).code}
else:
    logger.debug(f"Conditional branch skipped", extra={{'function': '{func_context}', 'branch': 'false', 'condition_id': str(uuid4())[:8]}})
    {cst.module(cst.IndentedBlock(body=list(original_node.orelse)) if original_node.orelse else cst.IndentedBlock(body=[])).code}
"""
            try:
                parsed_if = cst.parse_statement(log_code.strip())
                self.changes_made = True
                return parsed_if
            except Exception:
                return updated_node
        except Exception as e:
            self.logger.warning(f'Error enhancing conditional with structured logging: {str(e)}')
            return updated_node

class StructuredLoggingErrorTransformer(StructuredLoggingTransformer):
    """Adds structured logging to error handling code."""
    def leave_ExceptHandler(self, original_node: cst.ExceptHandler, updated_node: cst.ExceptHandler) -> cst.ExceptHandler:
        """Enhance except blocks with structured logging."""
        try:
            has_structured_logging = False
            for statement in original_node.body.body:
                if isinstance(statement, cst.SimpleStatementLine):
                    for expr in statement.body:
                        if isinstance(expr, cst.Expr) and isinstance(expr.value, cst.Call):
                            if isinstance(expr.value.func, cst.Attribute) and isinstance(expr.value.func.value, cst.Name) and (expr.value.func.value.value == 'logger') and any((kw.keyword and kw.keyword.value == 'extra' for kw in expr.value.keywords)):
                                has_structured_logging = True
            if has_structured_logging:
                return updated_node
            self.changes_made = True
            exception_name = original_node.name.value if original_node.name else 'e'
            func_context = 'unknown_function'
            if hasattr(self, 'current_function') and self.current_function:
                func_context = self.current_function[-1]
            log_stmt = f"""
except {(original_node.type.value if original_node.type else 'Exception')} as {exception_name}:
    logger.error(f"Error occurred: {{str({exception_name})}}", extra={{'exc_type': {exception_name}.__class__.__name__, 'function': '{func_context}', 'trace_id': str(uuid4())}})
    {cst.module(cst.IndentedBlock(body=list(original_node.body.body))).code}
"""
            try:
                parsed_except = cst.parse_statement(log_stmt.strip())
                return parsed_except
            except Exception:
                return updated_node
        except Exception as e:
            self.logger.warning(f'Error enhancing except handler with structured logging: {str(e)}')
            return updated_node

class ExceptionHandlingTransformer(TransformationVisitor):
    """Adds proper exception handling to functions."""
    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
        """Add exception handling to functions without it."""
        has_try_except = False
        for statement in original_node.body.body:
            if isinstance(statement, cst.Try):
                has_try_except = True
                break
        if has_try_except:
            return updated_node
        func_name = original_node.name.value
        try_block = cst.Try(
            body=cst.IndentedBlock(body=list(original_node.body.body)),
            handlers=[
                cst.ExceptHandler(
                    type=cst.Name('Exception'),
                    name=cst.Name('e'),
                    body=cst.IndentedBlock(body=[
                        cst.SimpleStatementLine(body=[
                            cst.Expr(value=cst.Call(
                                func=cst.Attribute(value=cst.Name('logger'), attr=cst.Name('error')),
                                args=[cst.Arg(value=cst.FormattedString(
                                    parts=[
                                        cst.FormattedStringText(value=f'Error in {func_name}: '),
                                        cst.FormattedStringExpression(expression=cst.Call(
                                            func=cst.Name('str'),
                                            args=[cst.Arg(value=cst.Name('e'))]
                                        ))
                                    ]
                                ))]
                            ))
                        ]),
                        cst.SimpleStatementLine(body=[
                            cst.Raise(
                                exc=cst.Call(
                                    func=cst.Name(f'{func_name.capitalize()}Error'),
                                    args=[cst.Arg(value=cst.FormattedString(
                                        parts=[
                                            cst.FormattedStringText(value=f'Failed to {func_name}: '),
                                            cst.FormattedStringExpression(expression=cst.Call(
                                                func=cst.Name('str'),
                                                args=[cst.Arg(value=cst.Name('e'))]
                                            ))
                                        ]
                                    ))]
                                ),
                                cause=cst.Name('e')
                            )
                        ])
                    ])
                )
            ]
        )
        self.changes_made = True
        return updated_node.with_changes(body=cst.IndentedBlock(body=[try_block]))

    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
        """Ensure the module has the necessary imports for exception handling."""
        if not self.changes_made:
            return updated_node
        return updated_node

class FunctionExtractionTransformer(TransformationVisitor):
    """Extracts code blocks into separate functions."""
    def __init__(self, transformation_data: Dict[str, Any]=None):
        """Initialize with optional transformation data."""
        super().__init__(transformation_data)
        self.extracted_functions = []
        self.current_class = None
        self.target_function = transformation_data.get('target_function') if transformation_data else None

    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        """Keep track of the current class scope."""
        self.current_class = node.name.value

    def leave_ClassDef(self, original_node: cst.ClassDef, updated_node: cst.ClassDef) -> cst.ClassDef:
        """Clear the current class scope and add extracted functions if needed."""
        class_body = list(updated_node.body.body)
        if self.extracted_functions and self.current_class:
            for func in self.extracted_functions:
                class_body.append(func)
            self.extracted_functions = []
            result = updated_node.with_changes(body=updated_node.body.with_changes(body=class_body))
            self.current_class = None
            return result
        self.current_class = None
        return updated_node

    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
        """Extract code blocks from large functions."""
        if self.target_function and original_node.name.value != self.target_function:
            return updated_node
        func_body = list(original_node.body.body)
        if len(func_body) < 5:
            return updated_node
        blocks = self._identify_blocks(func_body)
        if not blocks:
            return updated_node
        new_function_calls = []
        for block_info in blocks:
            block_statements = block_info['statements']
            block_name = block_info['name']
            block_params = block_info['params']
            block_returns = block_info['returns']
            new_function = self._create_extracted_function(block_name, block_statements, block_params, block_returns, original_node)
            if self.current_class:
                self.extracted_functions.append(new_function)
            else:
                self.extracted_functions.append(new_function)
            function_call = self._create_function_call(block_name, block_params, block_returns)
            new_function_calls.append({'start_idx': block_info['start_idx'], 'end_idx': block_info['end_idx'], 'call': function_call})
        if not new_function_calls:
            return updated_node
        new_body = []
        idx = 0
        new_function_calls.sort(key=lambda x: x['start_idx'], reverse=True)
        for i, stmt in enumerate(func_body):
            skip = False
            for call_info in new_function_calls:
                if i >= call_info['start_idx'] and i <= call_info['end_idx']:
                    if i == call_info['start_idx']:
                        new_body.append(call_info['call'])
                    skip = True
                    break
            if not skip:
                new_body.append(stmt)
        self.changes_made = True
        return updated_node.with_changes(body=updated_node.body.with_changes(body=new_body))

    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
        """Add extracted functions to the module if they weren't added to a class."""
        if not self.extracted_functions:
            return updated_node
        module_body = list(updated_node.body)
        insert_idx = 0
        for i, stmt in enumerate(module_body):
            if isinstance(stmt, cst.SimpleStatementLine) and any((isinstance(s, (cst.Import, cst.ImportFrom)) for s in stmt.body)):
                insert_idx = i + 1
            elif isinstance(stmt, (cst.FunctionDef, cst.ClassDef)):
                break
        for func in reversed(self.extracted_functions):
            module_body.insert(insert_idx, func)
        self.extracted_functions = []
        return updated_node.with_changes(body=module_body)

    def _identify_blocks(self, statements: List[cst.CSTNode]) -> List[Dict[str, Any]]:
        """Identify extractable blocks of code within a function."""
        blocks = []
        current_block = None
        current_block_name = None
        block_comment = False
        for i, stmt in enumerate(statements):
            leading_comment = None
            trailing_comment = None
            if hasattr(stmt, 'leading_lines'):
                for line in stmt.leading_lines:
                    if line.comment:
                        leading_comment = line.comment.value
            if hasattr(stmt, 'trailing_whitespace') and hasattr(stmt.trailing_whitespace, 'comment'):
                trailing_comment = stmt.trailing_whitespace.comment.value
            if leading_comment and ('###' in leading_comment or 'Step' in leading_comment or 'STEP' in leading_comment or ('start' in leading_comment.lower())):
                if current_block:
                    blocks.append(self._process_block(current_block, current_block_name, statements))
                current_block = {'start_idx': i, 'end_idx': i, 'statements': [stmt]}
                current_block_name = self._extract_name_from_comment(leading_comment)
                block_comment = True
                continue
            if leading_comment and ('###' in leading_comment or 'end' in leading_comment.lower()) and current_block:
                current_block['end_idx'] = i - 1
                blocks.append(self._process_block(current_block, current_block_name, statements))
                current_block = None
                current_block_name = None
                block_comment = False
                continue
            if current_block:
                current_block['statements'].append(stmt)
                current_block['end_idx'] = i
        if current_block:
            blocks.append(self._process_block(current_block, current_block_name, statements))
        if not blocks and len(statements) > 10:
            i = 0
            while i < len(statements) - 3:
                if self._is_similar_structure(statements[i:i + 3]):
                    block_start = i
                    block_end = i + 2
                    while block_end < len(statements) - 1 and self._is_similar_structure([statements[block_end], statements[block_end + 1]]):
                        block_end += 1
                    if block_end - block_start >= 2:
                        block = {'start_idx': block_start, 'end_idx': block_end, 'statements': statements[block_start:block_end + 1]}
                        block_name = f'process_block_{len(blocks) + 1}'
                        blocks.append(self._process_block(block, block_name, statements))
                        i = block_end + 1
                        continue
                i += 1
        return blocks

    def _is_similar_structure(self, statements: List[cst.CSTNode]) -> bool:
        """Check if a sequence of statements has similar structure."""
        if not statements:
            return False
        first_type = type(statements[0])
        return all((isinstance(stmt, first_type) for stmt in statements))

    def _extract_name_from_comment(self, comment: str) -> str:
        """Extract a function name from a comment."""
        text = comment.strip('# ')
        phrases = ['Step', 'STEP', 'Processing', 'Handle', 'Calculate', 'Validate', 'Check']
        for phrase in phrases:
            if phrase in text:
                parts = text.split(phrase, 1)
                if len(parts) > 1:
                    name_part = parts[1].strip().strip(':').strip()
                    words = ''.join((c if c.isalnum() else ' ' for c in name_part)).split()
                    if words:
                        return '_'.join((w.lower() for w in words))
        words = ''.join((c if c.isalnum() else ' ' for c in text)).split()
        if words:
            return '_'.join((w.lower() for w in words[:3 if len(words) >= 3 else len(words)]))
        return 'extracted_function'

    def _process_block(self, block: Dict[str, Any], name: str, all_statements: List[cst.CSTNode]) -> Dict[str, Any]:
        """Process a block to determine parameters and return values."""
        if not name:
            name = f'process_block_{block['start_idx']}'
        var_analyzer = VariableUsageVisitor()
        for stmt in block['statements']:
            var_analyzer.visit(stmt)
        defined_vars = var_analyzer.defined_vars
        used_vars = var_analyzer.used_vars
        potential_params = used_vars - defined_vars
        actually_defined_before = set()
        for i in range(block['start_idx']):
            pre_stmt_analyzer = VariableUsageVisitor()
            pre_stmt_analyzer.visit(all_statements[i])
            actually_defined_before.update(pre_stmt_analyzer.defined_vars)
        params = potential_params.intersection(actually_defined_before)
        potentially_used_after = set()
        for i in range(block['end_idx'] + 1, len(all_statements)):
            post_stmt_analyzer = VariableUsageVisitor()
            post_stmt_analyzer.visit(all_statements[i])
            potentially_used_after.update(post_stmt_analyzer.used_vars)
        returns = defined_vars.intersection(potentially_used_after)
        return {
            'name': name,
            'statements': block['statements'],
            'params': list(params),
            'returns': list(returns),
            'start_idx': block['start_idx'],
            'end_idx': block['end_idx']
        }

    def _create_extracted_function(self, name: str, statements: List[cst.CSTNode], params: List[str], returns: List[str], original_function: cst.FunctionDef) -> cst.FunctionDef:
        """Create a new function from extracted code block."""
        parameters = []
        for param in params:
            parameters.append(cst.Param(name=cst.Name(param), annotation=None))
        docstring_parts = [f'Extracted from {original_function.name.value}.', '']
        if params:
            docstring_parts.append('Args:')
            for param in params:
                docstring_parts.append(f'    {param}: Description')
            docstring_parts.append('')
        if returns:
            docstring_parts.append('Returns:')
            if len(returns) == 1:
                docstring_parts.append(f'    {returns[0]}: Description')
            else:
                docstring_parts.append('    Tuple containing:')
                for ret in returns:
                    docstring_parts.append(f'        - {ret}: Description')
        docstring = cst.SimpleStatementLine(body=[cst.Expr(value=cst.SimpleString(value='"""' + '\n'.join(docstring_parts) + '"""'))])
        if returns:
            if len(returns) == 1:
                return_stmt = cst.SimpleStatementLine(body=[cst.Return(value=cst.Name(returns[0]))])
            else:
                return_stmt = cst.SimpleStatementLine(body=[cst.Return(value=cst.Tuple(elements=[cst.Element(value=cst.Name(ret)) for ret in returns]))])
            if not isinstance(statements[-1], cst.Return):
                statements = list(statements) + [return_stmt]
        body = [docstring] + list(statements)
        new_func = cst.FunctionDef(
            name=cst.Name(name),
            params=cst.Parameters(params=parameters),
            body=cst.IndentedBlock(body=body),
            returns=None
        )
        return new_func

    def _create_function_call(self, name: str, params: List[str], returns: List[str]) -> cst.SimpleStatementLine:
        """Create a call to the extracted function."""
        args = [cst.Arg(value=cst.Name(param)) for param in params]
        func_call = cst.Call(func=cst.Name(name), args=args)
        if not returns:
            return cst.SimpleStatementLine(body=[cst.Expr(value=func_call)])
        elif len(returns) == 1:
            return cst.SimpleStatementLine(body=[cst.Assign(targets=[cst.AssignTarget(target=cst.Name(returns[0]))], value=func_call)])
        else:
            return cst.SimpleStatementLine(body=[cst.Assign(targets=[cst.AssignTarget(target=cst.Tuple(elements=[cst.Element(value=cst.Name(ret)) for ret in returns]))], value=func_call)])

class VariableUsageVisitor(cst.CSTVisitor):
    """CST visitor that tracks variable usage."""
    def __init__(self):
        self.defined_vars = set()
        self.used_vars = set()

    def visit_Name(self, node: cst.Name) -> None:
        """Record variable usage."""
        self.used_vars.add(node.value)

    def visit_Assign(self, node: cst.Assign) -> None:
        """Record variable definition from assignments."""
        for target in node.targets:
            if isinstance(target.target, cst.Name):
                self.defined_vars.add(target.target.value)
            elif isinstance(target.target, cst.Tuple):
                for element in target.target.elements:
                    if isinstance(element.value, cst.Name):
                        self.defined_vars.add(element.value.value)
        self.generic_visit(node)

    def visit_For(self, node: cst.For) -> None:
        """Record loop variable definition."""
        if isinstance(node.target, cst.Name):
            self.defined_vars.add(node.target.value)
        elif isinstance(node.target, cst.Tuple):
            for element in node.target.elements:
                if isinstance(element.value, cst.Name):
                    self.defined_vars.add(element.value.value)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        """Record function definition."""
        self.defined_vars.add(node.name.value)
        for param in node.params.params:
            self.defined_vars.add(param.name.value)
        self.generic_visit(node)

class FileSplitTransformer(TransformationVisitor):
    """Splits a file into multiple files."""
    def __init__(self, transformation_data: Dict[str, Any]=None):
        """Initialize with optional transformation data."""
        super().__init__(transformation_data)
        self.file_path = transformation_data.get('file_path') if transformation_data else None
        self.base_path = Path(transformation_data.get('base_path', '.')) if transformation_data else Path('.')
        self.new_files = {}
        self.modified_imports = []
        self.class_to_file = {}
        self.function_to_file = {}
        self.imports = []

    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
        """Split module into multiple files."""
        if not self.file_path:
            return updated_node
        imports = []
        classes = []
        functions = []
        constants = []
        other_statements = []
        for stmt in updated_node.body:
            if isinstance(stmt, cst.SimpleStatementLine):
                if any((isinstance(s, (cst.Import, cst.ImportFrom)) for s in stmt.body)):
                    imports.append(stmt)
                elif any((isinstance(s, cst.Assign) and isinstance(s.targets[0].target, cst.Name) and s.targets[0].target.value.isupper() for s in stmt.body)):
                    constants.append(stmt)
                else:
                    other_statements.append(stmt)
            elif isinstance(stmt, cst.ClassDef):
                classes.append(stmt)
            elif isinstance(stmt, cst.FunctionDef):
                functions.append(stmt)
            else:
                other_statements.append(stmt)
        class_groups = self._group_classes_by_inheritance(classes)
        function_groups = self._group_functions_by_usage(functions)
        merged_groups = self._merge_small_groups(class_groups, function_groups)
        original_module_path = Path(self.file_path)
        original_module_name = original_module_path.stem
        package_dir = original_module_path.parent
        init_imports = []
        for group_idx, group in enumerate(merged_groups):
            if any((isinstance(item, cst.ClassDef) for item in group)):
                for item in group:
                    if isinstance(item, cst.ClassDef):
                        group_name = f'{item.name.value.lower()}'
                        break
            else:
                func_names = [item.name.value for item in group if isinstance(item, cst.FunctionDef)]
                if func_names:
                    common_prefix = self._find_common_prefix(func_names)
                    if common_prefix:
                        group_name = common_prefix.lower()
                    else:
                        group_name = func_names[0].lower()
                else:
                    group_name = f'group_{group_idx + 1}'
            new_file_content = self._create_file_content(imports, constants if group_idx == 0 else [], group, docstring=f'Module containing functionality extracted from {original_module_name}.py')
            new_file_name = f'{group_name}.py'
            self.new_files[new_file_name] = new_file_content
            for item in group:
                if isinstance(item, cst.ClassDef):
                    self.class_to_file[item.name.value] = new_file_name
                elif isinstance(item, cst.FunctionDef):
                    self.function_to_file[item.name.value] = new_file_name
            import_items = []
            for item in group:
                if isinstance(item, (cst.ClassDef, cst.FunctionDef)):
                    import_items.append(item.name.value)
            if import_items:
                init_imports.append(f'from .{group_name} import {", ".join(import_items)}')
        init_content = f'"""\nPackage containing functionality extracted from {original_module_name}.py\n"""\n\n'
        init_content += '\n'.join(init_imports)
        self.new_files['__init__.py'] = init_content
        simplified_module = self._create_reexport_module(original_module_name, merged_groups, original_node.header)
        self.changes_made = True
        if self.transformation_data:
            self.transformation_data['new_files'] = self.new_files
        return simplified_module

    def _group_classes_by_inheritance(self, classes: List[cst.ClassDef]) -> List[List[cst.ClassDef]]:
        """Group classes based on inheritance relationships."""
        inheritance_graph = {}
        for cls in classes:
            class_name = cls.name.value
            inheritance_graph[class_name] = {'class': cls, 'bases': [], 'derived': []}
            for base in cls.bases:
                if isinstance(base.value, cst.Name):
                    base_name = base.value.value
                    inheritance_graph[class_name]['bases'].append(base_name)
        for cls_name, data in inheritance_graph.items():
            for base_name in data['bases']:
                if base_name in inheritance_graph:
                    inheritance_graph[base_name]['derived'].append(cls_name)
        visited = set()
        groups = []

        def dfs(node, current_group):
            """Depth-first search to find connected components."""
            if node in visited:
                return
            visited.add(node)
            current_group.append(inheritance_graph[node]['class'])
            for base in inheritance_graph[node]['bases']:
                if base in inheritance_graph:
                    dfs(base, current_group)
            for derived in inheritance_graph[node]['derived']:
                dfs(derived, current_group)
        for cls_name in inheritance_graph:
            if cls_name not in visited:
                current_group = []
                dfs(cls_name, current_group)
                groups.append(current_group)
        return groups

    def _group_functions_by_usage(self, functions: List[cst.FunctionDef]) -> List[List[cst.FunctionDef]]:
        """Group functions based on usage patterns."""
        function_graph = {}
        for func in functions:
            func_name = func.name.value
            function_graph[func_name] = {'function': func, 'calls': [], 'called_by': [], 'name_similarity': {}}
        for func_name, data in function_graph.items():
            func = data['function']
            class FunctionCallVisitor(cst.CSTVisitor):
                def __init__(self):
                    self.calls = []
                def visit_Call(self, node: cst.Call) -> None:
                    if isinstance(node.func, cst.Name):
                        called_func = node.func.value
                        if called_func in function_graph:
                            self.calls.append(called_func)
            visitor = FunctionCallVisitor()
            func.visit(visitor)
            for called_func in visitor.calls:
                data['calls'].append(called_func)
                function_graph[called_func]['called_by'].append(func_name)
            for other_func in function_graph:
                if other_func != func_name:
                    similarity = self._calculate_name_similarity(func_name, other_func)
                    if similarity > 0.5:
                        data['name_similarity'][other_func] = similarity
        visited = set()
        groups = []
        def dfs(node, current_group):
            """Depth-first search to find related functions."""
            if node in visited:
                return
            visited.add(node)
            current_group.append(function_graph[node]['function'])
            for called in function_graph[node]['calls']:
                dfs(called, current_group)
            for caller in function_graph[node]['called_by']:
                dfs(caller, current_group)
            for similar, score in function_graph[node]['name_similarity'].items():
                if score > 0.7:
                    dfs(similar, current_group)
        for func_name in function_graph:
            if func_name not in visited:
                current_group = []
                dfs(func_name, current_group)
                groups.append(current_group)
        return groups

    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity between two function names."""
        name1_lower = name1.lower()
        name2_lower = name2.lower()
        common_prefix_len = 0
        min_len = min(len(name1_lower), len(name2_lower))
        for i in range(min_len):
            if name1_lower[i] == name2_lower[i]:
                common_prefix_len += 1
            else:
                break
        if common_prefix_len < 3:
            return 0.0
        return common_prefix_len / max(len(name1_lower), len(name2_lower))

    def _find_common_prefix(self, names: List[str]) -> str:
        """Find a common prefix among a list of names."""
        if not names:
            return ''
        lower_names = [name.lower() for name in names]
        min_len = min((len(name) for name in lower_names))
        common_prefix = ''
        for i in range(min_len):
            char = lower_names[0][i]
            if all((name[i] == char for name in lower_names[1:])):
                common_prefix += char
            else:
                break
        if len(common_prefix) < 3:
            return ''
        return common_prefix

    def _merge_small_groups(self, class_groups: List[List[cst.ClassDef]], function_groups: List[List[cst.FunctionDef]]) -> List[List[cst.CSTNode]]:
        """Merge small groups to avoid too many files."""
        merged_groups = []
        for group in class_groups:
            if group:
                merged_groups.append(group)
        for group in function_groups:
            if group:
                merged_groups.append(group)
        MIN_GROUP_SIZE = 1
        result = []
        current_group = []
        for group in merged_groups:
            if len(group) < MIN_GROUP_SIZE:
                current_group.extend(group)
            else:
                result.append(group)
        if current_group:
            result.append(current_group)
        return result

    def _create_file_content(self, imports: List[cst.SimpleStatementLine], constants: List[cst.SimpleStatementLine], items: List[cst.CSTNode], docstring: str) -> str:
        """Create the content for a new file."""
        module = cst.Module(
            body=[
                cst.SimpleStatementLine(body=[cst.Expr(value=cst.SimpleString(value=f'"""{docstring}"""'))]),
                cst.EmptyLine(),
                *imports,
                *([] if not constants else [cst.EmptyLine()]),
                *constants,
                *([] if not items else [cst.EmptyLine()]),
                *items
            ]
        )
        return module.code

    def _create_reexport_module(self, module_name: str, groups: List[List[cst.CSTNode]], header: str) -> cst.Module:
        """Create a simplified module that re-exports all the split content."""
        import_statements = []
        for group_idx, group in enumerate(groups):
            for item in group:
                if isinstance(item, (cst.ClassDef, cst.FunctionDef)):
                    name = item.name.value
                    file_name = self.class_to_file.get(name) or self.function_to_file.get(name)
                    if file_name:
                        package_name = file_name[:-3]
                        import_statements.append(cst.SimpleStatementLine(body=[
                            cst.ImportFrom(
                                module=cst.Name(f'.{package_name}'),
                                names=[cst.ImportAlias(name=cst.Name(name))],
                                relative=[cst.Dot()]
                            )
                        ]))
        return cst.Module(
            header=header,
            body=[
                cst.SimpleStatementLine(body=[
                    cst.Expr(value=cst.SimpleString(value=f'"""\n{module_name} module - refactored into smaller modules.\n\nThis file re-exports all the functionality for backwards compatibility.\n"""'))
                ]),
                cst.EmptyLine(),
                *import_statements
            ]
        )

class StringFormatTransformer(TransformationVisitor):
    """Optimizes string concatenations by converting them to f-strings."""
    def leave_BinaryOperation(self, original_node: cst.BinaryOperation, updated_node: cst.BinaryOperation) -> cst.CSTNode:
        """Convert string concatenations to f-strings."""
        try:
            if isinstance(original_node.operator, cst.Add):
                left = original_node.left
                right = original_node.right
                # Handle string + string or string + variable
                if isinstance(left, cst.SimpleString) and isinstance(right, cst.SimpleString):
                    self.log_decision('string_concat', 'transformed', f'Converting string concatenation: {left.value} + {right.value}')
                    combined = left.value[:-1] + right.value[1:]
                    self.changes_made = True
                    return cst.SimpleString(value=combined)
                elif isinstance(left, cst.SimpleString) and isinstance(right, (cst.Name, cst.Call)):
                    self.log_decision('string_concat', 'transformed', f'Converting string + variable: {left.value} + {right}')
                    expr_code = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=right)])]).code.strip()
                    f_string = cst.FormattedString(
                        parts=[
                            cst.FormattedStringText(value=left.value[1:-1]),
                            cst.FormattedStringExpression(expression=right)
                        ]
                    )
                    self.changes_made = True
                    self.log_decision('string_concat', 'converted', f'Created f-string with expression: {expr_code}')
                    return f_string
                elif isinstance(right, cst.SimpleString) and isinstance(left, (cst.Name, cst.Call)):
                    self.log_decision('string_concat', 'transformed', f'Converting variable + string: {left} + {right.value}')
                    expr_code = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=left)])]).code.strip()
                    f_string = cst.FormattedString(
                        parts=[
                            cst.FormattedStringExpression(expression=left),
                            cst.FormattedStringText(value=right.value[1:-1])
                        ]
                    )
                    self.changes_made = True
                    self.log_decision('string_concat', 'converted', f'Created f-string with expression: {expr_code}')
                    return f_string
                elif isinstance(left, cst.SimpleString) and isinstance(right, cst.BinaryOperation):
                    # Handle nested concatenations like "a" + ("b" + var)
                    nested_result = self.leave_BinaryOperation(right, right)
                    if isinstance(nested_result, cst.FormattedString):
                        self.log_decision('string_concat', 'transformed', f'Combining string with nested f-string: {left.value}')
                        combined_parts = [
                            cst.FormattedStringText(value=left.value[1:-1])
                        ] + list(nested_result.parts)
                        self.changes_made = True
                        return cst.FormattedString(parts=combined_parts)
                    elif isinstance(nested_result, cst.SimpleString):
                        self.log_decision('string_concat', 'transformed', f'Combining strings: {left.value} + {nested_result.value}')
                        combined = left.value[:-1] + nested_result.value[1:]
                        self.changes_made = True
                        return cst.SimpleString(value=combined)
                elif isinstance(right, cst.SimpleString) and isinstance(left, cst.BinaryOperation):
                    # Handle nested concatenations like (var + "b") + "c"
                    nested_result = self.leave_BinaryOperation(left, left)
                    if isinstance(nested_result, cst.FormattedString):
                        self.log_decision('string_concat', 'transformed', f'Combining nested f-string with string: {right.value}')
                        combined_parts = list(nested_result.parts) + [
                            cst.FormattedStringText(value=right.value[1:-1])
                        ]
                        self.changes_made = True
                        return cst.FormattedString(parts=combined_parts)
                    elif isinstance(nested_result, cst.SimpleString):
                        self.log_decision('string_concat', 'transformed', f'Combining strings: {nested_result.value} + {right.value}')
                        combined = nested_result.value[:-1] + right.value[1:]
                        self.changes_made = True
                        return cst.SimpleString(value=combined)
                else:
                    self.log_decision('string_concat', 'skipped', f'Not a string concatenation: {left} + {right}')
            return updated_node
        except Exception as e:
            self.log_decision('string_concat', 'error', f'Error processing concatenation: {str(e)}')
            self.logger.warning(f'Error in StringFormatTransformer: {str(e)}')
            return updated_node

class StructuredLoggingConditionalTransformer(StructuredLoggingTransformer):
    """Adds structured logging to conditional logic."""
    def leave_If(self, original_node: cst.If, updated_node: cst.If) -> cst.If:
        """Add structured logging around conditional logic."""
        try:
            if isinstance(original_node.test, cst.Compare) and isinstance(original_node.test.left, cst.Name) and (original_node.test.left.value == '__name__'):
                self.log_decision('if_statement', 'skipped', 'Main module check, no logging needed')
                return updated_node
            has_structured_logging = False
            for statement in original_node.body.body:
                if isinstance(statement, cst.SimpleStatementLine):
                    for expr in statement.body:
                        if isinstance(expr, cst.Expr) and isinstance(expr.value, cst.Call):
                            if isinstance(expr.value.func, cst.Attribute) and isinstance(expr.value.func.value, cst.Name) and (expr.value.func.value.value == 'logger') and any((kw.keyword and kw.keyword.value == 'extra' for kw in expr.value.keywords)):
                                has_structured_logging = True
            if has_structured_logging:
                self.log_decision('if_statement', 'skipped', 'Already has structured logging')
                return updated_node
            func_context = 'unknown_function'
            if hasattr(self, 'current_function') and self.current_function:
                func_context = self.current_function[-1]
            condition_code = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=original_node.test)])]).code.strip()
            log_code = f"""
if {condition_code}:
    logger.debug("Conditional branch taken", extra={{'function': '{func_context}', 'branch': 'true', 'condition_id': str(uuid4())[:8]}})
    {cst.Module(body=[cst.IndentedBlock(body=list(original_node.body.body))]).code.strip()}
else:
    logger.debug("Conditional branch skipped", extra={{'function': '{func_context}', 'branch': 'false', 'condition_id': str(uuid4())[:8]}})
    {cst.Module(body=[cst.IndentedBlock(body=list(original_node.orelse.body)) if original_node.orelse else cst.IndentedBlock(body=[])]).code.strip()}
"""
            try:
                parsed_if = cst.parse_statement(log_code.strip())
                self.changes_made = True
                self.log_decision('if_statement', 'transformed', f'Added structured logging for condition: {condition_code}')
                return parsed_if
            except Exception as e:
                self.log_decision('if_statement', 'error', f'Failed to parse logging code: {str(e)}')
                return updated_node
        except Exception as e:
            self.log_decision('if_statement', 'error', f'Error enhancing conditional: {str(e)}')
            self.logger.warning(f'Error in StructuredLoggingConditionalTransformer: {str(e)}')
            return updated_node

class StructuredLoggingErrorTransformer(StructuredLoggingTransformer):
    """Adds structured logging to error handling code."""
    def leave_ExceptHandler(self, original_node: cst.ExceptHandler, updated_node: cst.ExceptHandler) -> cst.ExceptHandler:
        """Enhance except blocks with structured logging."""
        try:
            has_structured_logging = False
            for statement in original_node.body.body:
                if isinstance(statement, cst.SimpleStatementLine):
                    for expr in statement.body:
                        if isinstance(expr, cst.Expr) and isinstance(expr.value, cst.Call):
                            if isinstance(expr.value.func, cst.Attribute) and isinstance(expr.value.func.value, cst.Name) and (expr.value.func.value.value == 'logger') and any((kw.keyword and kw.keyword.value == 'extra' for kw in expr.value.keywords)):
                                has_structured_logging = True
            if has_structured_logging:
                self.log_decision('except_handler', 'skipped', 'Already has structured logging')
                return updated_node
            self.changes_made = True
            exception_name = original_node.name.value if original_node.name else 'e'
            func_context = 'unknown_function'
            if hasattr(self, 'current_function') and self.current_function:
                func_context = self.current_function[-1]
            log_stmt = f"""
except {(original_node.type.value if original_node.type else 'Exception')} as {exception_name}:
    logger.error(f"Error occurred: {{str({exception_name})}}", extra={{'exc_type': {exception_name}.__class__.__name__, 'function': '{func_context}', 'trace_id': str(uuid4())}})
    {cst.Module(body=[cst.IndentedBlock(body=list(original_node.body.body))]).code.strip()}
"""
            try:
                parsed_except = cst.parse_statement(log_stmt.strip())
                self.log_decision('except_handler', 'transformed', f'Added structured logging for exception: {exception_name}')
                return parsed_except
            except Exception as e:
                self.log_decision('except_handler', 'error', f'Failed to parse logging code: {str(e)}')
                return updated_node
        except Exception as e:
            self.log_decision('except_handler', 'error', f'Error enhancing except handler: {str(e)}')
            self.logger.warning(f'Error in StructuredLoggingErrorTransformer: {str(e)}')
            return updated_node

class ExceptionHandlingTransformer(TransformationVisitor):
    """Adds proper exception handling to functions."""
    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
        """Add exception handling to functions without it."""
        has_try_except = False
        for statement in original_node.body.body:
            if isinstance(statement, cst.Try):
                has_try_except = True
                break
        if has_try_except:
            self.log_decision('function_def', 'skipped', f'Function {original_node.name.value} already has exception handling')
            return updated_node
        func_name = original_node.name.value
        try_block = cst.Try(
            body=cst.IndentedBlock(body=list(original_node.body.body)),
            handlers=[
                cst.ExceptHandler(
                    type=cst.Name('Exception'),
                    name=cst.Name('e'),
                    body=cst.IndentedBlock(body=[
                        cst.SimpleStatementLine(body=[
                            cst.Expr(value=cst.Call(
                                func=cst.Attribute(value=cst.Name('logger'), attr=cst.Name('error')),
                                args=[cst.Arg(value=cst.FormattedString(
                                    parts=[
                                        cst.FormattedStringText(value=f'Error in {func_name}: '),
                                        cst.FormattedStringExpression(expression=cst.Call(
                                            func=cst.Name('str'),
                                            args=[cst.Arg(value=cst.Name('e'))]
                                        ))
                                    ]
                                ))]
                            ))
                        ]),
                        cst.SimpleStatementLine(body=[
                            cst.Raise(
                                exc=cst.Call(
                                    func=cst.Name(f'{func_name.capitalize()}Error'),
                                    args=[cst.Arg(value=cst.FormattedString(
                                        parts=[
                                            cst.FormattedStringText(value=f'Failed to {func_name}: '),
                                            cst.FormattedStringExpression(expression=cst.Call(
                                                func=cst.Name('str'),
                                                args=[cst.Arg(value=cst.Name('e'))]
                                            ))
                                        ]
                                    ))]
                                ),
                                cause=cst.Name('e')
                            )
                        ])
                    ])
                )
            ]
        )
        self.changes_made = True
        self.log_decision('function_def', 'transformed', f'Added exception handling to {func_name}')
        return updated_node.with_changes(body=cst.IndentedBlock(body=[try_block]))

    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
        """Ensure the module has the necessary imports for exception handling."""
        if not self.changes_made:
            return updated_node
        imports_to_add = []
        has_logging_import = False
        for statement in original_node.body:
            if isinstance(statement, cst.SimpleStatementLine):
                for stmt in statement.body:
                    if isinstance(stmt, cst.Import):
                        for name in stmt.names:
                            if name.name.value == 'logging':
                                has_logging_import = True
        if not has_logging_import:
            imports_to_add.append(cst.SimpleStatementLine(body=[cst.Import(names=[cst.ImportAlias(name=cst.Name('logging'))])]))
        if imports_to_add:
            new_body = imports_to_add + list(updated_node.body)
            return updated_node.with_changes(body=new_body)
        return updated_node

class FunctionExtractionTransformer(TransformationVisitor):
    """Extracts code blocks into separate functions."""
    def __init__(self, transformation_data: Dict[str, Any]=None):
        """Initialize with optional transformation data."""
        super().__init__(transformation_data)
        self.extracted_functions = []
        self.current_class = None
        self.target_function = transformation_data.get('target_function') if transformation_data else None

    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        """Keep track of the current class scope."""
        self.current_class = node.name.value

    def leave_ClassDef(self, original_node: cst.ClassDef, updated_node: cst.ClassDef) -> cst.ClassDef:
        """Clear the current class scope and add extracted functions if needed."""
        class_body = list(updated_node.body.body)
        if self.extracted_functions and self.current_class:
            for func in self.extracted_functions:
                class_body.append(func)
            self.extracted_functions = []
            self.log_decision('class_def', 'transformed', f'Added {len(self.extracted_functions)} extracted functions to class {self.current_class}')
            result = updated_node.with_changes(body=updated_node.body.with_changes(body=class_body))
            self.current_class = None
            return result
        self.current_class = None
        return updated_node

    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
        """Extract code blocks from large functions."""
        if self.target_function and original_node.name.value != self.target_function:
            self.log_decision('function_def', 'skipped', f'Function {original_node.name.value} is not target {self.target_function}')
            return updated_node
        func_body = list(original_node.body.body)
        if len(func_body) < 5:
            self.log_decision('function_def', 'skipped', f'Function {original_node.name.value} too small ({len(func_body)} statements)')
            return updated_node
        blocks = self._identify_blocks(func_body)
        if not blocks:
            self.log_decision('function_def', 'skipped', f'No extractable blocks in {original_node.name.value}')
            return updated_node
        new_function_calls = []
        for block_info in blocks:
            block_statements = block_info['statements']
            block_name = block_info['name']
            block_params = block_info['params']
            block_returns = block_info['returns']
            new_function = self._create_extracted_function(block_name, block_statements, block_params, block_returns, original_node)
            if self.current_class:
                self.extracted_functions.append(new_function)
            else:
                self.extracted_functions.append(new_function)
            function_call = self._create_function_call(block_name, block_params, block_returns)
            new_function_calls.append({'start_idx': block_info['start_idx'], 'end_idx': block_info['end_idx'], 'call': function_call})
        if not new_function_calls:
            self.log_decision('function_def', 'skipped', f'No function calls created for {original_node.name.value}')
            return updated_node
        new_body = []
        idx = 0
        new_function_calls.sort(key=lambda x: x['start_idx'], reverse=True)
        for i, stmt in enumerate(func_body):
            skip = False
            for call_info in new_function_calls:
                if i >= call_info['start_idx'] and i <= call_info['end_idx']:
                    if i == call_info['start_idx']:
                        new_body.append(call_info['call'])
                    skip = True
                    break
            if not skip:
                new_body.append(stmt)
        self.changes_made = True
        self.log_decision('function_def', 'transformed', f'Extracted {len(new_function_calls)} blocks from {original_node.name.value}')
        return updated_node.with_changes(body=updated_node.body.with_changes(body=new_body))

    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
        """Add extracted functions to the module if they weren't added to a class."""
        if not self.extracted_functions:
            return updated_node
        module_body = list(updated_node.body)
        insert_idx = 0
        for i, stmt in enumerate(module_body):
            if isinstance(stmt, cst.SimpleStatementLine) and any((isinstance(s, (cst.Import, cst.ImportFrom)) for s in stmt.body)):
                insert_idx = i + 1
            elif isinstance(stmt, (cst.FunctionDef, cst.ClassDef)):
                break
        for func in reversed(self.extracted_functions):
            module_body.insert(insert_idx, func)
        self.extracted_functions = []
        self.log_decision('module', 'transformed', f'Added {len(self.extracted_functions)} extracted functions to module')
        return updated_node.with_changes(body=module_body)

    def _identify_blocks(self, statements: List[cst.CSTNode]) -> List[Dict[str, Any]]:
        """Identify extractable blocks of code within a function."""
        blocks = []
        current_block = None
        current_block_name = None
        block_comment = False
        for i, stmt in enumerate(statements):
            leading_comment = None
            trailing_comment = None
            if hasattr(stmt, 'leading_lines'):
                for line in stmt.leading_lines:
                    if line.comment:
                        leading_comment = line.comment.value
            if hasattr(stmt, 'trailing_whitespace') and hasattr(stmt.trailing_whitespace, 'comment'):
                trailing_comment = stmt.trailing_whitespace.comment.value
            if leading_comment and ('###' in leading_comment or 'Step' in leading_comment or 'STEP' in leading_comment or ('start' in leading_comment.lower())):
                if current_block:
                    blocks.append(self._process_block(current_block, current_block_name, statements))
                current_block = {'start_idx': i, 'end_idx': i, 'statements': [stmt]}
                current_block_name = self._extract_name_from_comment(leading_comment)
                block_comment = True
                continue
            if leading_comment and ('###' in leading_comment or 'end' in leading_comment.lower()) and current_block:
                current_block['end_idx'] = i - 1
                blocks.append(self._process_block(current_block, current_block_name, statements))
                current_block = None
                current_block_name = None
                block_comment = False
                continue
            if current_block:
                current_block['statements'].append(stmt)
                current_block['end_idx'] = i
        if current_block:
            blocks.append(self._process_block(current_block, current_block_name, statements))
        if not blocks and len(statements) > 10:
            i = 0
            while i < len(statements) - 3:
                if self._is_similar_structure(statements[i:i + 3]):
                    block_start = i
                    block_end = i + 2
                    while block_end < len(statements) - 1 and self._is_similar_structure([statements[block_end], statements[block_end + 1]]):
                        block_end += 1
                    if block_end - block_start >= 2:
                        block = {'start_idx': block_start, 'end_idx': block_end, 'statements': statements[block_start:block_end + 1]}
                        block_name = f'process_block_{len(blocks) + 1}'
                        blocks.append(self._process_block(block, block_name, statements))
                        i = block_end + 1
                        continue
                i += 1
        return blocks

    def _is_similar_structure(self, statements: List[cst.CSTNode]) -> bool:
        """Check if a sequence of statements has similar structure."""
        if not statements:
            return False
        first_type = type(statements[0])
        return all((isinstance(stmt, first_type) for stmt in statements))

    def _extract_name_from_comment(self, comment: str) -> str:
        """Extract a function name from a comment."""
        text = comment.strip('# ')
        phrases = ['Step', 'STEP', 'Processing', 'Handle', 'Calculate', 'Validate', 'Check']
        for phrase in phrases:
            if phrase in text:
                parts = text.split(phrase, 1)
                if len(parts) > 1:
                    name_part = parts[1].strip().strip(':').strip()
                    words = ''.join((c if c.isalnum() else ' ' for c in name_part)).split()
                    if words:
                        return '_'.join((w.lower() for w in words))
        words = ''.join((c if c.isalnum() else ' ' for c in text)).split()
        if words:
            return '_'.join((w.lower() for w in words[:3 if len(words) >= 3 else len(words)]))
        return 'extracted_function'

    def _process_block(self, block: Dict[str, Any], name: str, all_statements: List[cst.CSTNode]) -> Dict[str, Any]:
        """Process a block to determine parameters and return values."""
        if not name:
            name = f'process_block_{block['start_idx']}'
        var_analyzer = VariableUsageVisitor()
        for stmt in block['statements']:
            var_analyzer.visit(stmt)
        defined_vars = var_analyzer.defined_vars
        used_vars = var_analyzer.used_vars
        potential_params = used_vars - defined_vars
        actually_defined_before = set()
        for i in range(block['start_idx']):
            pre_stmt_analyzer = VariableUsageVisitor()
            pre_stmt_analyzer.visit(all_statements[i])
            actually_defined_before.update(pre_stmt_analyzer.defined_vars)
        params = potential_params.intersection(actually_defined_before)
        potentially_used_after = set()
        for i in range(block['end_idx'] + 1, len(all_statements)):
            post_stmt_analyzer = VariableUsageVisitor()
            post_stmt_analyzer.visit(all_statements[i])
            potentially_used_after.update(post_stmt_analyzer.used_vars)
        returns = defined_vars.intersection(potentially_used_after)
        return {
            'name': name,
            'statements': block['statements'],
            'params': list(params),
            'returns': list(returns),
            'start_idx': block['start_idx'],
            'end_idx': block['end_idx']
        }

    def _create_extracted_function(self, name: str, statements: List[cst.CSTNode], params: List[str], returns: List[str], original_function: cst.FunctionDef) -> cst.FunctionDef:
        """Create a new function from extracted code block."""
        parameters = []
        for param in params:
            parameters.append(cst.Param(name=cst.Name(param), annotation=None))
        docstring_parts = [f'Extracted from {original_function.name.value}.', '']
        if params:
            docstring_parts.append('Args:')
            for param in params:
                docstring_parts.append(f'    {param}: Description')
            docstring_parts.append('')
        if returns:
            docstring_parts.append('Returns:')
            if len(returns) == 1:
                docstring_parts.append(f'    {returns[0]}: Description')
            else:
                docstring_parts.append('    Tuple containing:')
                for ret in returns:
                    docstring_parts.append(f'        - {ret}: Description')
        docstring = cst.SimpleStatementLine(body=[cst.Expr(value=cst.SimpleString(value='"""' + '\n'.join(docstring_parts) + '"""'))])
        if returns:
            if len(returns) == 1:
                return_stmt = cst.SimpleStatementLine(body=[cst.Return(value=cst.Name(returns[0]))])
            else:
                return_stmt = cst.SimpleStatementLine(body=[cst.Return(value=cst.Tuple(elements=[cst.Element(value=cst.Name(ret)) for ret in returns]))])
            if not isinstance(statements[-1], cst.SimpleStatementLine) or not isinstance(statements[-1].body[0], cst.Return):
                statements = list(statements) + [return_stmt]
        body = [docstring] + list(statements)
        new_func = cst.FunctionDef(
            name=cst.Name(name),
            params=cst.Parameters(params=parameters),
            body=cst.IndentedBlock(body=body),
            returns=None
        )
        self.log_decision('function_extraction', 'created', f'Created function {name} with {len(params)} params and {len(returns)} returns')
        return new_func

    def _create_function_call(self, name: str, params: List[str], returns: List[str]) -> cst.SimpleStatementLine:
        """Create a call to the extracted function."""
        args = [cst.Arg(value=cst.Name(param)) for param in params]
        func_call = cst.Call(func=cst.Name(name), args=args)
        if not returns:
            return cst.SimpleStatementLine(body=[cst.Expr(value=func_call)])
        elif len(returns) == 1:
            return cst.SimpleStatementLine(body=[cst.Assign(targets=[cst.AssignTarget(target=cst.Name(returns[0]))], value=func_call)])
        else:
            return cst.SimpleStatementLine(body=[cst.Assign(targets=[cst.AssignTarget(target=cst.Tuple(elements=[cst.Element(value=cst.Name(ret)) for ret in returns]))], value=func_call)])

class VariableUsageVisitor(cst.CSTVisitor):
    """CST visitor that tracks variable usage."""
    def __init__(self):
        self.defined_vars = set()
        self.used_vars = set()

    def visit_Name(self, node: cst.Name) -> None:
        """Record variable usage."""
        self.used_vars.add(node.value)

    def visit_Assign(self, node: cst.Assign) -> None:
        """Record variable definition from assignments."""
        for target in node.targets:
            if isinstance(target.target, cst.Name):
                self.defined_vars.add(target.target.value)
            elif isinstance(target.target, cst.Tuple):
                for element in target.target.elements:
                    if isinstance(element.value, cst.Name):
                        self.defined_vars.add(element.value.value)
        self.generic_visit(node)

    def visit_For(self, node: cst.For) -> None:
        """Record loop variable definition."""
        if isinstance(node.target, cst.Name):
            self.defined_vars.add(node.target.value)
        elif isinstance(node.target, cst.Tuple):
            for element in node.target.elements:
                if isinstance(element.value, cst.Name):
                    self.defined_vars.add(element.value.value)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        """Record function definition."""
        self.defined_vars.add(node.name.value)
        for param in node.params.params:
            self.defined_vars.add(param.name.value)
        self.generic_visit(node)

class FileSplitTransformer(TransformationVisitor):
    """Splits a file into multiple files."""
    def __init__(self, transformation_data: Dict[str, Any]=None):
        """Initialize with optional transformation data."""
        super().__init__(transformation_data)
        self.file_path = transformation_data.get('file_path') if transformation_data else None
        self.base_path = Path(transformation_data.get('base_path', '.')) if transformation_data else Path('.')
        self.new_files = {}
        self.modified_imports = []
        self.class_to_file = {}
        self.function_to_file = {}
        self.imports = []

    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
        """Split module into multiple files."""
        if not self.file_path:
            self.log_decision('module', 'skipped', 'No file path provided')
            return updated_node
        imports = []
        classes = []
        functions = []
        constants = []
        other_statements = []
        for stmt in updated_node.body:
            if isinstance(stmt, cst.SimpleStatementLine):
                if any((isinstance(s, (cst.Import, cst.ImportFrom)) for s in stmt.body)):
                    imports.append(stmt)
                elif any((isinstance(s, cst.Assign) and isinstance(s.targets[0].target, cst.Name) and s.targets[0].target.value.isupper() for s in stmt.body)):
                    constants.append(stmt)
                else:
                    other_statements.append(stmt)
            elif isinstance(stmt, cst.ClassDef):
                classes.append(stmt)
            elif isinstance(stmt, cst.FunctionDef):
                functions.append(stmt)
            else:
                other_statements.append(stmt)
        class_groups = self._group_classes_by_inheritance(classes)
        function_groups = self._group_functions_by_usage(functions)
        merged_groups = self._merge_small_groups(class_groups, function_groups)
        original_module_path = Path(self.file_path)
        original_module_name = original_module_path.stem
        package_dir = original_module_path.parent
        init_imports = []
        for group_idx, group in enumerate(merged_groups):
            if any((isinstance(item, cst.ClassDef) for item in group)):
                for item in group:
                    if isinstance(item, cst.ClassDef):
                        group_name = f'{item.name.value.lower()}'
                        break
            else:
                func_names = [item.name.value for item in group if isinstance(item, cst.FunctionDef)]
                if func_names:
                    common_prefix = self._find_common_prefix(func_names)
                    if common_prefix:
                        group_name = common_prefix.lower()
                    else:
                        group_name = func_names[0].lower()
                else:
                    group_name = f'group_{group_idx + 1}'
            new_file_content = self._create_file_content(imports, constants if group_idx == 0 else [], group, docstring=f'Module containing functionality extracted from {original_module_name}.py')
            new_file_name = f'{group_name}.py'
            self.new_files[new_file_name] = new_file_content
            for item in group:
                if isinstance(item, cst.ClassDef):
                    self.class_to_file[item.name.value] = new_file_name
                elif isinstance(item, cst.FunctionDef):
                    self.function_to_file[item.name.value] = new_file_name
            import_items = []
            for item in group:
                if isinstance(item, (cst.ClassDef, cst.FunctionDef)):
                    import_items.append(item.name.value)
            if import_items:
                init_imports.append(f'from .{group_name} import {", ".join(import_items)}')
        init_content = f'"""\nPackage containing functionality extracted from {original_module_name}.py\n"""\n\n'
        init_content += '\n'.join(init_imports)
        self.new_files['__init__.py'] = init_content
        simplified_module = self._create_reexport_module(original_module_name, merged_groups, original_node.header)
        self.changes_made = True
        if self.transformation_data:
            self.transformation_data['new_files'] = self.new_files
        self.log_decision('module', 'transformed', f'Split module into {len(self.new_files)} files')
        return simplified_module

    def _group_classes_by_inheritance(self, classes: List[cst.ClassDef]) -> List[List[cst.ClassDef]]:
        """Group classes based on inheritance relationships."""
        inheritance_graph = {}
        for cls in classes:
            class_name = cls.name.value
            inheritance_graph[class_name] = {'class': cls, 'bases': [], 'derived': []}
            for base in cls.bases:
                if isinstance(base.value, cst.Name):
                    base_name = base.value.value
                    inheritance_graph[class_name]['bases'].append(base_name)
        for cls_name, data in inheritance_graph.items():
            for base_name in data['bases']:
                if base_name in inheritance_graph:
                    inheritance_graph[base_name]['derived'].append(cls_name)
        visited = set()
        groups = []
        def dfs(node, current_group):
            """Depth-first search to find connected components."""
            if node in visited:
                return
            visited.add(node)
            current_group.append(inheritance_graph[node]['class'])
            for base in inheritance_graph[node]['bases']:
                if base in inheritance_graph:
                    dfs(base, current_group)
            for derived in inheritance_graph[node]['derived']:
                dfs(derived, current_group)
        for cls_name in inheritance_graph:
            if cls_name not in visited:
                current_group = []
                dfs(cls_name, current_group)
                groups.append(current_group)
        self.log_decision('class_grouping', 'grouped', f'Grouped {len(classes)} classes into {len(groups)} groups')
        return groups

    def _group_functions_by_usage(self, functions: List[cst.FunctionDef]) -> List[List[cst.FunctionDef]]:
        """Group functions based on usage patterns."""
        function_graph = {}
        for func in functions:
            func_name = func.name.value
            function_graph[func_name] = {'function': func, 'calls': [], 'called_by': [], 'name_similarity': {}}
        for func_name, data in function_graph.items():
            func = data['function']
            class FunctionCallVisitor(cst.CSTVisitor):
                def __init__(self):
                    self.calls = []
                def visit_Call(self, node: cst.Call) -> None:
                    if isinstance(node.func, cst.Name):
                        called_func = node.func.value
                        if called_func in function_graph:
                            self.calls.append(called_func)
            visitor = FunctionCallVisitor()
            func.visit(visitor)
            for called_func in visitor.calls:
                data['calls'].append(called_func)
                function_graph[called_func]['called_by'].append(func_name)
            for other_func in function_graph:
                if other_func != func_name:
                    similarity = self._calculate_name_similarity(func_name, other_func)
                    if similarity > 0.5:
                        data['name_similarity'][other_func] = similarity
        visited = set()
        groups = []
        def dfs(node, current_group):
            """Depth-first search to find related functions."""
            if node in visited:
                return
            visited.add(node)
            current_group.append(function_graph[node]['function'])
            for called in function_graph[node]['calls']:
                dfs(called, current_group)
            for caller in function_graph[node]['called_by']:
                dfs(caller, current_group)
            for similar, score in function_graph[node]['name_similarity'].items():
                if score > 0.7:
                    dfs(similar, current_group)
        for func_name in function_graph:
            if func_name not in visited:
                current_group = []
                dfs(func_name, current_group)
                groups.append(current_group)
        self.log_decision('function_grouping', 'grouped', f'Grouped {len(functions)} functions into {len(groups)} groups')
        return groups

    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity between two function names."""
        name1_lower = name1.lower()
        name2_lower = name2.lower()
        common_prefix_len = 0
        min_len = min(len(name1_lower), len(name2_lower))
        for i in range(min_len):
            if name1_lower[i] == name2_lower[i]:
                common_prefix_len += 1
            else:
                break
        if common_prefix_len < 3:
            return 0.0
        return common_prefix_len / max(len(name1_lower), len(name2_lower))

    def _find_common_prefix(self, names: List[str]) -> str:
        """Find a common prefix among a list of names."""
        if not names:
            return ''
        lower_names = [name.lower() for name in names]
        min_len = min((len(name) for name in lower_names))
        common_prefix = ''
        for i in range(min_len):
            char = lower_names[0][i]
            if all((name[i] == char for name in lower_names[1:])):
                common_prefix += char
            else:
                break
        if len(common_prefix) < 3:
            return ''
        return common_prefix

    def _merge_small_groups(self, class_groups: List[List[cst.ClassDef]], function_groups: List[List[cst.FunctionDef]]) -> List[List[cst.CSTNode]]:
        """Merge small groups to avoid too many files."""
        merged_groups = []
        for group in class_groups:
            if group:
                merged_groups.append(group)
        for group in function_groups:
            if group:
                merged_groups.append(group)
        MIN_GROUP_SIZE = 1
        result = []
        current_group = []
        for group in merged_groups:
            if len(group) < MIN_GROUP_SIZE:
                current_group.extend(group)
            else:
                result.append(group)
        if current_group:
            result.append(current_group)
        self.log_decision('group_merging', 'merged', f'Merged into {len(result)} groups')
        return result

    def _create_file_content(self, imports: List[cst.SimpleStatementLine], constants: List[cst.SimpleStatementLine], items: List[cst.CSTNode], docstring: str) -> str:
        """Create the content for a new file."""
        module = cst.Module(
            body=[
                cst.SimpleStatementLine(body=[cst.Expr(value=cst.SimpleString(value=f'"""{docstring}"""'))]),
                cst.EmptyLine(),
                *imports,
                *([] if not constants else [cst.EmptyLine()]),
                *constants,
                *([] if not items else [cst.EmptyLine()]),
                *items
            ]
        )
        return module.code

    def _create_reexport_module(self, module_name: str, groups: List[List[cst.CSTNode]], header: str) -> cst.Module:
        """Create a simplified module that re-exports all the split content."""
        import_statements = []
        for group_idx, group in enumerate(groups):
            for item in group:
                if isinstance(item, (cst.ClassDef, cst.FunctionDef)):
                    name = item.name.value
                    file_name = self.class_to_file.get(name) or self.function_to_file.get(name)
                    if file_name:
                        package_name = file_name[:-3]
                        import_statements.append(cst.SimpleStatementLine(body=[
                            cst.ImportFrom(
                                module=cst.Name(f'.{package_name}'),
                                names=[cst.ImportAlias(name=cst.Name(name))],
                                relative=[cst.Dot()]
                            )
                        ]))
        return cst.Module(
            header=header,
            body=[
                cst.SimpleStatementLine(body=[
                    cst.Expr(value=cst.SimpleString(value=f'"""\n{module_name} module - refactored into smaller modules.\n\nThis file re-exports all the functionality for backwards compatibility.\n"""'))
                ]),
                cst.EmptyLine(),
                *import_statements
            ]
        )