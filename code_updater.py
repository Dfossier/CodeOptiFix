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
    """Enum for tracking the status of code updates."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    NO_CHANGES = "no_changes"

class CodeTransformation:
    """Represents a single code transformation with metadata."""
    def __init__(self, file_path: Path, transformer: Any, goal: ImprovementGoal, transformation_data: Dict[str, Any]=None):
        self.file_path = file_path
        self.transformer = transformer
        self.goal = goal
        self.transformation_data = transformation_data or {}
        self.status = UpdateStatus.NO_CHANGES
        self.message = ""
        self.changes: Dict[str, Any] = {}

    def apply(self) -> Tuple[UpdateStatus, str]:
        """Apply the transformation to the code."""
        try:
            module = cst.parse_module(self.file_path.read_text(encoding='utf-8'))
            wrapper = MetadataWrapper(module)
            transformed_module = wrapper.visit(self.transformer)
            if self.transformer.changes_made:
                self.file_path.write_text(transformed_module.code, encoding='utf-8')
                self.status = UpdateStatus.SUCCESS
                self.message = "Transformation applied successfully"
                self.changes = {'decision_log': self.transformer.decision_log}
            else:
                self.status = UpdateStatus.NO_CHANGES
                self.message = "No changes were necessary"
            return self.status, self.message
        except Exception as e:
            self.status = UpdateStatus.FAILED
            self.message = f"Transformation failed: {str(e)}"
            logger.error(f"Error applying transformation to {self.file_path}: {str(e)}")
            return self.status, self.message

    def to_dict(self) -> Dict[str, Any]:
        """Convert the transformation to a dictionary for serialization."""
        return {
            'file_path': str(self.file_path),
            'goal': self.goal.to_dict(),
            'status': self.status.value,
            'message': self.message,
            'changes': self.changes
        }

class DependencyGraph:
    """Tracks dependencies between Python modules."""
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.graph = nx.DiGraph()
        self.logger = logger

    def build_from_files(self, files: List[Path]) -> None:
        """Build the dependency graph from a list of Python files."""
        self.graph.clear()
        for file_path in files:
            try:
                module = cst.parse_module(file_path.read_text(encoding='utf-8'))
                visitor = DependencyVisitor(file_path)
                module.visit(visitor)
                for dep in visitor.dependencies:
                    self.graph.add_edge(str(file_path.relative_to(self.base_path)), str(dep.relative_to(self.base_path)))
            except Exception as e:
                self.logger.warning(f"Error parsing {file_path} for dependencies: {str(e)}")

    def get_affected_files(self, file_path: Path) -> Set[Path]:
        """Get all files affected by changes to the given file."""
        try:
            file_key = str(file_path.relative_to(self.base_path))
            affected = set(nx.descendants(self.graph, file_key) or [])
            affected.add(file_key)
            return {self.base_path / Path(f) for f in affected}
        except nx.NetworkXError:
            return {file_path}

class DependencyVisitor(cst.CSTVisitor):
    """Visitor to collect dependencies from a Python module."""
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.dependencies: Set[Path] = set()
        self.base_path = file_path.parent
        self.logger = logger  # Fixed logger initialization

    def visit_Import(self, node: cst.Import) -> None:
        """Collect dependencies from import statements."""
        for name in node.names:
            if isinstance(name, cst.ImportAlias):
                module = name.name.value
                self._add_dependency(module)

    def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
        """Collect dependencies from from-import statements."""
        if isinstance(node.module, cst.Name):
            module = node.module.value
            self._add_dependency(module)

    def _add_dependency(self, module: str) -> None:
        """Resolve module to file path and add to dependencies."""
        try:
            module_path = self.base_path / f"{module}.py"
            if module_path.exists():
                self.dependencies.add(module_path)
        except Exception as e:
            self.logger.debug(f"Error resolving dependency {module}: {str(e)}")

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
            'optimize_string_formatting': self._create_string_format_transformer,
            'add_dict_get_default': self._create_dict_get_default_transformer,
            'enhance_subprocess_security': self._create_subprocess_security_transformer,
            'document_return_values': self._create_doc_return_values_transformer
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
            python_files = [
                f for f in self.base_path.glob('**/*.py')
                if not str(f).startswith(str(self.base_path / '.venv'))
            ]
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
        """Generate a plan of transformations for a given goal."""
        transformations = []
        try:
            factory = self.transformation_factories.get(goal.improvement_type)
            if not factory:
                self.logger.warning(f"No transformation factory for goal type: {goal.improvement_type}")
                return transformations
            transformer_class, target_files = factory(goal)
            if not target_files:
                target_files = [self.base_path / goal.target_module]
            for file_path in target_files:
                if file_path.is_file() and file_path.suffix == '.py':
                    transformer = transformer_class(goal.transformation_data)
                    transformation = CodeTransformation(file_path, transformer, goal, goal.transformation_data)
                    transformations.append(transformation)
                    self.logger.info(f"Planned transformation: {transformer.__class__.__name__} for {file_path}")
            return transformations
        except Exception as e:
            self.logger.error(f"Error generating plan for goal '{goal.description}': {str(e)}")
            return []

    def _apply_transformations(self, transformations: List[CodeTransformation]) -> Tuple[str, str, List[CodeTransformation]]:
        """Apply a list of transformations and validate results."""
        applied = []
        try:
            with tempfile.TemporaryDirectory(dir=self.sandbox_dir) as temp_dir:
                temp_path = Path(temp_dir)
                self._copy_codebase(temp_path)
                for transformation in transformations:
                    temp_file = temp_path / transformation.file_path.relative_to(self.base_path)
                    transformation.file_path = temp_file
                    status, message = transformation.apply()
                    self.logger.info(f"Applied transformation to {temp_file}: {status} - {message}")
                    applied.append(transformation)
                    if status == UpdateStatus.FAILED:
                        return 'failed', f"Transformation failed for {temp_file}: {message}", applied
                if not self._validate_changes(temp_path):
                    return 'failed', "Validation failed for transformed code", applied
                self._commit_changes(temp_path)
                status = 'success' if any(t.status == UpdateStatus.SUCCESS for t in applied) else 'no_changes'
                message = "All transformations applied and validated successfully" if status == 'success' else "No changes were necessary"
                return status, message, applied
        except Exception as e:
            self.logger.error(f"Error applying transformations: {str(e)}")
            return 'failed', f"Error applying transformations: {str(e)}", applied

    def _copy_codebase(self, target_path: Path) -> None:
        """Copy the codebase to a temporary directory for safe transformation."""
        try:
            for file_path in self.base_path.glob('**/*.py'):
                if str(file_path).startswith(str(self.base_path / '.venv')):
                    continue
                relative_path = file_path.relative_to(self.base_path)
                target_file = target_path / relative_path
                target_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file_path, target_file)
            self.logger.info(f"Copied codebase to {target_path}")
        except Exception as e:
            self.logger.error(f"Error copying codebase to {target_path}: {str(e)}")
            raise CodeUpdateError(f"Failed to copy codebase: {str(e)}")

    def _validate_changes(self, temp_path: Path) -> bool:
        """Validate the transformed code using static checkers and tests."""
        try:
            for checker in self.static_checkers:
                cmd = [checker, str(temp_path)]
                result = subprocess.run(cmd, capture_output=True, text=True, check=False)
                if result.returncode != 0:
                    self.logger.error(f"Static checker {checker} failed: {result.stderr}")
                    return False
            if self.test_command:
                cmd = self.test_command.split()
                result = subprocess.run(cmd, cwd=temp_path, capture_output=True, text=True, check=False)
                if result.returncode != 0:
                    self.logger.error(f"Tests failed: {result.stderr}")
                    return False
            self.logger.info("Validation successful")
            return True
        except Exception as e:
            self.logger.error(f"Error during validation: {str(e)}")
            return False

    def _commit_changes(self, temp_path: Path) -> None:
        """Commit validated changes back to the original codebase."""
        try:
            for temp_file in temp_path.glob('**/*.py'):
                relative_path = temp_file.relative_to(temp_path)
                original_file = self.base_path / relative_path
                shutil.copy2(temp_file, original_file)
            self.logger.info("Changes committed to original codebase")
        except Exception as e:
            self.logger.error(f"Error committing changes: {str(e)}")
            raise CodeUpdateError(f"Failed to commit changes: {str(e)}")

    def _create_print_to_logging_transformer(self, goal: ImprovementGoal) -> Tuple[Any, List[Path]]:
        """Create a transformer for replacing print statements with logging."""
        from transforms.print_to_logging import PrintToLoggingTransformer
        target_files = [self.base_path / f for f in goal.target_files] if goal.target_files else []
        return PrintToLoggingTransformer, target_files

    def _create_exception_handling_transformer(self, goal: ImprovementGoal) -> Tuple[Any, List[Path]]:
        """Create a transformer for adding exception handling."""
        from transforms.exception_handling import ExceptionHandlingTransformer
        target_files = [self.base_path / f for f in goal.target_files] if goal.target_files else []
        return ExceptionHandlingTransformer, target_files

    def _create_function_extraction_transformer(self, goal: ImprovementGoal) -> Tuple[Any, List[Path]]:
        """Create a transformer for extracting functions."""
        from transforms.function_extraction import FunctionExtractionTransformer
        target_files = [self.base_path / f for f in goal.target_files] if goal.target_files else []
        return FunctionExtractionTransformer, target_files

    def _create_file_split_transformer(self, goal: ImprovementGoal) -> Tuple[Any, List[Path]]:
        """Create a transformer for splitting large files."""
        from transforms.file_split import FileSplitTransformer
        target_files = [self.base_path / f for f in goal.target_files] if goal.target_files else []
        return FileSplitTransformer, target_files

    def _create_structured_logging_transformer(self, goal: ImprovementGoal) -> Tuple[Any, List[Path]]:
        """Create a transformer for adding structured logging."""
        from structured_logging import StructuredLoggingTransformer, StructuredLoggingConditionalTransformer, StructuredLoggingErrorTransformer
        target_files = [self.base_path / f for f in goal.target_files] if goal.target_files else []
        if goal.improvement_type == 'add_structured_logging_conditional':
            return StructuredLoggingConditionalTransformer, target_files
        elif goal.improvement_type == 'add_structured_logging_error':
            return StructuredLoggingErrorTransformer, target_files
        return StructuredLoggingTransformer, target_files

    def _create_string_format_transformer(self, goal: ImprovementGoal) -> Tuple[Any, List[Path]]:
        """Create a transformer for optimizing string formatting."""
        from transforms.string_formatting import StringFormatTransformer
        target_files = [self.base_path / f for f in goal.target_files] if goal.target_files else []
        return StringFormatTransformer, target_files

    def _create_dict_get_default_transformer(self, goal: ImprovementGoal) -> Tuple[Any, List[Path]]:
        """Create a transformer for adding default values to dict.get calls."""
        from transforms.dict_get_default import DictGetDefaultTransformer
        target_files = [self.base_path / f for f in goal.target_files] if goal.target_files else []
        return DictGetDefaultTransformer, target_files

    def _create_subprocess_security_transformer(self, goal: ImprovementGoal) -> Tuple[Any, List[Path]]:
        """Create a transformer for enhancing subprocess call security."""
        from transforms.subprocess_security import SubprocessSecurityTransformer
        target_files = [self.base_path / f for f in goal.target_files] if goal.target_files else []
        return SubprocessSecurityTransformer, target_files

    def _create_doc_return_values_transformer(self, goal: ImprovementGoal) -> Tuple[Any, List[Path]]:
        """Create a transformer for documenting return values in docstrings."""
        from transforms.doc_return_values import DocReturnValuesTransformer
        target_files = [self.base_path / f for f in goal.target_files] if goal.target_files else []
        return DocReturnValuesTransformer, target_files

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
                extra_dict = [
                    cst.DictElement(
                        key=cst.SimpleString("'function'"),
                        value=cst.SimpleString(f"'{func_context}'")
                    )
                ] if func_context else []
                return cst.Call(
                    func=cst.Attribute(value=cst.Name('logger'), attr=cst.Name(log_level)),
                    args=original_node.args,
                    keywords=[cst.Arg(
                        keyword=cst.Name('extra'),
                        value=cst.Dict(extra_dict)
                    )] if extra_dict else []
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
    """Adds structured logging to code with context."""
    def leave_Call(self, original_node: cst.Call, updated_node: cst.Call) -> cst.Call:
        """Enhance logging calls with structured context."""
        try:
            if isinstance(original_node.func, cst.Attribute) and isinstance(original_node.func.value, cst.Name) and original_node.func.value.value == 'logger':
                self.log_decision('logger', 'analyzing', 'found logger call')
                if not any(kw.keyword and kw.keyword.value == 'extra' for kw in original_node.keywords):
                    self.changes_made = True
                    extra_dict = [
                        cst.DictElement(
                            key=cst.SimpleString("'trace_id'"),
                            value=cst.Call(
                                func=cst.Name('str'),
                                args=[cst.Arg(cst.Call(func=cst.Name('uuid4')))]
                            )
                        )
                    ]
                    return updated_node.with_changes(
                        keywords=list(updated_node.keywords) + [
                            cst.Arg(
                                keyword=cst.Name('extra'),
                                value=cst.Dict(extra_dict)
                            )
                        ]
                    )
                else:
                    self.log_decision('logger', 'skipped', 'already has extra parameter')
            return updated_node
        except Exception as e:
            self.log_decision('error', 'exception', f'Error in StructuredLoggingTransformer: {str(e)}')
            self.logger.warning(f'Error in StructuredLoggingTransformer: {str(e)}')
            return updated_node

    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
        """Ensure the module has necessary imports for structured logging."""
        if not self.changes_made:
            return updated_node
        has_uuid_import = False
        for statement in original_node.body:
            if isinstance(statement, cst.SimpleStatementLine):
                for stmt in statement.body:
                    if isinstance(stmt, cst.ImportFrom) and stmt.module and stmt.module.value == 'uuid':
                        for name in stmt.names:
                            if name.name.value == 'uuid4':
                                has_uuid_import = True
        if not has_uuid_import:
            new_body = [
                cst.SimpleStatementLine(body=[
                    cst.ImportFrom(
                        module=cst.Name('uuid'),
                        names=[cst.ImportAlias(name=cst.Name('uuid4'))]
                    )
                ])
            ] + list(updated_node.body)
            return updated_node.with_changes(body=new_body)
        return updated_node

class ExceptionHandlingTransformer(TransformationVisitor):
    """Adds exception handling to critical code sections."""
    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
        """Add try-except blocks to function bodies."""
        try:
            has_try_except = False
            for statement in original_node.body.body:
                if isinstance(statement, cst.Try):
                    has_try_except = True
                    break
            if has_try_except:
                self.log_decision('function_def', 'skipped', f'Function {original_node.name.value} already has try-except')
                return updated_node
            self.changes_made = True
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
                                            cst.FormattedStringText(value='Error in '),
                                            cst.FormattedStringExpression(expression=cst.Name(original_node.name.value)),
                                            cst.FormattedStringText(value=': '),
                                            cst.FormattedStringExpression(expression=cst.Call(
                                                func=cst.Name('str'),
                                                args=[cst.Arg(value=cst.Name('e'))]
                                            ))
                                        ]
                                    ))]
                                ))
                            ]),
                            cst.SimpleStatementLine(body=[cst.Raise()])
                        ])
                    )
                ]
            )
            self.log_decision('function_def', 'transformed', f'Added try-except to {original_node.name.value}')
            return updated_node.with_changes(body=cst.IndentedBlock(body=[try_block]))
        except Exception as e:
            self.log_decision('error', 'exception', f'Error in ExceptionHandlingTransformer: {str(e)}')
            self.logger.warning(f'Error in ExceptionHandlingTransformer: {str(e)}')
            return updated_node