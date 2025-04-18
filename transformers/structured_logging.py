"""
Structured Logging Transformer

Transforms print statements and basic logging to structured logging with context.
"""
import libcst as cst
from typing import Dict, List, Any, Optional, Tuple, Union

from code_updater import TransformationVisitor  # Adjusted to match code_updater.py

class StructuredLoggingTransformer(TransformationVisitor):
    """Base class for adding structured logging to code."""
    
    def __init__(self, transformation_data: Dict[str, Any] = None):
        """Initialize with optional transformation data."""
        super().__init__(transformation_data)
        
    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        """Track the current function name for context."""
        if not hasattr(self, "current_function"):
            self.current_function = []
        self.current_function.append(node.name.value)
    
    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
        """Clear the current function name when leaving the function."""
        if hasattr(self, "current_function") and self.current_function and self.current_function[-1] == original_node.name.value:
            self.current_function.pop()
        return updated_node
    
    def leave_Call(self, original_node: cst.Call, updated_node: cst.Call) -> cst.Call:
        """Replace print calls with logging calls or enhance existing logging."""
        try:
            # Case 1: Handle print statements
            if isinstance(original_node.func, cst.Name) and original_node.func.value == "print":
                self.log_decision("print", "analyzing", "found print statement")
                
                # Determine the logging level based on the arguments or context
                log_level = "info"  # Default level
                log_content = "unknown"
                
                # Check if this is an error message
                if original_node.args:
                    arg_value = None
                    
                    if isinstance(original_node.args[0].value, cst.SimpleString):
                        arg_value = original_node.args[0].value.value
                        log_content = arg_value[:30] + "..." if len(arg_value) > 30 else arg_value
                    
                    if arg_value and any(kw in arg_value.lower() for kw in ["error", "exception", "fail", "failed"]):
                        log_level = "error"
                        self.log_decision("print", "classified", f"identified as error log: '{log_content}'")
                    elif arg_value and any(kw in arg_value.lower() for kw in ["debug", "trace"]):
                        log_level = "debug"
                        self.log_decision("print", "classified", f"identified as debug log: '{log_content}'")
                    elif arg_value and any(kw in arg_value.lower() for kw in ["warn", "warning"]):
                        log_level = "warning"
                        self.log_decision("print", "classified", f"identified as warning log: '{log_content}'")
                    else:
                        self.log_decision("print", "classified", f"default to info log: '{log_content}'")
                
                func_context = None
                if hasattr(self, "current_function") and self.current_function:
                    func_context = self.current_function[-1]
                    self.log_decision("print", "context", f"adding function context: '{func_context}'")
                else:
                    self.log_decision("print", "context", "no function context available")
                
                self.changes_made = True
                self.log_decision("print", "transformed", f"converted to logger.{log_level}")
                
                extra_dict = [
                    cst.DictElement(
                        key=cst.SimpleString("'function'"),
                        value=cst.SimpleString(f"'{func_context}'")
                    ),
                    cst.DictElement(
                        key=cst.SimpleString("'trace_id'"),
                        value=cst.Call(
                            func=cst.Name("str"),
                            args=[cst.Arg(cst.Call(func=cst.Name("uuid4")))]
                        )
                    )
                ] if func_context else [
                    cst.DictElement(
                        key=cst.SimpleString("'trace_id'"),
                        value=cst.Call(
                            func=cst.Name("str"),
                            args=[cst.Arg(cst.Call(func=cst.Name("uuid4")))]
                        )
                    )
                ]
                return cst.Call(
                    func=cst.Attribute(
                        value=cst.Name("logger"),
                        attr=cst.Name(log_level)
                    ),
                    args=original_node.args,
                    keywords=[
                        cst.Arg(
                            keyword=cst.Name("extra"),
                            value=cst.Dict(extra_dict)
                        )
                    ]
                )
            
            # Case 2: Enhance existing logging
            elif (isinstance(original_node.func, cst.Attribute) and 
                  isinstance(original_node.func.value, cst.Name) and
                  original_node.func.value.value == "logger"):
                
                self.log_decision("logger", "analyzing", "found existing logger call")
                
                log_level = original_node.func.attr.value if hasattr(original_node.func.attr, "value") else "unknown"
                self.log_decision("logger", "info", f"log level: {log_level}")
                
                if not any(kw.keyword and kw.keyword.value == "extra" for kw in original_node.keywords):
                    self.log_decision("logger", "enhancing", "no 'extra' parameter found, will add context")
                    
                    func_context = None
                    if hasattr(self, "current_function") and self.current_function:
                        func_context = self.current_function[-1]
                        self.log_decision("logger", "context", f"adding function context: '{func_context}'")
                    else:
                        self.log_decision("logger", "context", "no function context available")
                    
                    extra_dict = [
                        cst.DictElement(
                            key=cst.SimpleString("'function'"),
                            value=cst.SimpleString(f"'{func_context}'")
                        ),
                        cst.DictElement(
                            key=cst.SimpleString("'trace_id'"),
                            value=cst.Call(
                                func=cst.Name("str"),
                                args=[cst.Arg(cst.Call(func=cst.Name("uuid4")))]
                            )
                        )
                    ] if func_context else [
                        cst.DictElement(
                            key=cst.SimpleString("'trace_id'"),
                            value=cst.Call(
                                func=cst.Name("str"),
                                args=[cst.Arg(cst.Call(func=cst.Name("uuid4")))]
                            )
                        )
                    ]
                    self.changes_made = True
                    self.log_decision("logger", "transformed", "added extra context parameter")
                    return updated_node.with_changes(
                        keywords=list(updated_node.keywords) + [
                            cst.Arg(
                                keyword=cst.Name("extra"),
                                value=cst.Dict(extra_dict)
                            )
                        ]
                    )
                else:
                    self.log_decision("logger", "skipped", "already has 'extra' parameter with context")
            
            return updated_node
            
        except Exception as e:
            self.log_decision("error", "exception", f"Error in StructuredLoggingTransformer: {str(e)}")
            self.logger.warning(f"Error in StructuredLoggingTransformer: {str(e)}")
            return updated_node
    
    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
        """Ensure the module has the necessary imports."""
        if not self.changes_made:
            return updated_node
        
        has_logging_import = False
        has_setup_logging_import = False
        has_uuid_import = False
        
        for statement in original_node.body:
            if isinstance(statement, cst.SimpleStatementLine):
                for stmt in statement.body:
                    if isinstance(stmt, cst.Import):
                        for name in stmt.names:
                            if name.name.value == "logging":
                                has_logging_import = True
                            elif name.name.value == "uuid":
                                has_uuid_import = True
                    elif isinstance(stmt, cst.ImportFrom):
                        if stmt.module and hasattr(stmt.module, 'value'):
                            if stmt.module.value == "utils":
                                for name in stmt.names:
                                    if name.name.value == "setup_logging":
                                        has_setup_logging_import = True
                            elif stmt.module.value == "uuid":
                                for name in stmt.names:
                                    if name.name.value == "uuid4":
                                        has_uuid_import = True
        
        imports_to_add = []
        
        if not has_logging_import:
            imports_to_add.append(
                cst.SimpleStatementLine(
                    body=[
                        cst.Import(
                            names=[cst.ImportAlias(name=cst.Name("logging"))]
                        )
                    ]
                )
            )
        
        if not has_uuid_import:
            imports_to_add.append(
                cst.SimpleStatementLine(
                    body=[
                        cst.ImportFrom(
                            module=cst.Name("uuid"),
                            names=[cst.ImportAlias(name=cst.Name("uuid4"))]
                        )
                    ]
                )
            )
        
        if not has_setup_logging_import:
            imports_to_add.append(
                cst.SimpleStatementLine(
                    body=[
                        cst.ImportFrom(
                            module=cst.Name("utils"),
                            names=[cst.ImportAlias(name=cst.Name("setup_logging"))]
                        )
                    ]
                )
            )
        
        has_logger_init = False
        for statement in original_node.body:
            if isinstance(statement, cst.SimpleStatementLine):
                for stmt in statement.body:
                    if isinstance(stmt, cst.Assign) and len(stmt.targets) == 1:
                        target = stmt.targets[0].target
                        if isinstance(target, cst.Name) and target.value == "logger":
                            has_logger_init = True
        
        if not has_logger_init:
            imports_to_add.append(
                cst.SimpleStatementLine(
                    body=[
                        cst.Assign(
                            targets=[cst.AssignTarget(target=cst.Name("logger"))],
                            value=cst.Call(
                                func=cst.Name("setup_logging"),
                                args=[cst.Arg(value=cst.Name("__name__"))]
                            )
                        )
                    ]
                )
            )
        
        if imports_to_add:
            inserted_imports = False
            new_body = []
            for statement in updated_node.body:
                if (not inserted_imports and 
                    isinstance(statement, cst.SimpleStatementLine) and 
                    any(isinstance(stmt, (cst.Import, cst.ImportFrom)) for stmt in statement.body)):
                    new_body.append(statement)
                    new_body.extend(imports_to_add)
                    inserted_imports = True
                else:
                    new_body.append(statement)
            
            if not inserted_imports:
                new_body = imports_to_add + new_body
            
            return updated_node.with_changes(body=new_body)
        
        return updated_node


class StructuredLoggingConditionalTransformer(StructuredLoggingTransformer):
    """Adds structured logging to conditional logic."""
    
    def is_potential_target(self, node: cst.CSTNode) -> bool:
        """Check if this node is a potential target for transformation."""
        return isinstance(node, cst.If)
    
    def leave_If(self, original_node: cst.If, updated_node: cst.If) -> cst.If:
        """Add structured logging around conditional logic."""
        try:
            # Skip 'if __name__ == "__main__"' blocks
            if (isinstance(original_node.test, cst.Comparison) and 
                isinstance(original_node.test.left, cst.Name) and 
                original_node.test.left.value == "__name__"):
                self.log_decision("if_statement", "skipped", "Main module check, no logging needed")
                return updated_node
            
            # Check for existing structured logging
            has_structured_logging = False
            for statement in original_node.body.body:
                if isinstance(statement, cst.SimpleStatementLine):
                    for expr in statement.body:
                        if isinstance(expr, cst.Expr) and isinstance(expr.value, cst.Call):
                            if (isinstance(expr.value.func, cst.Attribute) and
                                isinstance(expr.value.func.value, cst.Name) and
                                expr.value.func.value.value == "logger" and
                                any(kw.keyword and kw.keyword.value == "extra" for kw in expr.value.keywords)):
                                has_structured_logging = True
            
            if has_structured_logging:
                self.log_decision("if_statement", "skipped", "Already has structured logging")
                return updated_node
            
            func_context = "unknown_function"
            if hasattr(self, "current_function") and self.current_function:
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
                self.log_decision("if_statement", "transformed", f"Added structured logging for condition: {condition_code}")
                return parsed_if
            except Exception as e:
                self.log_decision("if_statement", "error", f"Failed to parse logging code: {str(e)}")
                self.logger.warning(f"Error parsing logging code: {str(e)}")
                return updated_node
        except Exception as e:
            self.log_decision("if_statement", "error", f"Error enhancing conditional: {str(e)}")
            self.logger.warning(f"Error in StructuredLoggingConditionalTransformer: {str(e)}")
            return updated_node


class StructuredLoggingErrorTransformer(StructuredLoggingTransformer):
    """Adds structured logging to error handling code."""
    
    def is_potential_target(self, node: cst.CSTNode) -> bool:
        """Check if this node is a potential target for transformation."""
        return isinstance(node, cst.ExceptHandler)
    
    def leave_ExceptHandler(self, original_node: cst.ExceptHandler, updated_node: cst.ExceptHandler) -> cst.ExceptHandler:
        """Enhance except blocks with structured logging."""
        try:
            has_structured_logging = False
            for statement in original_node.body.body:
                if isinstance(statement, cst.SimpleStatementLine):
                    for expr in statement.body:
                        if isinstance(expr, cst.Expr) and isinstance(expr.value, cst.Call):
                            if (isinstance(expr.value.func, cst.Attribute) and
                                isinstance(expr.value.func.value, cst.Name) and
                                expr.value.func.value.value == "logger" and
                                any(kw.keyword and kw.keyword.value == "extra" for kw in expr.value.keywords)):
                                has_structured_logging = True
            
            if has_structured_logging:
                self.log_decision("except_handler", "skipped", "Already has structured logging")
                return updated_node
                
            self.changes_made = True
            exception_name = original_node.name.value if original_node.name else "e"
            func_context = "unknown_function"
            if hasattr(self, "current_function") and self.current_function:
                func_context = self.current_function[-1]
            
            log_stmt = f"""
except {original_node.type.value if original_node.type else "Exception"} as {exception_name}:
    logger.error(f"Error occurred: {{str({exception_name})}}", extra={{'exc_type': {exception_name}.__class__.__name__, 'function': '{func_context}', 'trace_id': str(uuid4())[:8]}})
    {cst.Module(body=[cst.IndentedBlock(body=list(original_node.body.body))]).code.strip()}
"""
            try:
                parsed_except = cst.parse_statement(log_stmt.strip())
                self.log_decision("except_handler", "transformed", f"Added structured logging for exception: {exception_name}")
                return parsed_except
            except Exception as e:
                self.log_decision("except_handler", "error", f"Failed to parse logging code: {str(e)}")
                self.logger.warning(f"Error parsing logging code: {str(e)}")
                return updated_node
        except Exception as e:
            self.log_decision("except_handler", "error", f"Error enhancing except handler: {str(e)}")
            self.logger.warning(f"Error in StructuredLoggingErrorTransformer: {str(e)}")
            return updated_node