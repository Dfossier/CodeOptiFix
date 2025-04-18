"""
Transformer to document return values in function docstrings.
"""
import libcst as cst
from code_updater import TransformationVisitor

class DocReturnValuesTransformer(TransformationVisitor):
    """Adds or enhances docstrings to document function return values."""
    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
        """Add Returns section to function docstrings."""
        try:
            if not original_node.docstring:
                self.changes_made = True
                self.log_decision('function_def', 'transformed', f'Added docstring to {original_node.name.value}')
                return updated_node.with_changes(
                    docstring=cst.SimpleString('"""Returns: None"""')
                )
            docstring = original_node.docstring.value
            if 'Returns:' not in docstring:
                self.changes_made = True
                self.log_decision('function_def', 'transformed', f'Enhanced docstring for {original_node.name.value}')
                new_docstring = f'{docstring.rstrip()}\n    Returns:\n        None'
                return updated_node.with_changes(
                    docstring=cst.SimpleString(f'"""{new_docstring}"""')
                )
            self.log_decision('function_def', 'skipped', f'Docstring for {original_node.name.value} already has Returns')
            return updated_node
        except Exception as e:
            self.log_decision('error', 'exception', f'Error in DocReturnValuesTransformer: {str(e)}')
            self.logger.warning(f'Error in DocReturnValuesTransformer: {str(e)}')
            return updated_node