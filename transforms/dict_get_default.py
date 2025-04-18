"""
Transformer to add default values to dictionary get methods.
"""
import libcst as cst
from code_updater import TransformationVisitor

class DictGetDefaultTransformer(TransformationVisitor):
    """Adds default values to dict.get() calls without them."""
    def leave_Call(self, original_node: cst.Call, updated_node: cst.Call) -> cst.Call:
        """Add default value to dict.get calls."""
        try:
            if (isinstance(original_node.func, cst.Attribute) and
                    original_node.func.attr.value == 'get' and
                    isinstance(original_node.func.value, cst.Name) and
                    len(original_node.args) == 1):
                self.changes_made = True
                self.log_decision('dict_get', 'transformed', 'Added default None to dict.get call')
                return updated_node.with_changes(
                    args=updated_node.args + [cst.Arg(value=cst.Name('None'))]
                )
            self.log_decision('dict_get', 'skipped', 'Not a single-argument dict.get call')
            return updated_node
        except Exception as e:
            self.log_decision('error', 'exception', f'Error in DictGetDefaultTransformer: {str(e)}')
            self.logger.warning(f'Error in DictGetDefaultTransformer: {str(e)}')
            return updated_node