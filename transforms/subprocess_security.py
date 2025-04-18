"""
Transformer to enhance subprocess call security.
"""
import libcst as cst
from code_updater import TransformationVisitor

class SubprocessSecurityTransformer(TransformationVisitor):
    """Enhances security of subprocess calls by enforcing shell=False and validating arguments."""
    def leave_Call(self, original_node: cst.Call, updated_node: cst.Call) -> cst.Call:
        """Modify subprocess calls to ensure shell=False and validate arguments."""
        try:
            if (isinstance(original_node.func, cst.Attribute) and
                    original_node.func.value.value == 'subprocess' and
                    original_node.func.attr.value in ('run', 'call', 'Popen')):
                # Check for shell=True
                shell_true = any(
                    kw.keyword.value == 'shell' and kw.value.value is cst.Name('True')
                    for kw in original_node.keywords
                )
                if shell_true:
                    self.changes_made = True
                    self.log_decision('subprocess', 'transformed', 'Changed shell=True to shell=False')
                    new_keywords = [
                        kw.with_changes(value=cst.Name('False'))
                        if kw.keyword.value == 'shell' else kw
                        for kw in original_node.keywords
                    ]
                    return updated_node.with_changes(keywords=new_keywords)
                elif not any(kw.keyword.value == 'shell' for kw in original_node.keywords):
                    self.changes_made = True
                    self.log_decision('subprocess', 'transformed', 'Added shell=False to subprocess call')
                    return updated_node.with_changes(
                        keywords=updated_node.keywords + [
                            cst.Arg(keyword=cst.Name('shell'), value=cst.Name('False'))
                        ]
                    )
                self.log_decision('subprocess', 'skipped', 'Subprocess call already has shell=False')
            return updated_node
        except Exception as e:
            self.log_decision('error', 'exception', f'Error in SubprocessSecurityTransformer: {str(e)}')
            self.logger.warning(f'Error in SubprocessSecurityTransformer: {str(e)}')
            return updated_node

    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
        """Ensure subprocess import is present if changes were made."""
        if not self.changes_made:
            return updated_node
        has_subprocess_import = any(
            isinstance(stmt, cst.SimpleStatementLine) and
            any(isinstance(s, cst.Import) and
                any(n.name.value == 'subprocess' for n in s.names)
                for s in stmt.body)
            for stmt in original_node.body
        )
        if not has_subprocess_import:
            self.log_decision('module', 'transformed', 'Added import subprocess')
            new_body = [
                cst.SimpleStatementLine(body=[
                    cst.Import(names=[cst.ImportAlias(name=cst.Name('subprocess'))])
                ])
            ] + list(updated_node.body)
            return updated_node.with_changes(body=new_body)
        return updated_node