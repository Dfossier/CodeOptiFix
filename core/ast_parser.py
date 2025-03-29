import ast
from typing import Dict

class AstParser:
    """Handles AST-based code parsing and summarization."""
    
    def __init__(self):
        self.cache: Dict[str, str] = {}

    def summarize_file_lite(self, filename: str, code: str) -> str:
        if filename in self.cache:
            return self.cache[filename]
        imports = []
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    import_line = ast.get_source_segment(code, node)
                    if import_line:
                        imports.append(import_line)
        except SyntaxError as e:
            logger.error(f"Syntax error in {filename}: {e}")
        imports_str = "\n".join(filter(None, imports))
        lines = code.splitlines()
        key_line = next((line.strip() for line in lines if line.strip() and not line.strip().startswith(("import", "from"))), "No key line")
        result = f"# {filename}\n{len(lines)} lines, {len(code)} chars\nImports:\n{imports_str}\nKey Line: {key_line}\n"
        self.cache[filename] = result
        return result

    def summarize_file_full(self, filename: str, code: str) -> str:
        return f"# {filename}\n{code}\n"