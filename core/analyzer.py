# core/analyzer.py
from .ast_parser import AstParser
from .suggestion_generator import SuggestionGenerator
from .report_formatter import ReportFormatter
import logging
import time
import shelve
import re
from typing import Dict, List, Tuple
from pathlib import Path
from asyncio import gather
import sys
import ast

from config import config_manager  # Assumes ConfigManager from previous suggestion

logger = logging.getLogger(__name__)

def handle_errors(func):
    """Decorator to handle and log exceptions."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            raise
    return wrapper

class CodeAnalyzer:
    def __init__(self, model, prompts: Dict[str, str], token_limit: int = 16000):
        """Initialize with a model and prompts."""
        self.model = model
        self.prompts = prompts
        self.token_limit = token_limit
        self.cache_file = "analyzer_cache.db"  # Persistent cache
        self.metrics = {"analysis_time": 0.0, "token_usage": 0, "success_rate": 0.0}
        self.ast_parser = AstParser()
        self.suggestion_gen = SuggestionGenerator(model, prompts, token_limit)
        self.report_formatter = ReportFormatter(config_manager.get("HOME_DIR"))
        
        # Validate prompts
        required_tasks = {"analyze", "assess", "sanitize", "propose"}
        if missing := required_tasks - set(self.prompts.keys()):
            raise ValueError(f"Missing required prompts: {missing}")

    async def _generate_with_cache(self, key: str, prompt: str) -> str:
        """Generate response with persistent caching."""
        with shelve.open(self.cache_file) as cache:
            cache_key = f"{key}:{hash(prompt)}"
            if cache_key in cache:
                logger.info(f"Using cached result for {key}")
                return cache[cache_key]
            try:
                result = self.model.generate(prompt)
                cache[cache_key] = result
                return result
            except Exception as e:
                logger.error(f"Model generation failed for {key}: {e}")
                return f"Error: {e}"

    async def analyze(self, files: Dict[str, str], cached_summary: str) -> str:
        """Analyze the package structure and purpose."""
        prompt = self.suggestion_gen.format_prompt("analyze", cached_summary)
        return await self._generate_with_cache("analyze", prompt)

    async def assess(self, files: Dict[str, str], cached_summary: str) -> str:
        """Assess complexity, bugs, and clarity."""
        prompt = self.suggestion_gen.format_prompt("assess", cached_summary)
        return await self._generate_with_cache("assess", prompt)

    async def sanitize(self, files: Dict[str, str], assessment: str, cached_summary: str) -> str:
        """Sanitize the assessment to extract suggestions."""
        prompt = self.suggestion_gen.format_prompt("sanitize", cached_summary, assessment=assessment)
        return await self._generate_with_cache("sanitize", prompt)

    async def propose(self, files: Dict[str, str], suggestions: str) -> Dict[str, str]:
        """Generate optimized code proposals."""
        if not suggestions:  # Handle empty suggestions gracefully
            logger.warning("No suggestions provided for proposal; returning original files")
            return files
        total_size = sum(len(self.ast_parser.summarize_file_full(f, c)) for f, c in files.items()) + len(suggestions)
        if total_size < self.token_limit:
            code_input = "\n\n".join(self.ast_parser.summarize_file_full(f, c) for f, c in files.items())
            prompt = self.suggestion_gen.format_prompt("propose", code_input, suggestions=suggestions)
            result = await self._generate_with_cache("propose", prompt)
            try:
                parsed = eval(result)
                if isinstance(parsed, dict):
                    return {k: v for k, v in parsed.items() if k in files}
            except:
                pattern = r"['\"]?(/.+?)['\"]?:\s*['\"](.*?)(?=['\"](?:,\s*['\"]?/|\s*\})|$)"
                matches = re.findall(pattern, result, re.DOTALL)
                if matches:
                    return {path: code.replace('\\n', '\n').strip() for path, code in matches if path in files}
                logger.warning(f"Failed to parse proposal output: {result[:100]}...")
                return files
        else:
            tasks = [self._propose_file(f, c, suggestions) for f, c in files.items()]
            return dict(await gather(*tasks))
        return files

    async def _propose_file(self, filename: str, code: str, suggestions: str) -> Tuple[str, str]:
        """Propose changes for a single file."""
        file_content = self.ast_parser.summarize_file_full(filename, code)
        prompt = self.suggestion_gen.format_prompt("propose", file_content, suggestions=suggestions)
        if len(prompt) > self.token_limit:
            logger.warning(f"File {filename} too large, using original")
            return (filename, code)
        result = await self._generate_with_cache(f"propose:{filename}", prompt)
        try:
            parsed = eval(result)
            if isinstance(parsed, dict) and filename in parsed:
                return (filename, parsed[filename])
        except:
            code_block = re.search(r"```(?:python)?\s*(.*?)\s*```", result, re.DOTALL)
            if code_block:
                return (filename, code_block.group(1).strip())
        logger.warning(f"Failed to parse proposal for {filename}")
        return (filename, code)

    def validate_components(self, files: Dict[str, str]) -> List[str]:
        """Validate components in proposed files."""
        defined = set()
        referenced = set()
        builtins = set(__builtins__.keys())
        stdlib = set(sys.stdlib_module_names)
        local_symbols = {'ModelFactory', 'CodeAnalyzer', 'FileHandler', 'run_cli', 'parse_args', 'run_cli_async'}
        ignorable = {"HOME_DIR", "DEEPSEEK_API_KEY", "PROMPTS", "SUPPORTED_EXTENSIONS"}
        ignorable.update(sys.modules.keys())
        ignorable.update(local_symbols)
        ignorable.update({'Dict', 'List', 'Tuple', 'Optional', 'Any', 'Path', 'Callable'})
        for path, code in files.items():
            try:
                tree = ast.parse(code)
                defined.update(n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef))
                for node in ast.walk(tree):
                    if isinstance(node, (ast.Import, ast.ImportFrom)):
                        referenced.update(n.name.split('.')[0] for n in node.names)
                    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                        referenced.add(node.func.id)
            except SyntaxError:
                logger.error(f"Syntax error in {path}, skipping validation")
        missing = referenced - defined - builtins - stdlib - ignorable
        if missing:
            logger.warning(f"Missing definitions: {missing}")
        return list(missing)

    def check_imports(self, files: Dict[str, str], project_files: List[str]) -> List[str]:
        """Check for unresolved imports."""
        unresolved = []
        project_modules = {Path(f).stem for f in project_files}
        for path, code in files.items():
            imports = re.findall(r"^(?:from\s+([\w.]+)\s+)?import\s+([\w, ]+)", code, re.MULTILINE)
            for from_part, import_part in imports:
                module = from_part or import_part.split(",")[0].strip()
                if module in project_modules or module in __builtins__ or module in sys.modules:
                    continue
                try:
                    __import__(module)
                except ImportError:
                    unresolved.append(f"{module} in {path}")
        if unresolved:
            logger.warning(f"Unresolved imports: {unresolved}")
        return unresolved

    def repair_proposal(self, original_files: Dict[str, str], proposed_files: Dict[str, str], 
                        missing_components: List[str], unresolved_imports: List[str], suggestions: str) -> Dict[str, str]:
        """Repair proposals with syntax validation."""
        repairs = {}
        for imp in unresolved_imports:
            file_path, module = imp.split(" in ", 1)
            if "Config" in module and file_path.endswith("cli.py"):
                repairs[file_path] = f"from config import Config\n" + proposed_files[file_path]
                logger.info(f"Added import for Config in {file_path}")
        for comp in missing_components:
            if comp == "DeepSeekProvider" and "core/model.py" in proposed_files:
                stub = """
class DeepSeekProvider(DeepSeekModel):
    def __init__(self, api_key):
        super().__init__(api_key=api_key)
"""
                repairs["core/model.py"] = proposed_files["core/model.py"].replace("class DeepSeekModel", stub + "\nclass DeepSeekModel")
                logger.info("Stubbed DeepSeekProvider in core/model.py")
            elif comp == "ComplexityMetrics" and "core/analyzer.py" in proposed_files:
                stub = """
class ComplexityMetrics:
    def analyze(self, code: str) -> dict:
        return {"lines": len(code.splitlines()), "complexity": 0}
"""
                repairs["core/analyzer.py"] = stub + proposed_files["core/analyzer.py"]
                logger.info("Stubbed ComplexityMetrics in core/analyzer.py")
        updated_files = {**proposed_files, **repairs}
        
        # Syntax validation
        for filename, code in updated_files.items():
            try:
                ast.parse(code)
            except SyntaxError as e:
                logger.error(f"Syntax error in proposed {filename}: {e}")
                updated_files[filename] = original_files[filename]
        
        if repairs:
            code_input = "\n\n".join(f"# {k}\n{v}" for k, v in original_files.items())
            prompt = self.suggestion_gen.format_prompt("propose", code_input, suggestions=suggestions)
            result = self.model.generate(prompt)
            try:
                parsed = eval(result)
                if isinstance(parsed, dict):
                    return {k: v for k, v in parsed.items() if k in original_files}
            except:
                logger.warning("Failed to parse re-proposed output, using repaired files")
                return updated_files
        return updated_files

    @handle_errors
    async def process_package(self, file_handler, file_paths: List[str]) -> bool:
        """Process the package through the analysis pipeline."""
        start_time = time.time()
        files = {path: file_handler.read_file(path) for path in file_paths}
        valid_files = {path: content for path, content in files.items() if content}
        if not valid_files:
            logger.warning("No valid files to process")
            raise ValueError("No valid files to process")
        if len(valid_files) < len(files):
            logger.warning(f"Filtered out {len(files) - len(valid_files)} empty files: {[p for p, c in files.items() if not c]}")

        cached_summary = "\n\n".join(self.ast_parser.summarize_file_lite(f, c) for f, c in valid_files.items())
        
        logger.info("Analyzing package")
        analysis = await self.analyze(valid_files, cached_summary)
        logger.debug(f"Analysis output: {analysis}")

        logger.info("Assessing package quality")
        assessment = await self.assess(valid_files, cached_summary)
        logger.debug(f"Assessment output: {assessment}")

        logger.info("Sanitizing assessment")
        sanitized = await self.sanitize(valid_files, assessment, cached_summary)
        logger.debug(f"Sanitized output: {sanitized}")
        suggestions_match = re.search(r"Suggestions:\n(.*?)(?=Code:|$)", sanitized, re.DOTALL)
        suggestions = suggestions_match.group(1).strip() if suggestions_match else ""
        logger.debug(f"Extracted suggestions: {suggestions}")

        logger.info("Generating proposals")
        proposed_files = await self.propose(valid_files, suggestions)

        missing_components = self.validate_components(proposed_files)
        unresolved_imports = self.check_imports(proposed_files, file_paths)
        if missing_components or unresolved_imports:
            logger.info("Issues detected, repairing")
            proposed_files = self.repair_proposal(valid_files, proposed_files, missing_components, unresolved_imports, suggestions)

        self.metrics["analysis_time"] = time.time() - start_time
        self.metrics["token_usage"] = sum(len(c) for c in valid_files.values()) // 4
        self.metrics["success_rate"] = 1.0 if proposed_files else 0.0

        content = self.report_formatter.format(analysis, assessment, suggestions, 
                                               {"missing": missing_components, "unresolved": unresolved_imports}, 
                                               proposed_files, self.metrics)
        file_handler._write_file(file_handler.proposals_file, content, append=True)
        logger.info(f"Proposal written to {file_handler.proposals_file}")
        return True

    @handle_errors
    def read_latest_proposal(self, file_handler) -> Dict[str, str]:
        """Read the latest proposal from proposals.txt."""
        content = file_handler._read_file(file_handler.proposals_file)
        matches = re.findall(r"=== Package: .*? ===\n(.*?)====================\n", content, re.DOTALL)
        if not matches:
            raise ValueError("No package proposals found")
        
        latest = matches[-1]
        code_section = re.search(r"Proposed Code:\n(.*?)$", latest, re.DOTALL)
        if not code_section:
            raise ValueError("No proposed code section found")
        
        proposed_files = {}
        current_file = None
        current_code = []
        for line in code_section.group(1).splitlines():
            if line.startswith("# "):
                if current_file and current_code:
                    proposed_files[current_file] = "\n".join(current_code).strip()
                current_file = line[2:].strip()
                current_code = []
            elif current_file:
                current_code.append(line)
        if current_file and current_code:
            proposed_files[current_file] = "\n".join(current_code).strip()
        
        if not proposed_files:
            raise ValueError("No valid files parsed from proposal")
        return proposed_files