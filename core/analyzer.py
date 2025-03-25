# core/analyzer.py
import logging
import re
from typing import Dict, List
from pydantic import BaseModel as PydanticBaseModel

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', force=True)
logger = logging.getLogger(__name__)

class AnalysisChunk(PydanticBaseModel):
    content: str
    analysis: str = None

class CodeAnalyzer:
    def __init__(self, model, prompts: Dict[str, str], token_limit: int = 8192):
        self.model = model
        self.prompts = prompts
        self.token_limit = token_limit
        logger.debug(f"Initialized with prompts: {list(prompts.keys())}")
        for task in ["analyze", "assess", "sanitize", "propose"]:
            if task not in self.prompts:
                raise ValueError(f"Missing prompt for task: {task}")
            logger.debug(f"Prompt for {task}: {self.prompts[task][:200]}...")
        
    def _truncate_prompt(self, prompt: str) -> str:
        if len(prompt) > self.token_limit:
            logger.warning(f"Prompt too long ({len(prompt)}), truncating to {self.token_limit}")
            return prompt[:self.token_limit]
        return prompt

    def _format_prompt(self, task: str, code: str, assessment: str = None, suggestions: str = None) -> str:
        try:
            logger.debug(f"Raw prompt for {task}: {self.prompts[task]}")
            prompt = self.prompts[task]
            if task == "sanitize":
                if not assessment:
                    raise ValueError("Assessment required for sanitize")
                formatted = prompt.replace("##ASSESSMENT##", assessment).replace("##CODE##", code)
            elif task == "propose":
                if not suggestions:
                    raise ValueError("Suggestions required for propose")
                formatted = prompt.replace("##SUGGESTIONS##", suggestions).replace("##CODE##", code)
            else:
                formatted = prompt.replace("##CODE##", code)
            logger.debug(f"Formatted prompt for {task}: {formatted[:500]}...")
            return formatted
        except Exception as e:
            logger.error(f"Prompt formatting failed for {task}: {e}")
            raise

    def _summarize_file(self, filename: str, code: str) -> str:
        lines = code.splitlines()
        summary = f"# {filename}\n{len(lines)} lines, {len(code)} chars\n"
        if len(lines) > 5:
            summary += "\n".join(lines[:3]) + "\n...\n" + lines[-1]
        else:
            summary += code
        return summary

    def analyze_package(self, files: Dict[str, str], task: str, assessment: str = None, suggestions: str = None) -> str:
        try:
            summarized_code = "\n\n".join(self._summarize_file(f, c) for f, c in files.items())
            prompt = self._format_prompt(task, summarized_code, assessment, suggestions)
            if len(prompt) > self.token_limit:
                summarized_code = "\n".join(f"# {f}\n{c[:100]}..." for f, c in files.items())
                prompt = self._format_prompt(task, summarized_code, assessment, suggestions)
                prompt = self._truncate_prompt(prompt)
            result = self.model.generate(prompt)
            if result.startswith("Error:"):
                raise ValueError(f"Model generation failed: {result}")
            return result
        except Exception as e:
            logger.error(f"Package analysis failed for {task}: {e}")
            raise

    def process_package(self, file_handler, file_paths: List[str]) -> bool:
        try:
            files = {path: file_handler.read_file(path) for path in file_paths}
            if not all(files.values()):
                logger.warning("One or more files empty or unreadable")
                raise ValueError("Empty or unreadable files detected")

            logger.info("Analyzing package")
            analysis = self.analyze_package(files, "analyze")

            logger.info("Assessing package quality")
            assessment = self.analyze_package(files, "assess")

            logger.info("Sanitizing assessment and code")
            sanitized = self.analyze_package(files, "sanitize", assessment=assessment)
            suggestions_match = re.search(r"Suggestions:\n(.*?)(?=Code:)", sanitized, re.DOTALL)
            code_match = re.search(r"Code:\n(.*)", sanitized, re.DOTALL)
            suggestions = suggestions_match.group(1).strip() if suggestions_match else ""
            sanitized_code = code_match.group(1).strip() if code_match else "\n\n".join(self._summarize_file(f, c) for f, c in files.items())
            logger.debug(f"Sanitized suggestions: {suggestions}")
            logger.debug(f"Sanitized code: {sanitized_code[:500]}...")

            logger.info("Generating package-wide proposals")
            proposed_raw = self.analyze_package(files, "propose", suggestions=suggestions)
            logger.debug(f"Raw proposed output: {proposed_raw[:500]}...")
            
            proposed_files = {}
            try:
                proposed_dict = eval(proposed_raw)
                if isinstance(proposed_dict, dict):
                    proposed_files = {k: v for k, v in proposed_dict.items() if k in files}
                else:
                    logger.warning("Proposed code not a dict, attempting text parse")
                    raise ValueError("Not a dict")
            except (SyntaxError, ValueError, NameError) as e:
                logger.error(f"Failed to parse proposed code as dict: {e}. Raw output: {proposed_raw[:100]}...")
                lines = proposed_raw.splitlines()
                current_file = None
                current_code = []
                for line in lines:
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
                    logger.warning("No valid proposed files parsed, using originals")
                    proposed_files = files

            content = (
                f"\n=== Package: {file_handler.base_dir} ===\n"
                f"Analysis:\n{analysis}\n\n"
                f"Assessment:\n{assessment}\n\n"
                f"Suggestions:\n{suggestions}\n\n"
                "Proposed Code:\n"
            )
            for filename, code in proposed_files.items():
                content += f"# {filename}\n{code}\n\n"
            content += "====================\n"

            file_handler._write_file(file_handler.proposals_file, content, append=True)
            logger.info(f"Proposal written to {file_handler.proposals_file}")
            return True
        except Exception as e:
            logger.error(f"Package processing failed: {e}")
            raise

    def read_latest_proposal(self, file_handler) -> Dict[str, str]:
        try:
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
        except Exception as e:
            logger.error(f"Error reading package proposal: {e}")
            raise