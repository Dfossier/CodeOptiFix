# core/workflow.py
from typing import Dict, List, Callable
import logging

logger = logging.getLogger(__name__)

class WorkflowStep:
    def __init__(self, name: str, func: Callable, depends_on: List[str] = None):
        self.name = name
        self.func = func
        self.depends_on = depends_on or []

class Workflow:
    def __init__(self):
        self.steps: Dict[str, WorkflowStep] = {}
        self.results: Dict[str, any] = {}

    def add_step(self, name: str, func: Callable, depends_on: List[str] = None):
        self.steps[name] = WorkflowStep(name, func, depends_on)

    async def run(self, initial_data: Dict):
        for step_name, step in self.steps.items():
            if not all(dep in self.results for dep in step.depends_on):
                logger.error(f"Missing dependencies for {step_name}: {step.depends_on}")
                return False
            try:
                self.results[step_name] = await step.func(initial_data, self.results)
                logger.info(f"Completed step: {step_name}")
            except Exception as e:
                logger.error(f"Step {step_name} failed: {e}")
                return False
        return True

# core/analyzer.py
class CodeAnalyzer:
    def __init__(self, model, prompts: Dict[str, str], token_limit: int = 16000):
        self.model = model
        self.workflow = Workflow()
        self.workflow.add_step("analyze", self._analyze, [])
        self.workflow.add_step("assess", self._assess, ["analyze"])
        self.workflow.add_step("sanitize", self._sanitize, ["assess"])
        self.workflow.add_step("propose", self._propose, ["sanitize"])
        # Other initialization...

    async def process_package(self, file_handler, file_paths: List[str]) -> bool:
        files = {path: file_handler.read_file(path) for path in file_paths}
        return await self.workflow.run({"files": files, "file_handler": file_handler})