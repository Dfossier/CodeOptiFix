"""
Goal Refiner module for refining and enhancing improvement goals.

Takes raw improvement goals and refines them based on code analysis and context.
"""
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from code_analyzer import CodeAnalyzer
from interfaces import ImprovementGoal
from utils import setup_logging
from observatory.code_state_manager import CodeStateManager
from outcome_repository.outcome_analyzer import OutcomeAnalyzer
from outcome_repository.outcome_logger import OutcomeLogger

logger = setup_logging(__name__)

class GoalRefiner:
    """Refines raw improvement goals based on code analysis and context."""
    def __init__(
        self,
        code_state_manager: Optional[CodeStateManager] = None,
        outcome_analyzer: Optional[OutcomeAnalyzer] = None,
        outcome_logger: Optional[OutcomeLogger] = None,
        base_path: Optional[Path] = None
    ):
        self.code_state_manager = code_state_manager
        self.outcome_analyzer = outcome_analyzer
        self.outcome_logger = outcome_logger
        self.base_path = (code_state_manager.base_path if code_state_manager
                         else base_path or Path.cwd())
        self.code_analyzer = CodeAnalyzer(self.base_path)
        self.logger = logger

    def refine_goals(self, goals: List[Dict[str, Any]]) -> List[ImprovementGoal]:
        """Refine a list of raw improvement goals."""
        refined_goals = []
        self.logger.info(f'Refining {len(goals)} goal validations')
        for goal in goals:
            try:
                self.logger.debug(f"Raw goal data: {goal}")
                refined_goal = self._refine_single_goal(goal)
                if refined_goal:
                    refined_goals.append(refined_goal)
                    if self.outcome_logger:
                        self.outcome_logger.log_event(
                            "goal_refined",
                            {"goal": goal, "refined_goal": refined_goal.to_dict()}
                        )
            except Exception as e:
                self.logger.error(f"Error refining goal '{goal.get('description', 'unknown')}': {str(e)}")
        self.logger.info(f"Refined {len(refined_goals)} goals")
        return refined_goals

    def _refine_single_goal(self, goal: Dict[str, Any]) -> Optional[ImprovementGoal]:
        """Refine a single improvement goal."""
        # Ensure target_module and description are strings
        target_module = goal.get('target_module')
        if isinstance(target_module, Path):
            target_module = str(target_module)
        elif not isinstance(target_module, str):
            self.logger.warning(f"Invalid target_module type {type(target_module)} for goal: {goal}, skipping")
            return None

        description = goal.get('description', '')
        if isinstance(description, Path):
            self.logger.warning(f"Description is a Path object: {description}, converting to string")
            description = str(description)
        elif not isinstance(description, str):
            self.logger.warning(f"Invalid description type {type(description)} for goal: {goal}, using default")
            description = ''

        self.logger.info(f'Refining goal: {description} for {target_module}')

        if not target_module:
            self.logger.warning(f"Goal '{description}' has no target module, skipping")
            return None

        module_path = self.base_path / target_module
        if not module_path.exists():
            self.logger.warning(f"Target module {target_module} does not exist, skipping")
            return None

        # Set improvement_type, checking both 'improvement_type' and 'type'
        if not goal.get('improvement_type'):
            goal_type = goal.get('type', '')
            description_lower = description.lower()
            self.logger.debug(f"Mapping improvement_type for description: {description_lower}")
            if goal_type:
                goal['improvement_type'] = goal_type
            elif 'print with proper logging' in description_lower:
                goal['improvement_type'] = 'replace_print_with_logging'
            elif 'default value to dictionary get' in description_lower:
                goal['improvement_type'] = 'add_dict_get_default'
            elif 'structured logging for better analytics' in description_lower:
                goal['improvement_type'] = 'add_structured_logging_conditional'
            elif 'subprocess security checks' in description_lower:
                goal['improvement_type'] = 'enhance_subprocess_security'
            elif 'document return values' in description_lower:
                goal['improvement_type'] = 'document_return_values'
            elif 'token security measures' in description_lower:
                goal['improvement_type'] = 'enhance_token_security'
            else:
                self.logger.warning(f"No improvement_type mapped for goal '{description}', skipping")
                return None

        # Analyze code to refine target files and functions
        analysis = self.code_analyzer.analyze_module(module_path)
        target_function = goal.get('target_function')
        if target_function and target_function not in analysis.get('functions', []):
            self.logger.warning(f"Target function {target_function} not found in {target_module}, ignoring")
            target_function = None

        # Check past outcomes if outcome_analyzer is available
        if self.outcome_analyzer:
            try:
                past_outcomes = self.outcome_analyzer.get_outcomes_for_module(target_module)
                if past_outcomes and any(o['status'] == 'failed' for o in past_outcomes):
                    self.logger.warning(f"Previous failures for {target_module}, adjusting priority")
                    goal['priority'] = min(goal.get('priority', 1) + 1, 5)
            except AttributeError:
                self.logger.warning("OutcomeAnalyzer.get_outcomes_for_module not available, skipping")

        try:
            refined_goal = ImprovementGoal(
                target_module=target_module,
                description=description,
                improvement_type=goal['improvement_type'],
                target_function=target_function,
                performance_target=goal.get('performance_target'),
                priority=goal.get('priority', 1)
            )
            self.logger.debug(f"Created refined goal: {refined_goal.to_dict()}")
            return refined_goal
        except Exception as e:
            self.logger.error(f"Error creating ImprovementGoal for '{description}': {str(e)}")
            return None