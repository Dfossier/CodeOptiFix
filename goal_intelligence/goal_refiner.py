"""
Goal Refiner for Goal Intelligence Framework.

Refines validated goals based on code state to ensure feasibility.
"""
import logging
from typing import Dict, List, Any
from observatory.code_state_manager import CodeStateManager

from utils import setup_logging

logger = setup_logging(__name__)

class GoalRefiner:
    def __init__(self, code_state_manager: CodeStateManager):
        self.code_state_manager = code_state_manager
        self.logger = logger

    def refine_goal(self, goal_validation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Refine a single goal validation result based on code state.
        """
        self.logger.debug(f"Processing goal validation: {goal_validation}")
        
        if not isinstance(goal_validation, dict) or "goal" not in goal_validation:
            self.logger.error(f"Invalid goal validation structure: {goal_validation}")
            return {"goal": {}, "is_valid": False, "reason": "Malformed validation dict"}
        
        goal = goal_validation.get("goal")
        if not isinstance(goal, dict):
            self.logger.error(f"Goal is not a dict: {goal}")
            return {"goal": goal, "is_valid": False, "reason": "Goal must be a dictionary"}
        
        description = goal.get("description", "No description provided")
        target_module = goal.get("target_module", "No target module")
        self.logger.info(f"Refining goal: {description} for {target_module}")

        if not goal_validation.get("is_valid", False):
            reason = goal_validation.get("reason", "Unknown validation failure")
            self.logger.info(f"Goal invalid, skipping refinement: {reason}")
            return goal_validation

        if not target_module or target_module == "No target module" or target_module not in self.code_state_manager.get_files():
            self.logger.warning(f"Invalid target_module: {target_module}")
            return {
                "goal": goal,
                "is_valid": False,
                "reason": f"Target module {target_module} not in codebase",
                "refined": False
            }

        refined_goal = goal.copy()
        if "type" not in refined_goal:
            refined_goal["type"] = "replace_print_with_logging"
        
        refined_validation = {
            "goal": refined_goal,
            "is_valid": True,
            "reason": "Goal refined successfully",
            "refined": True
        }
        self.logger.debug(f"Refined goal validation: {refined_validation}")
        return refined_validation

    def refine_goals(self, goal_validations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Refine a list of goal validations.
        """
        self.logger.info(f"Refining {len(goal_validations)} goal validations")
        refined = [self.refine_goal(validation) for validation in goal_validations]
        self.logger.debug(f"Refined goals: {refined}")
        return refined