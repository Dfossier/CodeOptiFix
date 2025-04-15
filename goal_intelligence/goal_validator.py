"""
Goal Validator for the Self-Improving AI Assistant.

Validates improvement goals against the current codebase state.
"""
import logging
from typing import Dict, Any, Optional
from pathlib import Path

from observatory.code_state_manager import CodeStateManager
from utils import setup_logging

logger = setup_logging(__name__)

class GoalValidator:
    def __init__(self, code_state_manager: CodeStateManager, base_path: Optional[Path] = None):
        self.code_state_manager = code_state_manager
        self.base_path = base_path or Path.cwd()
        self.logger = logger
    
    def validate_goal(self, goal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a single improvement goal.
        
        Args:
            goal: Dictionary containing goal details
            
        Returns:
            Dictionary with validation results
        """
        description = goal.get('description', 'unknown')
        self.logger.info(f"Validating goal: {description}")
        validation = {
            "is_valid": False,
            "reason": "",
            "description": description
        }
        
        try:
            target_module = goal.get("target_module")
            if not target_module:
                validation["reason"] = "No target module specified"
                self.logger.warning(f"Goal missing target_module: {goal}")
                return validation
            
            # Normalize path
            normalized_target = str(Path(target_module)).replace("\\", "/")
            tracked_files = self.code_state_manager.get_files()
            self.logger.debug(f"Goal target_module: {normalized_target}")
            self.logger.debug(f"Tracked files: {tracked_files}")
            
            # Fallback for empty tracked files
            known_files = ['utils.py', 'self_improvement_loop.py', 'code_analyzer.py', 'goal_generator.py']
            if not tracked_files:
                self.logger.warning("No tracked files in CodeStateManager, using known files")
                if normalized_target in known_files or (self.base_path / normalized_target).exists():
                    validation["is_valid"] = True
                    self.logger.info(f"Goal valid: {normalized_target} matches known file")
                    return validation
            
            # Check if target_module matches any tracked file
            if (normalized_target in tracked_files or 
                any(normalized_target == str(Path(f).relative_to(self.base_path)).replace("\\", "/") 
                    for f in tracked_files)):
                if not description:
                    self.logger.warning(f"Goal has empty description but valid module: {normalized_target}")
                validation["is_valid"] = True
                self.logger.info(f"Goal valid: {normalized_target} found in codebase")
            else:
                validation["reason"] = f"Target module {normalized_target} not found in codebase"
                self.logger.warning(f"Goal invalid: {normalized_target} not in codebase")
            
            return validation
        
        except Exception as e:
            validation["reason"] = f"Validation error: {str(e)}"
            self.logger.error(f"Error validating goal: {str(e)} - Goal: {goal}")
            return validation