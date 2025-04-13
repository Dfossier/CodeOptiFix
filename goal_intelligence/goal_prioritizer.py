"""
Goal Prioritizer for the Self-Improving AI Assistant.

Prioritizes goals based on impact, feasibility, and historical outcomes.
"""
import logging
from typing import List, Dict, Any
import asyncio
from pathlib import Path

from observatory.code_state_manager import CodeStateManager
from outcome_repository.outcome_analyzer import OutcomeAnalyzer
from utils import setup_logging

logger = setup_logging(__name__)

class GoalPrioritizer:
    def __init__(
        self,
        code_state_manager: CodeStateManager,
        outcome_analyzer: OutcomeAnalyzer,
        base_path: Path = None
    ):
        self.code_state_manager = code_state_manager
        self.outcome_analyzer = outcome_analyzer
        self.base_path = base_path or Path.cwd()
        self.logger = logger
    
    async def prioritize_goals(self, validations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Prioritize a list of goal validations.
        
        Args:
            validations: List of goal validations with metadata
        
        Returns:
            Prioritized list of goals
        """
        self.logger.debug(f"Prioritizing {len(validations)} validations")
        try:
            if not validations:
                self.logger.warning("No validations provided for prioritization")
                return []
            
            prioritized_goals = []
            for validation in validations:
                try:
                    goal = validation.get("goal", {})
                    if not goal:
                        self.logger.warning(f"Skipping validation with no goal: {validation}")
                        continue
                    
                    priority = goal.get("priority", 1)
                    is_valid = validation.get("is_valid", False)
                    
                    if not is_valid:
                        self.logger.debug(f"Skipping invalid goal: {goal.get('description', 'unknown')}")
                        continue
                    
                    # Calculate priority score
                    file_path = goal.get("target_module", "")
                    file_state = self.code_state_manager.get_file_state(file_path)
                    
                    impact_score = 1.0
                    if file_state:
                        metrics = file_state.get("metrics", {})
                        if goal.get("type") == "replace_print_with_logging":
                            impact_score += metrics.get("print_statements", 0) * 0.1
                        elif goal.get("type") == "optimize_string_formatting":
                            impact_score += metrics.get("string_concatenations", 0) * 0.2
                    
                    # Adjust based on past outcomes
                    past_outcomes = self.outcome_analyzer.get_outcomes_for_goal(goal)
                    success_rate = sum(1 for outcome in past_outcomes if outcome.get("status") == "success") / (len(past_outcomes) + 1)
                    priority_score = priority * impact_score * (0.5 + success_rate)
                    
                    prioritized_goal = {
                        "goal": goal,
                        "priority_score": priority_score,
                        "is_valid": is_valid,
                        "description": validation.get("description", goal.get("description", ""))
                    }
                    prioritized_goals.append(prioritized_goal)
                    self.logger.debug(f"Prioritized goal: {goal.get('description', 'unknown')} with score {priority_score}")
                
                except Exception as e:
                    self.logger.error(f"Error prioritizing goal: {str(e)} - Validation: {validation}")
                    continue
            
            # Sort by priority score
            prioritized_goals.sort(key=lambda x: x.get("priority_score", 0), reverse=True)
            self.logger.debug(f"Returning {len(prioritized_goals)} prioritized goals")
            return prioritized_goals
        
        except Exception as e:
            self.logger.error(f"Error in prioritize_goals: {str(e)}")
            raise