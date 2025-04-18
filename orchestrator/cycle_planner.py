"""
Cycle Planner for the Self-Improving AI Assistant.

Plans improvement cycles by validating, refining, and prioritizing goals.
"""
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import asyncio
from datetime import datetime
import uuid

from observatory.code_state_manager import CodeStateManager
from outcome_repository.outcome_analyzer import OutcomeAnalyzer
from outcome_repository.outcome_logger import OutcomeLogger
from goal_intelligence.goal_validator import GoalValidator
from goal_intelligence.goal_refiner import GoalRefiner
from goal_intelligence.goal_prioritizer import GoalPrioritizer
from utils import setup_logging

logger = setup_logging(__name__)

class CyclePlanner:
    def __init__(
        self,
        code_state_manager: CodeStateManager,
        outcome_analyzer: OutcomeAnalyzer,
        outcome_logger: OutcomeLogger,
        base_path: Optional[Path] = None
    ):
        self.code_state_manager = code_state_manager
        self.outcome_analyzer = outcome_analyzer
        self.outcome_logger = outcome_logger
        self.base_path = base_path or Path.cwd()
        self.goal_validator = GoalValidator(
            code_state_manager=code_state_manager
        )
        self.goal_refiner = GoalRefiner(
            code_state_manager=code_state_manager
        )
        self.goal_prioritizer = GoalPrioritizer(
            code_state_manager=code_state_manager,
            outcome_analyzer=outcome_analyzer
        )
        self.logger = logger
    
    async def plan_cycle(self, recommendations: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Plan a single improvement cycle.
        
        Args:
            recommendations: List of recommended goals (dictionaries with target_module, description, etc.)
            metadata: Optional metadata for the cycle
            
        Returns:
            Dictionary containing the cycle plan
        """
        cycle_id = str(uuid.uuid4())
        self.logger.debug(f"Planning cycle {cycle_id} with {len(recommendations)} recommendations")
        
        try:
            # Log incoming recommendations for debugging
            self.logger.debug(f"Received recommendations: {recommendations}")
            
            # Validate input recommendations
            validated_goals = []
            for goal in recommendations:
                if not isinstance(goal, dict) or "target_module" not in goal or "description" not in goal:
                    self.logger.error(f"Invalid recommendation format, skipping: {goal}")
                    continue
                try:
                    self.logger.debug(f"Validating goal: {goal}")
                    validation_result = self.goal_validator.validate_goal(goal)
                    if validation_result.get("is_valid"):
                        # Create a new dictionary to avoid modifying the original
                        validated_goal = {
                            "target_module": str(goal["target_module"]),
                            "description": str(goal["description"]),
                            "type": goal.get("type", ""),
                            "improvement_type": goal.get("improvement_type", goal.get("type", "")),
                            "target_function": goal.get("target_function"),
                            "performance_target": goal.get("performance_target"),
                            "priority": goal.get("priority", 1)
                        }
                        validated_goals.append(validated_goal)
                        self.logger.debug(f"Validated goal: {validated_goal}")
                    else:
                        self.logger.warning(f"Goal invalid: {goal.get('description', 'unknown')} - {validation_result.get('error', 'Unknown error')}")
                except Exception as e:
                    self.logger.error(f"Error validating goal: {str(e)} - Goal: {goal}")
                    continue
            
            self.logger.debug(f"Validated {len(validated_goals)} goals: {validated_goals}")
            
            # Refine goals
            self.logger.debug("Calling goal_refiner.refine_goals")
            refined_validations = self.goal_refiner.refine_goals(validated_goals)
            self.logger.debug(f"Refined {len(refined_validations)} validations: {refined_validations}")
            
            # Prioritize goals
            self.logger.debug("Awaiting goal_prioritizer.prioritize_goals")
            prioritized_goals = await self.goal_prioritizer.prioritize_goals(refined_validations)
            self.logger.debug(f"Prioritized {len(prioritized_goals)} goals: {prioritized_goals}")
            
            # Prepare plan
            plan = {
                "cycle_id": cycle_id,
                "timestamp": datetime.now().isoformat(),
                "goals": prioritized_goals,
                "metadata": metadata or {},
                "status": "planned"
            }
            
            self.logger.debug(f"Cycle {cycle_id} planned successfully")
            return plan
        
        except Exception as e:
            self.logger.error(f"Error planning cycle {cycle_id}: {str(e)}")
            return {
                "cycle_id": cycle_id,
                "timestamp": datetime.now().isoformat(),
                "goals": [],
                "metadata": metadata or {},
                "status": "error",
                "error": str(e)
            }
    
    def finalize_plan(self, status: str, results: Dict[str, Any]):
        """
        Finalize the cycle plan with results.
        
        Args:
            status: Final status of the cycle (e.g., success, error)
            results: Results of the cycle execution
        """
        self.logger.debug(f"Finalizing plan with status: {status}")
        try:
            cycle_id = results.get("cycle_id", "unknown")
            plan_data = {
                "cycle_id": cycle_id,
                "timestamp": datetime.now().isoformat(),
                "status": status,
                "results": results
            }
            self.outcome_logger.log_cycle(plan_data)
            self.logger.debug(f"Plan {cycle_id} finalized")
        except Exception as e:
            self.logger.error(f"Error finalizing plan: {str(e)}")