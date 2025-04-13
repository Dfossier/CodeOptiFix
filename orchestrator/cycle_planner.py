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
            recommendations: List of recommended goals with validations
            metadata: Optional metadata for the cycle
            
        Returns:
            Dictionary containing the cycle plan
        """
        cycle_id = str(uuid.uuid4())
        self.logger.debug(f"Planning cycle {cycle_id} with {len(recommendations)} recommendations")
        
        try:
            # Normalize recommendations
            normalized_validations = []
            for rec in recommendations:
                try:
                    goal = rec.get("goal", {})
                    validation = rec.get("validation", {})
                    normalized = {
                        "goal": {
                            "target_module": goal.get("target_module", ""),
                            "description": goal.get("description", ""),
                            "type": goal.get("type", ""),
                            "priority": goal.get("priority", 1),
                            "target_function": goal.get("target_function"),
                            "performance_target": goal.get("performance_target")
                        },
                        "is_valid": validation.get("is_valid", False),
                        "reason": validation.get("reason", ""),
                        "description": validation.get("description", goal.get("description", ""))
                    }
                    normalized_validations.append(normalized)
                except Exception as e:
                    self.logger.error(f"Error normalizing recommendation: {str(e)} - Data: {rec}")
                    continue
            
            self.logger.debug(f"Normalized {len(normalized_validations)} validations")
            
            # Validate goals
            validated_goals = []
            for validation in normalized_validations:
                try:
                    goal = validation.get("goal", {})
                    self.logger.debug(f"Validating goal: {goal.get('description', 'unknown')}")
                    validated = self.goal_validator.validate_goal(goal)
                    validated["goal"] = goal
                    validated["description"] = goal.get("description", "")
                    validated_goals.append(validated)
                except Exception as e:
                    self.logger.error(f"Error validating goal: {str(e)} - Goal: {goal}")
                    continue
            
            self.logger.debug(f"Validated {len(validated_goals)} goals")
            
            # Refine goals
            self.logger.debug("Calling goal_refiner.refine_goals")
            refined_validations = self.goal_refiner.refine_goals(validated_goals)
            self.logger.debug(f"Refined {len(refined_validations)} validations")
            
            # Prioritize goals
            self.logger.debug("Awaiting goal_prioritizer.prioritize_goals")
            prioritized_goals = await self.goal_prioritizer.prioritize_goals(refined_validations)
            self.logger.debug(f"Prioritized {len(prioritized_goals)} goals")
            
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