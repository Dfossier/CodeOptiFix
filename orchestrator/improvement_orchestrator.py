"""
Improvement Orchestrator for the Self-Improving AI Assistant.

Coordinates improvement cycles by integrating goal generation, validation,
prioritization, code updates, and outcome analysis.
"""
import logging
from typing import Dict, List, Any, Optional
import asyncio
from datetime import datetime
from pathlib import Path
import uuid

from orchestrator.cycle_planner import CyclePlanner
from goal_intelligence.goal_validator import GoalValidator
from goal_intelligence.goal_refiner import GoalRefiner
from goal_intelligence.goal_prioritizer import GoalPrioritizer
from code_updater import CodeUpdater
from outcome_repository.outcome_analyzer import OutcomeAnalyzer
from outcome_repository.outcome_logger import OutcomeLogger
from utils import setup_logging

logger = setup_logging(__name__)
logger.info("Loaded improvement_orchestrator.py version: 2025-04-12-fix-refine-goal")

class ImprovementOrchestrator:
    def __init__(self, cycle_planner: CyclePlanner):
        self.cycle_planner = cycle_planner
        self.goal_validator = GoalValidator(
            code_state_manager=cycle_planner.code_state_manager
        )
        self.goal_refiner = GoalRefiner(
            code_state_manager=cycle_planner.code_state_manager
        )
        self.goal_prioritizer = GoalPrioritizer(
            code_state_manager=cycle_planner.code_state_manager,
            outcome_analyzer=cycle_planner.outcome_analyzer
        )
        self.code_updater = CodeUpdater(base_path=cycle_planner.base_path)
        self.outcome_logger = OutcomeLogger(base_path=cycle_planner.base_path)
        self.outcome_analyzer = cycle_planner.outcome_analyzer
        self.logger = logger
    
    async def recommend_goals(self, goals_file: Optional[str] = None) -> List[Dict[str, Any]]:
        """Generate and validate improvement goals."""
        self.logger.debug("Entering recommend_goals")
        try:
            recommendations = []
            if goals_file:
                self.logger.debug(f"Loading goals from file: {goals_file}")
                import json
                with open(goals_file, 'r', encoding='utf-8') as f:
                    recommendations = json.load(f)
            else:
                self.logger.debug("Awaiting code_state_manager.recommend_transformations")
                recommendations = await self.cycle_planner.code_state_manager.recommend_transformations()
                self.logger.debug(f"Received {len(recommendations)} recommendations")
            
            # Validate and refine goals
            validated_goals = []
            for goal in recommendations:
                try:
                    # Ensure goal is a dict and has required fields
                    if not isinstance(goal, dict):
                        self.logger.error(f"Invalid goal format, expected dict, got: {type(goal)} - Data: {goal}")
                        continue
                    goal_dict = goal.get("goal", goal)  # Handle nested goal
                    if not isinstance(goal_dict, dict) or not goal_dict.get("target_module"):
                        self.logger.error(f"Malformed goal, missing target_module: {goal_dict}")
                        continue
                    self.logger.debug(f"Validating goal: {goal_dict.get('description', 'unknown')}")
                    validation = self.goal_validator.validate_goal(goal_dict)
                    self.logger.debug(f"Validation result: {validation}")
                    if validation.get("is_valid", False):
                        self.logger.debug(f"Refining goal: {goal_dict.get('description', 'unknown')}")
                        # Create goal_validation dict for refine_goal
                        goal_validation = {
                            "goal": goal_dict,
                            "is_valid": validation.get("is_valid", False),
                            "reason": validation.get("reason", ""),
                            "description": validation.get("description", goal_dict.get("description", ""))
                        }
                        refined_validation = self.goal_refiner.refine_goal(goal_validation)
                        self.logger.debug(f"Refinement result: {refined_validation}")
                        if refined_validation.get("is_valid", False):
                            validated_goals.append({
                                "goal": refined_validation.get("goal", goal_dict),
                                "validation": refined_validation
                            })
                        else:
                            self.logger.debug(f"Goal refinement failed: {refined_validation.get('reason', 'unknown')}")
                    else:
                        self.logger.debug(f"Goal invalid: {goal_dict.get('description', 'unknown')} - Reason: {validation.get('reason', 'unknown')}")
                except Exception as e:
                    self.logger.error(f"Error processing goal: {str(e)} - Goal data: {goal}")
                    continue
            
            self.logger.debug(f"Returning {len(validated_goals)} validated goals")
            return validated_goals
        except Exception as e:
            self.logger.error(f"Error recommending goals: {str(e)}")
            return []
        finally:
            self.logger.debug("Exiting recommend_goals")
    
    async def execute_cycle(self, recommendations: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a single improvement cycle.
        
        Args:
            recommendations: List of recommended goals with validations
            metadata: Optional metadata about the cycle
            
        Returns:
            Dictionary summarizing cycle results
        """
        self.logger.debug("Entering execute_cycle")
        cycle_id = str(uuid.uuid4())
        start_time = datetime.now()
        self.logger.info(f"Starting improvement cycle {cycle_id}")
        
        try:
            # Plan the cycle
            self.logger.debug("Awaiting cycle_planner.plan_cycle")
            plan = await self.cycle_planner.plan_cycle(recommendations, metadata)
            self.logger.debug(f"Got plan with {len(plan.get('goals', []))} goals")
            prioritized_goals = plan.get("goals", [])
            
            # Prepare improvement goals for code updater
            improvement_goals = []
            for entry in prioritized_goals:
                goal_dict = entry.get("goal", {})
                validation = entry.get("validation", {})
                try:
                    from interfaces import ImprovementGoal
                    improvement_goal = ImprovementGoal(
                        target_module=goal_dict.get("target_module", ""),
                        description=goal_dict.get("description", ""),
                        priority=goal_dict.get("priority", 1),
                        type=goal_dict.get("type", ""),
                        target_function=goal_dict.get("target_function"),
                        performance_target=goal_dict.get("performance_target")
                    )
                    improvement_goals.append(improvement_goal)
                    self.logger.debug(f"Created ImprovementGoal: {goal_dict.get('description', 'unknown')}")
                except Exception as e:
                    self.logger.error(f"Error creating ImprovementGoal: {str(e)}")
                    continue
            
            # Execute code updates
            transformations = []
            if improvement_goals:
                self.logger.debug(f"Awaiting code_updater.update_codebase with {len(improvement_goals)} goals")
                try:
                    transformation_results = await self.code_updater.update_codebase(improvement_goals)
                    transformations = transformation_results
                    self.logger.debug(f"Got {len(transformations)} transformation results")
                except Exception as e:
                    self.logger.error(f"Error applying transformations: {str(e)}")
                    transformations.append({
                        "status": "error",
                        "message": str(e),
                        "goal": {"description": "Code update failure"}
                    })
            else:
                self.logger.warning("No valid goals to process")
            
            # Log outcomes
            cycle_data = {
                "cycle_id": cycle_id,
                "start_time": start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "goals": [g.to_dict() for g in improvement_goals],
                "transformations": transformations,
                "status": "success" if transformations and any(t.get("status") == "success" for t in transformations) else "no_changes",
                "metadata": metadata or {}
            }
            
            try:
                self.logger.debug("Calling outcome_logger.log_cycle")
                self.outcome_logger.log_cycle(cycle_data)
                self.logger.debug("Calling outcome_analyzer.refresh_cache")
                self.outcome_analyzer.refresh_cache()
            except Exception as e:
                self.logger.error(f"Error logging cycle: {str(e)}")
            
            # Update plan status
            status = "success" if cycle_data["status"] == "success" else "error"
            self.logger.debug(f"Finalizing plan with status: {status}")
            self.cycle_planner.finalize_plan(status, cycle_data)
            
            # Prepare results
            results = {
                "success": cycle_data["status"] == "success",
                "cycle_id": cycle_id,
                "goals_processed": len(improvement_goals),
                "transformations": transformations,
                "metadata": metadata or {}
            }
            
            self.logger.debug(f"Completed improvement cycle {cycle_id}: {results['goals_processed']} goals processed")
            return results
        
        except Exception as e:
            self.logger.error(f"Cycle {cycle_id} failed: {str(e)}")
            error_result = {
                "success": False,
                "cycle_id": cycle_id,
                "goals_processed": 0,
                "transformations": [{
                    "status": "error",
                    "message": str(e),
                    "goal": {"description": "Cycle execution failure"}
                }],
                "metadata": metadata or {}
            }
            try:
                self.outcome_logger.log_cycle({
                    "cycle_id": cycle_id,
                    "start_time": start_time.isoformat(),
                    "end_time": datetime.now().isoformat(),
                    "goals": [],
                    "transformations": error_result["transformations"],
                    "status": "error",
                    "metadata": metadata or {}
                })
                self.cycle_planner.finalize_plan("error", error_result)
            except Exception as log_e:
                self.logger.error(f"Error logging failed cycle: {str(log_e)}")
            return error_result
        finally:
            self.logger.debug("Exiting execute_cycle")