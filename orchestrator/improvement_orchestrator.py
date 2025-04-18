"""
Improvement Orchestrator for the Self-Improving AI Assistant.

Coordinates the execution of improvement cycles.
"""
import logging
import asyncio
from typing import Dict, Any, Optional, List
from pathlib import Path
import uuid
from datetime import datetime

from interfaces import ImprovementGoal
from observatory.code_state_manager import CodeStateManager
from orchestrator.cycle_planner import CyclePlanner
from code_updater import CodeUpdater
from outcome_repository.outcome_logger import OutcomeLogger
from outcome_repository.outcome_analyzer import OutcomeAnalyzer
from utils import setup_logging
from goal_generator import GoalGenerator

logger = setup_logging(__name__)

class ImprovementOrchestrator:
    def __init__(
        self,
        code_state_manager: CodeStateManager,
        cycle_planner: CyclePlanner,
        code_updater: CodeUpdater,
        outcome_logger: OutcomeLogger,
        outcome_analyzer: OutcomeAnalyzer,
        base_path: Optional[Path] = None
    ):
        self.code_state_manager = code_state_manager
        self.cycle_planner = cycle_planner
        self.code_updater = code_updater
        self.outcome_logger = outcome_logger
        self.outcome_analyzer = outcome_analyzer
        self.base_path = base_path or Path.cwd()
        self.logger = logger
    
    async def execute_cycle(self, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a single improvement cycle.
        
        Args:
            metadata: Optional metadata for the cycle
            
        Returns:
            Dictionary containing cycle results
        """
        cycle_id = str(uuid.uuid4())
        self.logger.info(f"Starting improvement cycle {cycle_id}")
        
        try:
            # Get transformation recommendations
            self.logger.info("Fetching transformation recommendations")
            try:
                goal_generator = GoalGenerator(base_path=self.base_path)
                recommendations = goal_generator.generate_goals(num_goals=5)
                self.logger.info(f"Received {len(recommendations)} recommendations")
                self.logger.info(f"Recommendations: {[r.to_dict() for r in recommendations]}")
                # Convert ImprovementGoal to dict for CyclePlanner
                recommendations_dict = [r.to_dict() for r in recommendations]
            except Exception as e:
                self.logger.error(f"Error fetching recommendations: {type(e).__name__} - {e}")
                raise
            
            # Plan the cycle
            self.logger.info("Planning cycle with cycle_planner.plan_cycle()")
            try:
                plan = await self.cycle_planner.plan_cycle(recommendations_dict, metadata)
                self.logger.debug(f"Cycle plan: {plan}")
                self.logger.info(f"Cycle planning completed with status: {plan.get('status', 'unknown')}")
                if plan.get("status") == "error":
                    self.logger.error(f"Cycle planning failed: {plan.get('error', 'Unknown error')}")
                    return plan
            except Exception as e:
                self.logger.error(f"Error planning cycle: {type(e).__name__} - {e}")
                raise
            
            # Execute transformations
            results = {
                "cycle_id": cycle_id,
                "timestamp": datetime.now().isoformat(),
                "goals": [],
                "transformations": [],
                "status": "success"
            }
            
            prioritized_goals = plan.get("goals", [])
            self.logger.info(f"Processing {len(prioritized_goals)} prioritized goals")
            
            for i, prioritized_goal in enumerate(prioritized_goals):
                self.logger.info(f"Processing goal {i+1} of {len(prioritized_goals)}")
                goal_dict = prioritized_goal.get("goal", {})
                self.logger.info(f"Goal details: {goal_dict}")
                try:
                    self.logger.info(f"Creating ImprovementGoal for: {goal_dict.get('description', 'unknown')}")
                    
                    # Log all parameters for debugging
                    target_module = goal_dict.get("target_module", "")
                    description = goal_dict.get("description", "")
                    improvement_type = goal_dict.get("type", "")
                    target_function = goal_dict.get("target_function")
                   

                    performance_target = goal_dict.get("performance_target")
                    priority = goal_dict.get("priority", 1)
                    
                    self.logger.info(f"ImprovementGoal parameters: target_module={target_module}, "
                                 f"description={description}, improvement_type={improvement_type}, "
                                 f"target_function={target_function}, performance_target={performance_target}, "
                                 f"priority={priority}")
                    
                    improvement_goal = ImprovementGoal(
                        target_module=target_module,
                        description=description,
                        improvement_type=improvement_type,
                        target_function=target_function,
                        performance_target=performance_target,
                        priority=priority
                    )
                    self.logger.info(f"Created ImprovementGoal: {improvement_goal.to_dict()}")
                    results["goals"].append(improvement_goal.to_dict())
                    
                    self.logger.info(f"Applying transformation {improvement_goal.improvement_type} to {improvement_goal.target_module}")
                    try:
                        transformation_result = await self.code_updater.update_codebase([improvement_goal])
                        self.logger.info(f"code_updater.update_codebase completed")
                        results["transformations"].append(transformation_result)
                        self.logger.info(f"Transformation result status: {transformation_result[0].get('status', 'unknown') if isinstance(transformation_result, list) else transformation_result.get('status', 'unknown')}")
                    except Exception as e:
                        self.logger.error(f"Error in code_updater.update_codebase: {type(e).__name__} - {e}")
                        raise
                    
                except Exception as e:
                    self.logger.error(f"Error processing goal {goal_dict.get('description', 'unknown')}: {type(e).__name__} - {e}")
                    results["transformations"].append({
                        "status": "error",
                        "message": f"Failed to process goal: {str(e)}",
                        "goal": goal_dict
                    })
                    results["status"] = "partial_success"
            
            # Log outcomes
            self.logger.info("Logging cycle outcomes")
            try:
                self.outcome_logger.log_cycle(results)
                self.logger.info(" respectively outcomes logged successfully")
            except Exception as e:
                self.logger.error(f"Error logging outcomes: {type(e).__name__} - {e}")
                # Continue despite error
            
            # Finalize plan
            self.logger.info("Finalizing plan with cycle_planner.finalize_plan()")
            try:
                self.cycle_planner.finalize_plan(results.get("status", "success"), results)
                self.logger.info("Plan finalized successfully")
            except Exception as e:
                self.logger.error(f"Error finalizing plan: {type(e).__name__} - {e}")
                # Continue despite error
            
            self.logger.info(f"Cycle {cycle_id} completed with status {results['status']}")
            return results
        
        except Exception as e:
            self.logger.error(f"Cycle {cycle_id} failed: {type(e).__name__} - {e}")
            self.logger.error(f"Exception details: {e}")
            results = {
                "cycle_id": cycle_id,
                "timestamp": datetime.now().isoformat(),
                "goals": [],
                "transformations": [],
                "status": "error",
                "error": str(e)
            }
            
            try:
                self.logger.info("Attempting to log error cycle")
                self.outcome_logger.log_cycle(results)
                self.logger.info("Error cycle logged successfully")
            except Exception as log_err:
                self.logger.error(f"Failed to log error cycle: {type(log_err).__name__} - {log_err}")
            
            try:
                self.logger.info("Attempting to finalize error plan")
                self.cycle_planner.finalize_plan("error", results)
                self.logger.info("Error plan finalized successfully")
            except Exception as plan_err:
                self.logger.error(f"Failed to finalize error plan: {type(plan_err).__name__} - {plan_err}")
                
            return results