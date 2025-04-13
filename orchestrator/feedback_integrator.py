"""
Feedback Integrator for Feedback-Driven Improvement Orchestrator.

Incorporates outcome data into planning and execution.
"""
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import logging

from utils import setup_logging
from interfaces import ImprovementGoal
from outcome_repository.outcome_analyzer import OutcomeAnalyzer
from code_updater import CodeUpdater

logger = setup_logging(__name__)

class FeedbackIntegrator:
    """
    Integrates feedback data into the transformation process.
    
    Enhances the transformation process with insights from past outcomes,
    making transformations more adaptive to the current codebase state.
    """
    
    def __init__(self, outcome_analyzer: OutcomeAnalyzer):
        """
        Initialize the feedback integrator.
        
        Args:
            outcome_analyzer: Instance of OutcomeAnalyzer
        """
        self.outcome_analyzer = outcome_analyzer
        self.logger = logger
    
    def enhance_transformer_config(self, transformation_type: str) -> Dict[str, Any]:
        """
        Enhance transformer configuration with feedback data.
        
        Args:
            transformation_type: Type of transformation
            
        Returns:
            Enhanced configuration for the transformer
        """
        # Get historical data for this transformation type
        success_rate = self.outcome_analyzer.get_transformation_success_rate(transformation_type)
        common_failures = self.outcome_analyzer.get_common_failure_reasons(transformation_type)
        
        # Create base configuration
        config = {
            "transformation_type": transformation_type,
            "historical_success_rate": success_rate,
            "common_failures": common_failures
        }
        
        # Add transformation-specific enhancements
        if transformation_type == "replace_print_with_logging":
            # If this transformation has failed before due to certain patterns,
            # add configuration to handle those patterns
            for failure in common_failures:
                if "no print statements" in failure.get("reason", "").lower():
                    config["enhance_existing_logging"] = True
                    config["fallback_to_basic_logging"] = True
                    self.logger.info("Enhanced replace_print_with_logging with fallbacks based on historical failures")
                    break
                    
        elif transformation_type in ["add_structured_logging", "add_structured_logging_conditional", "add_structured_logging_error"]:
            # Configure structured logging based on past failures
            for failure in common_failures:
                if "no logger calls" in failure.get("reason", "").lower():
                    config["create_logger_if_missing"] = True
                    self.logger.info("Enhanced structured_logging with logger creation based on historical failures")
                    break
                    
        elif transformation_type == "add_exception_handling":
            # Configure exception handling based on past failures
            for failure in common_failures:
                if "already has exception handling" in failure.get("reason", "").lower():
                    config["enhance_existing_handling"] = True
                    self.logger.info("Enhanced exception_handling to improve existing handlers")
                    break
        
        return config
    
    def adapt_code_updater(self, updater: CodeUpdater) -> None:
        """
        Adapt the CodeUpdater with feedback from past outcomes.
        
        Args:
            updater: Instance of CodeUpdater to adapt
        """
        # Get overall transformation statistics
        transformation_stats = self.outcome_analyzer.get_transformation_stats()
        
        # Adapt validation rules based on historical failures
        validation_failures = [
            stat for _, stat in transformation_stats.items()
            if any("validation failed" in failure.get("reason", "").lower() 
                  for failure in stat.get("common_failures", []))
        ]
        
        if validation_failures:
            self.logger.info("Adapting validation rules based on historical failures")
            
            # Modify validation rules to be more lenient if validation failures are common
            updater.static_checkers = self._adjust_static_checkers(updater.static_checkers)
        
        # Adapt error handling based on common errors
        # (This would update the CodeUpdater's error handling behavior)
        
        # Adapt the way the updater chooses transformers
        trend_analysis = self.outcome_analyzer.get_trend_analysis()
        
        if trend_analysis.get("declining", False):
            self.logger.info("Success rate declining, adapting transformer selection")
            
            # If success rates are declining, use more conservative transformation selection
            # This is a placeholder for more sophisticated adaptation
    
    def _adjust_static_checkers(self, current_checkers: List[str]) -> List[str]:
        """Adjust static checkers based on historical outcomes."""
        # This is a simple example - in a real implementation, we would
        # analyze specific validation failures to determine which checkers
        # to modify or adjust
        
        # For now, just return the current checkers
        return current_checkers
    
    def suggest_goal_modifications(self, goals: List[ImprovementGoal]) -> List[Dict[str, Any]]:
        """
        Suggest modifications to goals based on feedback data.
        
        Args:
            goals: List of improvement goals
            
        Returns:
            List of suggestions for each goal
        """
        suggestions = []
        
        for goal in goals:
            # Determine the transformation type from the goal
            description = goal.description.lower()
            transformation_type = None
            
            if "print" in description and "log" in description:
                transformation_type = "replace_print_with_logging"
            elif "structured logging" in description:
                if "conditional" in description:
                    transformation_type = "add_structured_logging_conditional"
                elif "error" in description:
                    transformation_type = "add_structured_logging_error"
                else:
                    transformation_type = "add_structured_logging"
            elif "exception" in description or "error handling" in description:
                transformation_type = "add_exception_handling"
            elif "extract" in description:
                transformation_type = "extract_function"
            elif "split" in description and ("file" in description or "module" in description):
                transformation_type = "split_file"
            
            if not transformation_type:
                suggestions.append({
                    "goal": goal.to_dict(),
                    "has_suggestions": False,
                    "reason": "Unknown transformation type"
                })
                continue
            
            # Get success rate and common failures
            success_rate = self.outcome_analyzer.get_transformation_success_rate(transformation_type)
            module_success_rate = self.outcome_analyzer.get_module_success_rate(
                goal.target_module, transformation_type
            )
            
            common_failures = self.outcome_analyzer.get_common_failure_reasons(transformation_type)
            
            # Generate suggestions based on historical data
            goal_suggestions = []
            
            if success_rate < 0.5 and module_success_rate < 0.5:
                # Both the transformation type and the module have low success rates
                goal_suggestions.append({
                    "type": "reconsider",
                    "message": f"Consider alternative goal, as {transformation_type} has low success rate ({success_rate:.0%}) for this module ({module_success_rate:.0%})"
                })
            elif success_rate < 0.5:
                # The transformation type has a low success rate overall
                goal_suggestions.append({
                    "type": "caution",
                    "message": f"Exercise caution, as {transformation_type} has low overall success rate ({success_rate:.0%})"
                })
            elif module_success_rate < 0.5:
                # The module has a low success rate for this transformation
                goal_suggestions.append({
                    "type": "caution",
                    "message": f"Exercise caution, as this module has low success rate ({module_success_rate:.0%}) with {transformation_type}"
                })
            
            # Add suggestions based on common failures
            for failure in common_failures:
                reason = failure.get("reason", "").lower()
                
                if "no print statements" in reason and transformation_type == "replace_print_with_logging":
                    goal_suggestions.append({
                        "type": "refine",
                        "message": "Consider refining to 'Enhance existing logging' as print statements may not be present"
                    })
                elif "already has structured logging" in reason and "structured_logging" in transformation_type:
                    goal_suggestions.append({
                        "type": "refine",
                        "message": "Consider refining to 'Enhance structured logging' as basic structured logging may already exist"
                    })
            
            suggestions.append({
                "goal": goal.to_dict(),
                "has_suggestions": len(goal_suggestions) > 0,
                "suggestions": goal_suggestions,
                "historical_data": {
                    "transformation_type": transformation_type,
                    "success_rate": success_rate,
                    "module_success_rate": module_success_rate,
                    "common_failures": common_failures
                }
            })
        
        return suggestions