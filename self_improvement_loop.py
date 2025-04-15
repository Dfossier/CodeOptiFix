"""
Self-Improvement Loop for the AI Assistant.

Main entry point to run improvement cycles.
"""
import asyncio
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

try:
    from observatory.code_state_manager import CodeStateManager
    from orchestrator.cycle_planner import CyclePlanner
    from orchestrator.improvement_orchestrator import ImprovementOrchestrator
    from code_updater import CodeUpdater
    from outcome_repository.outcome_logger import OutcomeLogger
    from outcome_repository.outcome_analyzer import OutcomeAnalyzer
    from goal_intelligence.goal_validator import GoalValidator
    from goal_intelligence.goal_prioritizer import GoalPrioritizer
    from goal_intelligence.goal_refiner import GoalRefiner
    from utils import setup_logging
except ImportError as e:
    print(f"Import error: {e}")
    raise

logger = setup_logging(__name__)

async def run_improvement_cycle(orchestrator: ImprovementOrchestrator, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run a single improvement cycle."""
    logger.debug("Starting single improvement cycle")
    try:
        logger.info("Calling orchestrator.execute_cycle")
        result = await orchestrator.execute_cycle(metadata)
        logger.info(f"Orchestrator returned result with status: {result.get('status', 'unknown')}")
        logger.debug("Completed single improvement cycle")
        return result
    except Exception as e:
        logger.error(f"Cycle failed: {str(e)}")
        logger.error(f"Exception details: {type(e).__name__} - {e}")
        return {"status": "error", "error": str(e), "goals": [], "transformations": []}

async def main(continuous: bool = False, interval: int = 3600) -> None:
    """Main function to initialize and run improvement cycles."""
    logger.debug("Entering main function")
    try:
        # Initialize dependencies
        base_path = Path.cwd()
        logger.info("Using base path: %s", base_path)
        
        logger.info("Initializing CodeStateManager")
        code_state_manager = CodeStateManager(base_path=base_path)
        logger.info("CodeStateManager initialized successfully")
        
        logger.info("Initializing OutcomeLogger")
        outcome_logger = OutcomeLogger(base_path=base_path / "outcome_repository" / "data")
        logger.info("OutcomeLogger initialized successfully")
        
        logger.info("Initializing OutcomeAnalyzer")
        outcome_analyzer = OutcomeAnalyzer(base_path=base_path / "outcome_repository" / "data")
        logger.info("OutcomeAnalyzer initialized successfully")
        
        logger.info("Initializing GoalValidator")
        goal_validator = GoalValidator(code_state_manager=code_state_manager)
        logger.info("GoalValidator initialized successfully")
        
        logger.info("Initializing GoalRefiner")
        goal_refiner = GoalRefiner(code_state_manager=code_state_manager)
        logger.info("GoalRefiner initialized successfully")
        
        logger.info("Initializing GoalPrioritizer")
        goal_prioritizer = GoalPrioritizer(code_state_manager=code_state_manager, outcome_analyzer=outcome_analyzer)
        logger.info("GoalPrioritizer initialized successfully")
        
        logger.info("Initializing CyclePlanner")
        cycle_planner = CyclePlanner(
            code_state_manager=code_state_manager,
            outcome_analyzer=outcome_analyzer,
            outcome_logger=outcome_logger,
            base_path=base_path
        )
        logger.info("CyclePlanner initialized successfully")
        
        logger.info("Initializing CodeUpdater")
        code_updater = CodeUpdater(base_path=base_path)
        logger.info("CodeUpdater initialized successfully")
        
        logger.info("Initializing ImprovementOrchestrator")
        orchestrator = ImprovementOrchestrator(
            code_state_manager=code_state_manager,
            cycle_planner=cycle_planner,
            code_updater=code_updater,
            outcome_logger=outcome_logger,
            outcome_analyzer=outcome_analyzer,
            base_path=base_path
        )
        logger.info("ImprovementOrchestrator initialized successfully")
        
        cycle_count = 0
        while True:
            cycle_count += 1
            logger.info(f"===== Starting improvement cycle {cycle_count} =====")
            try:
                logger.info("About to call run_improvement_cycle()")
                result = await run_improvement_cycle(orchestrator, metadata={"cycle_number": cycle_count})
                logger.info("run_improvement_cycle() completed")
                
                status = result.get("status", "unknown")
                goal_count = len(result.get("goals", []))
                logger.info(f"Cycle {cycle_count} completed: {status}, {goal_count} goals")
                
                if not continuous:
                    logger.info("Continuous mode disabled, exiting after one cycle")
                    break
                
                logger.info(f"Waiting {interval} seconds before next cycle")
                await asyncio.sleep(interval)
                logger.info(f"Wait completed, continuing to next cycle")
            except Exception as e:
                logger.error(f"Error in cycle {cycle_count}: {type(e).__name__} - {e}")
                if not continuous:
                    raise
                logger.info(f"Continuing despite error due to continuous mode")
                await asyncio.sleep(interval)
            
    except Exception as e:
        logger.error(f"Initialization error: {str(e)}")
        raise

if __name__ == "__main__":
    logger.info("Script starting")
    parser = argparse.ArgumentParser(description="Self-Improving AI Assistant")
    parser.add_argument("--continuous", action="store_true", help="Run continuously")
    parser.add_argument("--interval", type=int, default=3600, help="Interval between cycles in seconds")
    args = parser.parse_args()
    
    try:
        asyncio.run(main(continuous=args.continuous, interval=args.interval))
    except KeyboardInterrupt:
        logger.info("Script interrupted by user")
    except Exception as e:
        logger.error(f"Script failed: {str(e)}")
        raise