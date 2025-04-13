"""
Self-improvement loop for code optimization.

Runs single or continuous improvement cycles using ImprovementOrchestrator.
"""
import argparse
import asyncio
import logging
from datetime import datetime
from pathlib import Path
import sys
import json
from typing import Optional, Dict, List, Any
from dotenv import load_dotenv
from orchestrator.improvement_orchestrator import ImprovementOrchestrator
from orchestrator.cycle_planner import CyclePlanner
from observatory.code_state_manager import CodeStateManager
from outcome_repository.outcome_analyzer import OutcomeAnalyzer
from outcome_repository.outcome_logger import OutcomeLogger
from utils import setup_logging

logger = setup_logging(__name__)

def parse_args():
    logger.debug("Parsing command-line arguments")
    parser = argparse.ArgumentParser(description='Self-improvement loop for code optimization')
    parser.add_argument('--continuous', action='store_true', help='Run in continuous mode with periodic improvement cycles')
    parser.add_argument('--interval', type=int, default=3600, help='Interval between cycles in seconds (default: 3600)')
    parser.add_argument('--goals-file', type=str, help='Path to custom goals JSON file')
    parser.add_argument('--text-goal', type=str, help='Path to a text file containing a natural language goal description')
    parser.add_argument('--goal-id', type=str, help='ID of a previously processed text goal to use')
    return parser.parse_args()

async def run_single_cycle(
    orchestrator: ImprovementOrchestrator,
    goals_file: Optional[str] = None,
    text_goal: Optional[str] = None,
    goal_id: Optional[str] = None
):
    """
    Execute a single improvement cycle.
    
    Args:
        orchestrator: ImprovementOrchestrator instance
        goals_file: Optional path to a JSON goals file
        text_goal: Optional path to a text file with a natural language goal description
        goal_id: Optional ID of a previously processed text goal to use
        
    Returns:
        Results of the cycle execution
    """
    logger.info('Starting single improvement cycle')
    print(f"{datetime.now().isoformat()} - Starting single improvement cycle")
    
    # Process text goal if provided
    if text_goal or goal_id:
        from goal_intelligence.goal_processor import GoalProcessor
        processor = GoalProcessor()
        
        if text_goal:
            logger.info(f'Processing text goal file: {text_goal}')
            goal_record = await processor.process_text_goal(text_goal)
            goal_id = goal_record['goal_id']
        
        # Update goal status to indicate an attempt is being made
        if goal_id:
            logger.info(f'Using goal with ID: {goal_id}')
            processor.update_goal_status(
                goal_id=goal_id,
                status="attempt",
                logs=[f"Starting improvement cycle at {datetime.now().isoformat()}"]
            )
            
            # Load goal from records
            goal_files = list(processor.goals_dir.glob(f"goal_{goal_id}_*.json"))
            if not goal_files:
                logger.error(f"No goal record found for goal ID: {goal_id}")
                return {"success": False, "error": f"No goal record found for goal ID: {goal_id}"}
                
            with open(goal_files[0], 'r', encoding='utf-8') as f:
                goal_record = json.load(f)
                
            # Convert to improvement goals
            custom_goals = processor.create_improvement_goals(goal_record)
            logger.info(f'Created {len(custom_goals)} improvement goals from text goal')
            
            # Use these goals directly instead of getting recommendations
            results = await orchestrator.execute_cycle(
                goals=[goal.to_dict() for goal in custom_goals],
                metadata={"goal_id": goal_id, "source": "text_goal"}
            )
            
            # Update goal status based on results
            status = "success" if results.get("success", False) else "error"
            processor.update_goal_status(
                goal_id=goal_id,
                status=status,
                logs=[f"Improvement cycle completed with status: {status}"]
            )
            
            # Update sub-goal statuses
            transformations = results.get("transformations", [])
            for i, sub_goal in enumerate(goal_record.get("structured_goals", [])):
                if i < len(transformations):
                    transform_status = transformations[i].get("status", "unknown")
                    sub_goal_status = "completed" if transform_status == "success" else "failed"
                    processor.update_goal_status(
                        goal_id=goal_id,
                        sub_goal_id=sub_goal.get("sub_goal_id"),
                        status=sub_goal_status,
                        logs=[f"Transformation status: {transform_status}", 
                              f"Message: {transformations[i].get('message', 'No message')}"]
                    )
            
            logger.info('Completed single improvement cycle with text goal')
            print(f"{datetime.now().isoformat()} - Completed single improvement cycle")
            return results
    
    # If no text goal or goal ID was provided, use the standard goal recommendation flow
    logger.debug("Calling recommend_goals")
    recommendations = await orchestrator.recommend_goals(goals_file)
    logger.debug(f"Got {len(recommendations)} recommendations")
    results = await orchestrator.execute_cycle(recommendations)
    logger.info('Completed single improvement cycle')
    print(f"{datetime.now().isoformat()} - Completed single improvement cycle")
    return results

async def run_continuous_cycles(args, orchestrator: ImprovementOrchestrator):
    """Execute continuous improvement cycles with specified interval."""
    cycle_count = 0
    
    # If using a text goal with continuous mode, process it once at the beginning
    text_goal_id = None
    if args.text_goal:
        from goal_intelligence.goal_processor import GoalProcessor
        processor = GoalProcessor()
        logger.info(f'Processing text goal file for continuous mode: {args.text_goal}')
        goal_record = await processor.process_text_goal(args.text_goal)
        text_goal_id = goal_record['goal_id']
        logger.info(f'Using text goal ID {text_goal_id} for continuous cycles')
    
    # If a goal ID was provided directly, use that
    if args.goal_id:
        text_goal_id = args.goal_id
        logger.info(f'Using provided goal ID {text_goal_id} for continuous cycles')
    
    while True:
        cycle_count += 1
        logger.info(f'Starting improvement cycle {cycle_count}')
        print(f"{datetime.now().isoformat()} - Starting improvement cycle {cycle_count}")
        try:
            # Run the cycle with the appropriate goal source
            results = await run_single_cycle(
                orchestrator, 
                goals_file=args.goals_file,
                goal_id=text_goal_id
            )
            
            # Process results
            status = 'success' if results.get('success', False) else 'error'
            goals_processed = results.get('goals_processed', 0)
            successful_transformations = len([t for t in results.get('transformations', []) if t.get('status') == 'success'])
            failed_transformations = [t for t in results.get('transformations', []) if t.get('status') != 'success']
            
            # Log summary
            logger.info(f'\nCycle {cycle_count} Summary:')
            logger.info(f'Status: {status}')
            logger.info(f'Goals processed: {goals_processed}')
            logger.info(f'Successful transformations: {successful_transformations}')
            if failed_transformations:
                logger.info(f'Failed transformations: {failed_transformations}')
            else:
                logger.info('Failed transformations: None')
            print(f"{datetime.now().isoformat()} - Cycle {cycle_count} completed: {status}, {goals_processed} goals")
                
            # If using a text goal, analyze the error logs for insights
            if text_goal_id and failed_transformations:
                from goal_intelligence.goal_processor import GoalProcessor
                processor = GoalProcessor()
                insights = processor.analyze_error_logs(text_goal_id)
                if insights:
                    logger.info(f'Extracted {len(insights)} insights from error logs')
                    for i, insight in enumerate(insights):
                        logger.info(f'Insight {i+1}: {insight.get("insight", "No insight")}')
            
            if not args.continuous:
                break
                
            logger.info(f'Waiting {args.interval} seconds until next cycle...')
            print(f"{datetime.now().isoformat()} - Waiting {args.interval} seconds")
            await asyncio.sleep(args.interval)
            
        except Exception as e:
            logger.error(f'Error in continuous cycles: {str(e)}')
            print(f"{datetime.now().isoformat()} - Error in cycle {cycle_count}: {str(e)}")
            raise

async def main_async():
    """Async implementation of the main entry point."""
    logger.debug("Entering main function")
    print(f"{datetime.now().isoformat()} - Entering main function")
    env_path = Path.cwd() / '.env'
    logger.debug(f"Loading environment variables from {env_path}")
    load_dotenv(env_path)
    logger.info(f'Loaded environment variables from {env_path}')
    
    logger.debug("Parsing arguments")
    args = parse_args()
    logger.debug(f"Arguments parsed: continuous={args.continuous}, interval={args.interval}, goals_file={args.goals_file}")
    
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.debug("Logging reconfigured to DEBUG level")
    
    base_path = Path.cwd()
    logger.debug(f"Initializing CodeStateManager with base_path={base_path}")
    try:
        code_state_manager = CodeStateManager(base_path=base_path)
        # Explicitly refresh state since we're already in an async context
        logger.debug("Refreshing CodeStateManager state")
        await code_state_manager.refresh_state()
        logger.debug("CodeStateManager initialized")
        
        logger.debug("Initializing OutcomeAnalyzer")
        outcome_analyzer = OutcomeAnalyzer()
        logger.debug("Initializing OutcomeLogger")
        outcome_logger = OutcomeLogger(base_path=base_path)
        logger.debug("Initializing CyclePlanner")
        cycle_planner = CyclePlanner(code_state_manager, outcome_analyzer, outcome_logger, base_path=base_path)
        logger.debug("Initializing ImprovementOrchestrator")
        orchestrator = ImprovementOrchestrator(cycle_planner)
        logger.debug("Initialization complete")
        
        try:
            if args.continuous:
                logger.debug("Starting continuous cycles")
                await run_continuous_cycles(args, orchestrator)
            else:
                logger.debug("Starting single cycle")
                results = await run_single_cycle(
                    orchestrator, 
                    goals_file=args.goals_file,
                    text_goal=args.text_goal,
                    goal_id=args.goal_id
                )
                logger.info(f'Improvement cycle results: {results}')
                print(f"{datetime.now().isoformat()} - Improvement cycle results: {results}")
        except KeyboardInterrupt:
            logger.info('Received shutdown signal, exiting gracefully...')
            print(f"{datetime.now().isoformat()} - Shutdown signal received")
            sys.exit(0)
        except Exception as e:
            logger.error(f'Fatal error: {str(e)}')
            print(f"{datetime.now().isoformat()} - Fatal error: {str(e)}")
            sys.exit(1)
    except Exception as e:
        logger.error(f'Initialization error: {str(e)}')
        print(f"{datetime.now().isoformat()} - Initialization error: {str(e)}")
        sys.exit(1)

def main():
    """Main entry point for the self-improvement loop."""
    logger.debug("Script starting")
    print(f"{datetime.now().isoformat()} - Script starting")
    asyncio.run(main_async())

if __name__ == '__main__':
    main()