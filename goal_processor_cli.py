import logging
"""
Goal Processor CLI for Natural Language Goal Descriptions.

Command-line interface for processing natural language goal descriptions into
structured improvement goals.
"""
import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from goal_intelligence.goal_processor import GoalProcessor
from utils import setup_logging
logger = setup_logging(__name__)

async def process_goal_file(file_path: str, output_path: str=None) -> None:
    """
    Process a natural language goal file.
    
    Args:
        file_path: Path to the text file containing the goal description
        output_path: Optional path to save the processed goal
    """
    try:
        logger.info(f'Processing goal file: {file_path}')
        processor = GoalProcessor()
        goal_record = await processor.process_text_goal(file_path)
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(goal_record, f, indent=2)
            logger.info(f'Processed goal saved to {output_path}')
        logger.info('\n=== Goal Processing Summary ===')
        logger.info('{}', (f'Goal ID: {goal_record['goal_id']}',))
        logger.info('{}', (f'Original Text: {goal_record['original_text'][:100]}...',))
        logger.info('{}', (f'Sub-goals Extracted: {len(goal_record['structured_goals'])}',))
        logger.info('\nExtracted Sub-goals:')
        for i, sub_goal in enumerate(goal_record['structured_goals']):
            logger.info('{}', (f'  {i + 1}. {sub_goal['description'][:80]}...',))
            logger.info('{}', (f'     Target Module: {sub_goal['target_module']}',))
            logger.info('{}', (f'     Priority: {sub_goal['priority']}',))
            logger.info('{}', (f'     Type: {sub_goal['type']}',))
            logger.info('')
        logger.info('{}', (f'Processing complete. Goal record saved to {goal_record.get('output_path', 'processed_goals directory')}',))
    except Exception as e:
        logger.error(f'Error processing goal file: {str(e)}')
        logger.info('{}', (f'Error: {str(e)}',))
        sys.exit(1)

async def analyze_logs(goal_id: str, scan_system_logs: bool=True) -> None:
    """
    Analyze logs for a specific goal.
    
    Args:
        goal_id: ID of the goal to analyze logs for
        scan_system_logs: Whether to also scan system logs
    """
    try:
        logger.info(f'Analyzing logs for goal ID: {goal_id}')
        processor = GoalProcessor()
        insights = await processor.analyze_error_logs(goal_id, scan_system_logs)
        logger.info('\n=== Log Analysis Results ===')
        logger.info('{}', (f'Goal ID: {goal_id}',))
        logger.info('{}', (f'Insights Extracted: {len(insights)}',))
        if insights:
            logger.info('\nInsights:')
            for i, insight in enumerate(insights):
                if insight.get('llm_enhanced', False):
                    logger.info('{}', (f'  {i + 1}. Error Pattern: {insight['error_pattern']}',))
                    logger.info('{}', (f'     Probable Cause: {insight['probable_cause']}',))
                    logger.info('{}', (f'     Recommendation: {insight['recommendation']}',))
                    logger.info('{}', (f'     Priority: {insight['priority']}',))
                    if 'original_log' in insight:
                        logger.info('{}', (f'     Based on: {insight['original_log'][:80]}...',))
                    logger.info('{}', (f'     Source: {insight.get('source', 'unknown')}',))
                else:
                    logger.info('{}', (f'  {i + 1}. Log: {insight.get('log', '')[:80]}...',))
                    logger.info('{}', (f'     Insight: {insight.get('insight', 'No insight')}',))
                    logger.info('{}', (f'     Source: {insight.get('source', 'unknown')}',))
                logger.info('')
        else:
            logger.info('\nNo insights extracted from logs.')
    except Exception as e:
        logger.error(f'Error analyzing logs: {str(e)}')
        logger.info('{}', (f'Error: {str(e)}',))
        sys.exit(1)

async def list_goals() -> None:
    """List all processed goals."""
    try:
        logger.info('Listing all processed goals')
        processor = GoalProcessor()
        goal_files = list(processor.goals_dir.glob('goal_*_*.json'))
        if not goal_files:
            logger.info('No processed goals found.')
            return
        logger.info('\n=== Processed Goals ===')
        logger.info('{}', (f'Total Goals: {len(goal_files)}',))
        logger.info('')
        for goal_file in goal_files:
            try:
                with open(goal_file, 'r', encoding='utf-8') as f:
                    goal_record = json.load(f)
                logger.info('{}', (f'Goal ID: {goal_record.get('goal_id', 'Unknown')}',))
                logger.info('{}', (f'Timestamp: {goal_record.get('timestamp', 'Unknown')}',))
                logger.info('{}', (f'Status: {goal_record.get('status', 'Unknown')}',))
                logger.info('{}', (f'Attempts: {goal_record.get('attempts', 0)}',))
                logger.info('{}', (f'Sub-goals: {goal_record.get('sub_goals_completed', 0)}/{goal_record.get('sub_goals_total', 0)}',))
                logger.info('{}', (f'Description: {goal_record.get('original_text', 'No description')[:100]}...',))
                logger.info('')
            except Exception as e:
                logger.warning(f'Error loading goal file {goal_file}: {str(e)}')
                logger.info('{}', (f'Error loading {goal_file.name}: {str(e)}',))
    except Exception as e:
        logger.error(f'Error listing goals: {str(e)}')
        logger.info('{}', (f'Error: {str(e)}',))
        sys.exit(1)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Process natural language goal descriptions into structured improvement goals')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    process_parser = subparsers.add_parser('process', help='Process a natural language goal file')
    process_parser.add_argument('file_path', help='Path to the text file containing the goal description')
    process_parser.add_argument('--output', '-o', help='Path to save the processed goal')
    analyze_parser = subparsers.add_parser('analyze', help='Analyze logs for a specific goal')
    analyze_parser.add_argument('goal_id', help='ID of the goal to analyze logs for')
    analyze_parser.add_argument('--no-system-logs', action='store_true', help="Don't scan system log files")
    list_parser = subparsers.add_parser('list', help='List all processed goals')
    return parser.parse_args()

async def main():
    """Main entry point for the goal processor CLI."""
    args = parse_args()
    try:
        if args.command == 'process':
            await process_goal_file(args.file_path, args.output)
        elif args.command == 'analyze':
            scan_system_logs = not args.no_system_logs
            await analyze_logs(args.goal_id, scan_system_logs)
        elif args.command == 'list':
            await list_goals()
        else:
            logger.info('Please specify a command. Use --help for more information.')
            sys.exit(1)
    except Exception as e:
        logger.info('{}', (f'Error: {str(e)}',))
        import traceback
        traceback.print_exc()
        sys.exit(1)
if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info('\nOperation cancelled by user.')
        sys.exit(0)
    except Exception as e:
        logger.info('{}', (f'Error: {str(e)}',))
        sys.exit(1)