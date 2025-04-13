import logging
logger = logging.getLogger(__name__)

"""
Command-line interface for the CodeOptiFix2 system.

This script provides a simple interface for applying code transformations.
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import List, Optional
from code_updater import CodeUpdater
from interfaces import ImprovementGoal
from orchestrator.improvement_orchestrator import ImprovementOrchestrator

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='CodeOptiFix2 - Code transformation system')
    parser.add_argument('--target-module', type=str, help='Path to the module to transform')
    parser.add_argument('--target-function', type=str, help='Function to transform (if applicable)')
    parser.add_argument('--transformation-type', type=str, help='Type of transformation to apply')
    parser.add_argument('--description', type=str, help='Description of the transformation')
    parser.add_argument('--goals-file', type=str, help='Path to a JSON file with improvement goals')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--base-path', type=str, help='Base path for the codebase')
    parser.add_argument('--list-transformers', action='store_true', help='List available transformers')
    parser.add_argument('--list-validators', action='store_true', help='List available validators')
    return parser.parse_args()

def load_goals_from_file(file_path: str) -> List[ImprovementGoal]:
    """Load improvement goals from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            goals_data = json.load(f)
        goals = []
        for goal_data in goals_data:
            goals.append(ImprovementGoal.from_dict(goal_data))
        return goals
    except Exception as e:
        logger.info('{}', (f'Error loading goals from {file_path}: {str(e)}',))
        sys.exit(1)

async def main():
    """Main entry point for the CLI."""
    args = parse_args()
    base_path = Path(args.base_path) if args.base_path else Path.cwd()
    orchestrator = ImprovementOrchestrator(base_path=base_path)
    config_file = Path(args.config) if args.config else None
    updater = CodeUpdater(base_path=base_path, config_file=config_file)
    if args.list_transformers:
        logger.info('Available transformers:', ('Available transformers:',))
        for name, transformer_cls in updater.registry.get_all_transformers().items():
            logger.info('{}', (f'- {name}: {(transformer_cls.get_description() if hasattr(transformer_cls, 'get_description') else '')}',))
        return
    if args.list_validators:
        logger.info('Available validators:', ('Available validators:',))
        for name, validator_cls in updater.registry.get_all_validators().items():
            logger.info('{}', (f'- {name}: {(validator_cls.get_description() if hasattr(validator_cls, 'get_description') else '')}',))
        return
    goals = []
    if args.goals_file:
        goals = load_goals_from_file(args.goals_file)
    elif args.target_module:
        if not args.transformation_type:
            logger.info('Error: When specifying --target-module, you must also specify --transformation-type', ('Error: When specifying --target-module, you must also specify --transformation-type',))
            sys.exit(1)
        goal = ImprovementGoal(target_module=args.target_module, target_function=args.target_function, description=args.description or f'Apply {args.transformation_type} to {args.target_module}', priority=1)
        goals.append(goal)
    else:
        logger.info('Error: You must specify either --goals-file or --target-module', ('Error: You must specify either --goals-file or --target-module',))
        parser.print_help()
        sys.exit(1)
    logger.info('{}', (f'Applying {len(goals)} improvement goals...',))
    results = await orchestrator.execute_cycle(goals)
    for i, result in enumerate(results):
        logger.info('{}', (f'\nGoal {i + 1}: {result['goal']['description']}',))
        logger.info('{}', (f'Status: {result['status']}',))
        logger.info('{}', (f'Message: {result['message']}',))
        if result['status'] == 'success' and 'transformations' in result:
            logger.info('{}', (f'Applied transformations: {len(result['transformations'])}',))
            for t in result['transformations']:
                logger.info('{}', (f'  - {t['file_path']}: {t['transformation_type']}',))
if __name__ == '__main__':
    asyncio.run(main())