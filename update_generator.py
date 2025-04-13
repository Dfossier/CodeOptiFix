import logging
logger = logging.getLogger(__name__)

"""
Update Generator module for the Self-Improving AI Assistant.

Generates code updates based on improvement goals from the Goal Generator.
"""
import json
import asyncio
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import config
import utils
from utils import setup_logging, UpdateGeneratorError
from interfaces import ImprovementGoal, CodeCandidate
from llm_synthesizer import LLMSynthesizer
logger = setup_logging(__name__)

class UpdateGenerator:
    """
    Generates code updates based on improvement goals.
    
    This class processes improvement goals, generates code candidates,
    and handles the coordination between different components.
    """

    def __init__(self, base_path: Optional[Path]=None):
        """
        Initialize the update generator.
        
        Args:
            base_path: Base path of the codebase to improve
        """
        self.logger = logger
        self.base_path = base_path or Path.cwd()
        self.llm_synthesizer = LLMSynthesizer()
        self.logger.info(f'Initialized UpdateGenerator with base path: {self.base_path}')

    async def batch_process(self, goals_file: Union[str, Path], output_dir: Union[str, Path], candidates_per_goal: int=2) -> Dict[str, List[CodeCandidate]]:
        """
        Process multiple improvement goals from a file.
        
        Args:
            goals_file: Path to a JSON file containing improvement goals
            output_dir: Directory to save generated candidates
            candidates_per_goal: Number of candidates to generate per goal
            
        Returns:
            Dictionary mapping goal descriptions to lists of candidates
        """
        self.logger.info(f'Processing goals from {goals_file}')
        goals_path = Path(goals_file)
        output_path = Path(output_dir)
        os.makedirs(output_path, exist_ok=True)
        try:
            with open(goals_path, 'r', encoding='utf-8') as f:
                goals_data = json.load(f)
            goals = []
            if isinstance(goals_data, list):
                for goal_dict in goals_data:
                    goals.append(ImprovementGoal.from_dict(goal_dict))
            else:
                goals.append(ImprovementGoal.from_dict(goals_data))
            self.logger.info(f'Loaded {len(goals)} goals from {goals_path}')
        except Exception as e:
            self.logger.error(f'Error loading goals from {goals_path}: {str(e)}')
            raise UpdateGeneratorError(f'Failed to load goals: {str(e)}') from e
        results = {}
        for goal in goals:
            try:
                candidates = await self.generate_updates(goal, candidates_per_goal=candidates_per_goal)
                results[goal.description] = candidates
                goal_dir = output_path / f'goal_{hash(goal.description) & 4294967295:08x}'
                os.makedirs(goal_dir, exist_ok=True)
                for i, candidate in enumerate(candidates):
                    candidate_file = goal_dir / f'candidate_{i + 1}.json'
                    with open(candidate_file, 'w', encoding='utf-8') as f:
                        f.write(candidate.to_json())
            except Exception as e:
                self.logger.error(f"Error processing goal '{goal.description}': {str(e)}")
                results[goal.description] = []
        return results

    async def generate_updates(self, goal: ImprovementGoal, candidates_per_goal: int=2) -> List[CodeCandidate]:
        """
        Generate code updates for a specific improvement goal.
        
        Args:
            goal: The improvement goal
            candidates_per_goal: Number of candidates to generate
            
        Returns:
            List of code candidates
        """
        try:
            self.logger.info(f'Generating updates for goal: {goal.description}')
            target_module_path = self.base_path / goal.target_module
            if not target_module_path.exists():
                raise UpdateGeneratorError(f'Target module not found: {goal.target_module}')
            context = self._create_context(goal)
            candidates = await self.llm_synthesizer.generate_candidates(goal, context, num_candidates=candidates_per_goal)
            self.logger.info(f'Generated {len(candidates)} candidates for goal: {goal.description}')
            return candidates
        except json.JSONDecodeError as e:
            raise UpdateGeneratorError(f'Invalid JSON configuration: {str(e)}') from e
        except FileNotFoundError as e:
            raise UpdateGeneratorError(f'Required file not found: {str(e)}') from e
        except PermissionError as e:
            raise UpdateGeneratorError(f'Permission denied: {str(e)}') from e
        except asyncio.TimeoutError as e:
            raise UpdateGeneratorError('Operation timed out') from e
        except utils.UpdateGeneratorError as e:
            raise
        except Exception as e:
            raise UpdateGeneratorError(f'Unexpected error during update generation: {str(e)}') from e

    def _create_context(self, goal: ImprovementGoal) -> Dict[str, Any]:
        """Create context information about the code to be improved."""
        try:
            target_module_path = self.base_path / goal.target_module
            with open(target_module_path, 'r', encoding='utf-8') as f:
                module_content = f.read()
            context = {'module_info': {'module_path': str(goal.target_module), 'content': module_content}}
            if goal.target_function:
                context['function_info'] = {'name': goal.target_function}
            return context
        except Exception as e:
            self.logger.error(f'Error creating context: {str(e)}')
            raise UpdateGeneratorError(f'Failed to create context: {str(e)}') from e

async def generate_updates_standalone() -> None:
    """Standalone function for generating updates."""
    try:
        import argparse
        parser = argparse.ArgumentParser(description='Generate code updates')
        parser.add_argument('--goal', type=str, required=True, help='Path to goal JSON file')
        parser.add_argument('--output', type=str, default='./output', help='Output directory')
        parser.add_argument('--candidates', type=int, default=2, help='Candidates per goal')
        args = parser.parse_args()
        update_generator = UpdateGenerator()
        await update_generator.batch_process(goals_file=args.goal, output_dir=args.output, candidates_per_goal=args.candidates)
    except json.JSONDecodeError as e:
        raise UpdateGeneratorError(f'Invalid JSON configuration: {str(e)}') from e
    except FileNotFoundError as e:
        raise UpdateGeneratorError(f'Required file not found: {str(e)}') from e
    except PermissionError as e:
        raise UpdateGeneratorError(f'Permission denied: {str(e)}') from e
    except asyncio.TimeoutError as e:
        raise UpdateGeneratorError('Operation timed out') from e
    except utils.UpdateGeneratorError as e:
        raise
    except Exception as e:
        raise UpdateGeneratorError(f'Unexpected error during update generation: {str(e)}') from e

def main() -> None:
    """Command-line entry point."""
    try:
        asyncio.run(generate_updates_standalone())
    except UpdateGeneratorError as e:
        logger.info('{}', (f'Error: {str(e)}',))
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info('\nOperation cancelled by user', ('\nOperation cancelled by user',))
        sys.exit(1)
    except Exception as e:
        logger.info('{}', (f'Unexpected error: {str(e)}',))
        sys.exit(1)
if __name__ == '__main__':
    main()