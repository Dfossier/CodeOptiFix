"""
Goal Generator module for the Self-Improving AI Assistant Update Generator.

Analyzes code to autonomously identify potential improvement opportunities
and generates corresponding improvement goals.
"""
import os
import sys
import ast
import re
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Union
import logging
import datetime
import config
import utils
from utils import setup_logging
from interfaces import ImprovementGoal
from code_analyzer import CodeAnalyzer

logger = setup_logging(__name__)

class GoalGenerator:
    """Analyzes code and generates improvement goals."""

    def __init__(self, base_path: Optional[Path]=None):
        """
        Initialize the goal generator.
        
        Args:
            base_path: Base path of the codebase to analyze
        """
        self.logger = logger
        self.base_path = base_path or Path.cwd()
        self.code_analyzer = CodeAnalyzer(self.base_path)
        self.patterns = {
            'performance': [
                ('for\\s+\\w+\\s+in\\s+range\\(\\w+\\)', 'Loop optimization potential'),
                ('\\.append\\(.*\\).*for\\s+', 'List comprehension opportunity'),
                ('time\\.sleep\\(', 'Consider async patterns for I/O operations'),
                ('import\\s+pandas|import\\s+numpy', 'Data processing optimization potential'),
                ('open\\(.*\\).*read\\(\\)', 'Optimize file reading with context manager'),
                ('\\.join\\(.*\\)', 'String concatenation optimization')
            ],
            'reliability': [
                ('except\\s*:', 'Add specific exception handling'),
                ('assert\\s+', 'Consider converting asserts to proper validation'),
                ('print\\(.*error|exception', 'Replace print with proper logging'),
                ('float\\(.*\\)', 'Add error handling for string to float conversion'),
                ('\\.get\\(.*\\)', 'Add default value to dictionary get method'),
                ('os\\.path\\.', 'Add file existence checks')
            ],
            'maintainability': [
                ('#\\s*TODO', 'Address TODO comment'),
                ('if\\s+.*if\\s+.*if', 'Reduce nested conditionals'),
                ('def\\s+\\w+\\([^)]{100,}\\)', 'Function with too many parameters'),
                ('(\\w+)\\s*=\\s*\\1\\s*\\+\\s*', 'Consider using augmented assignment'),
                ('([\\"\']).*\\1.*\\+', 'Use f-strings instead of concatenation'),
                ('\\t', 'Replace tabs with spaces for consistent formatting'),
                ('def\\s+\\w+\\(', 'Add docstring to function')
            ],
            'features': [
                ('datetime\\.|date\\.', 'Consider timezone handling'),
                ('weekday|monday|tuesday|wednesday|thursday|friday|saturday|sunday', 'Add weekend/holiday handling'),
                ('password|auth|login', 'Add additional security measures'),
                ('\\.csv|\\.json|\\.xml', 'Add support for additional file formats'),
                ('logging\\.|logger\\.', 'Add structured logging for better analytics')
            ],
            'security': [
                ('subprocess\\.', 'Enhance subprocess security checks'),
                ('eval\\(|exec\\(', 'Replace dangerous eval/exec functions'),
                ('\\.read_csv\\(|\\.read_json\\(', 'Add input validation for data loading'),
                ('jwt\\.|token', 'Improve token security measures'),
                ('\\.execute\\(', 'Add SQL injection protection')
            ],
            'testing': [
                ('def test_', 'Expand test coverage'),
                ('@pytest', 'Add more test assertions'),
                ('assert ', 'Improve test assertions with better messages'),
                ('mock\\.|patch\\.', 'Enhance test mocking strategies')
            ],
            'documentation': [
                ('def [^\\"\']*\\):', 'Add missing function docstring'),
                ('class [^\\"\']*:', 'Add missing class docstring'),
                (':[^\\n]*\\n\\s*[^\\s#]', 'Add missing parameter documentation'),
                ('return [^#\\n]*', 'Document return values')
            ]
        }

    def generate_goals(self, num_goals: int=5, output_file: Optional[Union[str, Path]]=None, max_per_category: int=2) -> List[ImprovementGoal]:
        """
        Generate improvement goals by analyzing the codebase.
        
        Args:
            num_goals: Maximum number of goals to generate
            output_file: Optional file path to save the generated goals
            max_per_category: Maximum number of goals per category
            
        Returns:
            List of generated ImprovementGoal objects
        """
        self.logger.info(f'Generating up to {num_goals} improvement goals...')
        python_files = self._find_python_files()
        self.logger.info(f'Found {len(python_files)} Python files to analyze')
        self.logger.debug(f'Python files: {[str(f) for f in python_files]}')
        opportunities = []
        for file_path in python_files:
            try:
                module_path = str(Path(file_path).relative_to(self.base_path)).replace("\\", "/")
                if not module_path:
                    self.logger.warning(f"Invalid module_path for {file_path}")
                    continue
                self.logger.debug(f'Analyzing {module_path}...')
                module_info = self.code_analyzer.analyze_module(module_path)
                self.logger.debug(f'Module info: {module_info}')
                if not module_info.get('module_path') or not module_info.get('content'):
                    self.logger.warning(f"Invalid module_info for {module_path}: missing module_path or content")
                    continue
                module_opportunities = self._find_opportunities_in_module(module_info)
                self.logger.debug(f'Found {len(module_opportunities)} opportunities in {module_path}')
                opportunities.extend(module_opportunities)
            except Exception as e:
                self.logger.error(f'Error analyzing {file_path}: {str(e)}')
        self.logger.debug(f'Raw opportunities: {opportunities}')
        # Filter invalid opportunities
        valid_opportunities = [
            opp for opp in opportunities
            if opp.get('module_path') and opp.get('description')
        ]
        self.logger.debug(f'Valid opportunities: {valid_opportunities}')
        if not valid_opportunities:
            self.logger.warning('No valid opportunities found')
        opportunities = valid_opportunities
        opportunities.sort(key=lambda x: x['score'], reverse=True)
        selected_opportunities = []
        category_counts = {}
        module_counts = {}
        descriptions_seen = set()
        for opportunity in opportunities:
            category = opportunity['category']
            module = opportunity['module_path']
            description = opportunity['description']
            if description in descriptions_seen:
                continue
            if category_counts.get(category, 0) >= max_per_category:
                continue
            if module_counts.get(module, 0) >= 2:
                continue
            selected_opportunities.append(opportunity)
            category_counts[category] = category_counts.get(category, 0) + 1
            module_counts[module] = module_counts.get(module, 0) + 1
            descriptions_seen.add(description)
            if len(selected_opportunities) >= num_goals:
                break
        if len(selected_opportunities) < num_goals:
            for opportunity in opportunities:
                if opportunity not in selected_opportunities:
                    description = opportunity['description']
                    if description not in descriptions_seen:
                        selected_opportunities.append(opportunity)
                        descriptions_seen.add(description)
                        if len(selected_opportunities) >= num_goals:
                            break
        goals = []
        for opportunity in selected_opportunities[:num_goals]:
            if not opportunity.get('module_path') or not opportunity.get('description'):
                self.logger.warning(f"Skipping invalid opportunity: {opportunity}")
                continue
            goal = ImprovementGoal(
                target_module=opportunity['module_path'],
                target_function=opportunity.get('function_name'),
                description=opportunity['description'],
                performance_target=opportunity.get('performance_target'),
                priority=min(5, max(1, round(opportunity['score'])))
            )
            self.logger.debug(f'Generated goal: {goal.to_dict()}')
            goals.append(goal)
        self.logger.info(f'Generated {len(goals)} improvement goals')
        if output_file:
            output_file = Path(output_file)
            os.makedirs(output_file.parent, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump([goal.to_dict() for goal in goals], f, indent=2)
            self.logger.info(f'Saved goals to {output_file}')
        return goals

    def _find_python_files(self) -> List[Path]:
        """Find all Python files in the codebase."""
        python_files = []
        skip_patterns = ['.venv', 'site-packages', '.git', '__pycache__', '.pytest_cache']
        for root, _, files in os.walk(self.base_path):
            if any((pattern in root for pattern in skip_patterns)):
                continue
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    if 'goal_generator.py' in str(file_path):
                        continue
                    python_files.append(file_path)
        self.logger.debug(f'Found python files: {[str(f) for f in python_files]}')
        return python_files

    def _find_opportunities_in_module(self, module_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find improvement opportunities in a module."""
        opportunities = []
        content = module_info.get('content', '')
        module_path = module_info.get('module_path', '')
        if not content or not module_path:
            self.logger.warning(f'Empty content or module_path for {module_path}')
            return []
        found_patterns = set()
        for category, patterns in self.patterns.items():
            for pattern, description_template in patterns:
                pattern_key = f'{pattern}:{description_template}'
                if pattern_key in found_patterns:
                    continue
                matches = list(re.finditer(pattern, content))
                if matches:
                    match = matches[0]
                    found_patterns.add(pattern_key)
                    line_num = content[:match.start()].count('\n') + 1
                    context = content.splitlines()[max(0, line_num - 2):line_num + 1]
                    context_str = '\n'.join(context)
                    function_name = self._find_function_containing_line(module_info, line_num)
                    description = self._generate_description(description_template, category, module_path, function_name, context_str)
                    score = self._calculate_opportunity_score(category, context_str, function_name, module_info)
                    performance_target = self._generate_performance_target(category, description)
                    opportunities.append({
                        'module_path': module_path,
                        'function_name': function_name,
                        'line': line_num,
                        'category': category,
                        'description': description,
                        'score': score,
                        'performance_target': performance_target,
                        'pattern_key': pattern_key
                    })
        for function_info in module_info.get('functions', []):
            complexity = function_info.get('complexity', {})
            cyclomatic_complexity = complexity.get('cyclomatic_complexity', 0)
            line_count = complexity.get('line_count', 0)
            if cyclomatic_complexity > 10 or line_count > 100:
                description = f'Refactor complex function {function_info['name']} '
                description += f'(complexity: {cyclomatic_complexity}, lines: {line_count})'
                opportunities.append({
                    'module_path': module_path,
                    'function_name': function_info['name'],
                    'line': function_info['ast_node'].lineno,
                    'category': 'maintainability',
                    'description': description,
                    'score': min(5, cyclomatic_complexity / 5 + line_count / 50),
                    'performance_target': None
                })
        self.logger.debug(f'Opportunities for {module_path}: {opportunities}')
        return opportunities

    def _find_function_containing_line(self, module_info: Dict[str, Any], line_num: int) -> Optional[str]:
        """Find which function contains the given line number."""
        for function_info in module_info.get('functions', []):
            node = function_info.get('ast_node')
            if node:
                func_start = node.lineno
                func_end = function_info.get('complexity', {}).get('line_count', func_start)
                if func_start <= line_num <= func_end:
                    return function_info['name']
        return None

    def _generate_description(self, template: str, category: str, module_path: str, function_name: Optional[str], context: str) -> str:
        """Generate a specific improvement description."""
        keywords = {
            'datetime': 'timestamp handling',
            'weekday': 'weekend or weekday logic',
            'monday|tuesday|wednesday|thursday|friday|saturday|sunday': 'day-specific logic',
            'password': 'password handling',
            'auth': 'authentication',
            'login': 'login process',
            'csv': 'CSV processing',
            'json': 'JSON processing',
            'xml': 'XML processing',
            'append': 'list operations',
            'range': 'loop',
            'sleep': 'blocking operation',
            'pandas': 'data processing',
            'numpy': 'numerical operations',
            'try|except': 'error handling',
            'if': 'conditional logic',
            'for|while': 'iteration'
        }
        focus = ''
        for keyword_pattern, description in keywords.items():
            if re.search(keyword_pattern, context, re.IGNORECASE):
                focus = description
                break
        focus = focus or 'code'
        if function_name:
            description = f'{template} in {function_name}() {focus}'
        else:
            description = f'{template} in {focus}'
        category_prefixes = {
            'performance': 'Optimize ',
            'reliability': 'Improve ',
            'maintainability': 'Refactor ',
            'features': 'Add '
        }
        return f'{category_prefixes.get(category, '')}{description}'

    def _calculate_opportunity_score(self, category: str, context: str, function_name: Optional[str], module_info: Dict[str, Any]) -> float:
        """
        Calculate a score for an improvement opportunity.
        Higher score = higher priority.
        """
        category_scores = {
            'performance': 3.0,
            'reliability': 4.0,
            'maintainability': 2.0,
            'features': 3.5
        }
        score = category_scores.get(category, 3.0)
        if 'error' in context or 'exception' in context:
            score += 1.0
        if 'TODO' in context or 'FIXME' in context:
            score += 1.5
        if function_name:
            for function_info in module_info.get('functions', []):
                if function_info['name'] == function_name:
                    complexity = function_info.get('complexity', {})
                    cyclomatic_complexity = complexity.get('cyclomatic_complexity', 0)
                    if cyclomatic_complexity > 15:
                        score += 1.5
                    elif cyclomatic_complexity > 10:
                        score += 1.0
                    elif cyclomatic_complexity > 5:
                        score += 0.5
        return min(5.0, max(1.0, score))

    def _generate_performance_target(self, category: str, description: str) -> Optional[str]:
        """Generate a performance target if applicable."""
        if category != 'performance':
            return None
        if 'loop' in description.lower() or 'optimization' in description.lower():
            return '<50ms execution time'
        if 'memory' in description.lower():
            return '<100MB peak memory usage'
        if 'data processing' in description.lower():
            return '<200ms execution time'
        return '<100ms execution time'

async def main():
    """Command-line entry point."""
    import argparse
    parser = argparse.ArgumentParser(description='Goal Generator for Self-Improving AI Assistant')
    parser.add_argument('--output', '-o', default='generated_goals.json', help='Output file path for generated goals')
    parser.add_argument('--count', '-c', type=int, default=5, help='Number of goals to generate')
    parser.add_argument('--path', '-p', default=None, help='Base path of the codebase to analyze')
    args = parser.parse_args()
    base_path = Path(args.path) if args.path else None
    goal_generator = GoalGenerator(base_path)
    goals = goal_generator.generate_goals(num_goals=args.count, output_file=args.output)
    logger.info(f'Generated {len(goals)} improvement goals:')
    for i, goal in enumerate(goals):
        logger.info(f'{i + 1}. [{goal.priority}] {goal.description}')
        logger.info(f'   Module: {goal.target_module}')
        if goal.target_function:
            logger.info(f'   Function: {goal.target_function}')
        if goal.performance_target:
            logger.info(f'   Target: {goal.performance_target}')
        logger.info('')

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())