# CodeOptiFix2

A feedback-driven, adaptive system for automatically improving code quality through AST-based transformations.

## 🔍 Overview

CodeOptiFix2 is a self-improving system that applies code transformations to enhance codebase quality. It features:

- **AST-based transformations**: Uses libcst for precise code manipulation
- **Feedback-driven improvements**: Learns from past outcomes to enhance future cycles
- **Context-aware transformations**: Analyzes codebase state to ensure relevant changes
- **Adaptive goal refinement**: Adjusts improvement goals to match code reality
- **Extensible plugin architecture**: Easily add new transformers and validators

## 🏗️ Architecture

The system follows a feedback-driven architecture that enables continuous learning and adaptation:

```
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│                 Feedback-Driven Improvement Loop                 │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌────────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│                    │    │                 │    │                 │
│  Code State        │◄──►│  Goal           │◄──►│  Outcome        │
│  Observatory       │    │  Intelligence   │    │  Repository     │
│                    │    │                 │    │                 │
└────────┬───────────┘    └────────┬────────┘    └────────┬────────┘
         │                         │                      │
         │                         │                      │
         ▼                         ▼                      ▼
┌────────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│                    │    │                 │    │                 │
│  Pattern Registry  │    │  Goal Validator │    │  Outcome Logger │
│  Dependency Mapper │    │  Goal Refiner   │    │  Outcome        │
│  State Persistence │    │  Goal           │    │  Analyzer       │
│                    │    │  Prioritizer    │    │                 │
└────────────────────┘    └─────────────────┘    └─────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│                      Transformation System                       │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌────────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│                    │    │                 │    │                 │
│  Transformer       │    │  Code Updater   │    │  Validation     │
│  Registry          │    │                 │    │  Rules          │
│                    │    │                 │    │                 │
└────────────────────┘    └─────────────────┘    └─────────────────┘
```

### Components

#### 1. Code State Observatory
Provides a comprehensive view of the codebase:
- **Pattern Registry**: Catalogs code patterns (prints, logging, conditionals)
- **Dependency Mapper**: Tracks relationships between code elements
- **State Persistence**: Stores snapshots for historical comparison

#### 2. Goal Intelligence Framework
Validates, refines, and processes improvement goals:
- **Goal Validator**: Pre-screens goals against code state
- **Goal Refiner**: Adjusts goals to match codebase reality
- **Goal Prioritizer**: Ranks goals by impact and success probability
- **Goal Processor**: Converts natural language goal descriptions into structured improvement goals

#### 3. Outcome Repository
Centralizes knowledge about transformation outcomes:
- **Outcome Logger**: Records detailed transformation results
- **Outcome Analyzer**: Identifies patterns in successes and failures

#### 4. Transformation System
Applies code transformations safely:
- **Transformer Registry**: Manages plugin-based transformers
- **Code Updater**: Applies AST transformations to code
- **Validation Rules**: Ensures transformed code meets quality standards

#### 5. Improvement Orchestrator
Coordinates the improvement process:
- **Cycle Planner**: Determines execution strategy for improvement cycles
- **Feedback Integrator**: Incorporates outcome data into planning

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/CodeOptiFix2.git
cd CodeOptiFix2

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Usage

### Command Line Interface

```bash
# Run a single improvement cycle with specific goals from a JSON file
python self_improvement_loop.py --goals-file goals.json --output results.json

# Run a single improvement cycle with a natural language goal description
python self_improvement_loop.py --text-goal example_nl_goal.txt

# Run using a previously processed goal by its ID
python self_improvement_loop.py --goal-id 8f7e6d5c

# Start continuous improvement mode with a natural language goal
python self_improvement_loop.py --continuous --interval 3600 --text-goal example_nl_goal.txt

# Apply specific transformations
python updater_cli.py --target-module utils.py --transformation-type replace_print_with_logging

# List available transformers
python updater_cli.py --list-transformers

# Process a natural language goal file without running an improvement cycle
python goal_processor_cli.py process example_nl_goal.txt

# List all processed goals
python goal_processor_cli.py list

# Analyze logs for a specific goal
python goal_processor_cli.py analyze 8f7e6d5c
```

### Python API

```python
from orchestrator.improvement_orchestrator import ImprovementOrchestrator
from interfaces import ImprovementGoal
from goal_intelligence.goal_processor import GoalProcessor

# Initialize orchestrator
orchestrator = ImprovementOrchestrator()

# Method 1: Using manually created improvement goals
goals = [
    ImprovementGoal(
        target_module="utils.py",
        description="Replace print statements with structured logging",
        priority=1
    )
]

# Method 2: Using natural language goal description
async def process_natural_language_goal():
    processor = GoalProcessor()
    goal_record = await processor.process_text_goal("example_nl_goal.txt")
    return processor.create_improvement_goals(goal_record)

# Execute improvement cycle with manual goals
async def run_manual_goals():
    results = await orchestrator.execute_cycle(goals)
    print(f"Status: {results['status']}")
    print(f"Transformations: {results['successful_transformations']}")

# Execute improvement cycle with natural language goals
async def run_nl_goals():
    nl_goals = await process_natural_language_goal()
    results = await orchestrator.execute_cycle([goal.to_dict() for goal in nl_goals])
    print(f"Status: {results['status']}")
    print(f"Transformations: {results['successful_transformations']}")

# Run the cycle
import asyncio
asyncio.run(run_manual_goals())
# or
# asyncio.run(run_nl_goals())
```

## 🔌 Extending with Plugins

### Creating a Custom Transformer

1. Create a new Python file in the `transformers` or `plugins` directory
2. Define a class that inherits from `TransformerBase`
3. Implement the required methods

```python
from code_updater import TransformerBase
import libcst as cst

class MyTransformer(TransformerBase):
    """Custom transformer that does something awesome."""
    
    def get_name(self) -> str:
        return "MyTransformer"
    
    def get_description(self) -> str:
        return "Does something awesome to your code."
    
    @classmethod
    def can_handle(cls, transformation_type: str) -> bool:
        return transformation_type == "do_something_awesome"
    
    def leave_FunctionDef(self, original_node, updated_node):
        # Your transformation logic here
        self.changes_made = True
        return modified_node
```

### Creating a Custom Validator

```python
from code_updater import ValidationRule
from pathlib import Path
from typing import Tuple, Optional

class MyValidator(ValidationRule):
    """Custom validator for specific rules."""
    
    def get_name(self) -> str:
        return "MyValidator"
    
    def get_description(self) -> str:
        return "Validates code against my custom rules."
    
    def validate(self, file_path: Path, code: str) -> Tuple[bool, Optional[str]]:
        # Your validation logic here
        if problem_detected:
            return False, "Problem detected: description"
        return True, None
```

## 🔧 Available Transformers

The system includes several built-in transformers:

1. **Replace Print with Logging**: Converts print statements to structured logging calls
2. **Add Exception Handling**: Adds try/except blocks with proper error handling
3. **Add Structured Logging**: Enhances logging with contextual information
4. **Extract Function**: Refactors large functions into smaller, focused functions
5. **Split File**: Divides large files into logical modules

## 📄 License

This project is licensed under the MIT License.