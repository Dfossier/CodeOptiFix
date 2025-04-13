"""
Interface definitions for the Self-Improving AI Assistant Update Generator.
"""
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
import json
from pathlib import Path

@dataclass
class ImprovementGoal:
    """Representation of an improvement goal from the Performance Analyzer."""
    target_module: str  # Path to the module to improve
    target_function: Optional[str] = None  # Function to improve (if applicable)
    description: str = ""  # Description of the improvement goal
    performance_target: Optional[str] = None  # e.g., "<100ms runtime"
    priority: int = 1  # Priority level (1-5)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ImprovementGoal':
        """Create an ImprovementGoal instance from a dictionary."""
        return cls(
            target_module=data["target_module"],
            target_function=data.get("target_function"),
            description=data.get("description", ""),
            performance_target=data.get("performance_target"),
            priority=data.get("priority", 1)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "target_module": self.target_module,
            "target_function": self.target_function,
            "description": self.description,
            "performance_target": self.performance_target,
            "priority": self.priority
        }

@dataclass
class CodeCandidate:
    """Representation of a generated code candidate."""
    code: str  # The generated code
    comments: str = ""  # Explanatory comments
    metadata: Dict[str, Any] = None  # Additional metadata
    
    def __post_init__(self):
        """Initialize default values."""
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "code": self.code,
            "comments": self.comments,
            "metadata": self.metadata
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    def to_text(self) -> str:
        """Convert to plain text format with delimiters."""
        result = []
        result.append("### CODE START ###")
        result.append(self.code)
        result.append("### CODE END ###")
        
        if self.comments:
            result.append("\n### COMMENTS START ###")
            result.append(self.comments)
            result.append("### COMMENTS END ###")
        
        if self.metadata:
            result.append("\n### METADATA START ###")
            result.append(json.dumps(self.metadata, indent=2))
            result.append("### METADATA END ###")
        
        return "\n".join(result)

class PerformanceAnalyzerInterface:
    """Interface for receiving improvement goals from the Performance Analyzer."""
    
    @staticmethod
    def load_goals(file_path: Union[str, Path]) -> List[ImprovementGoal]:
        """Load improvement goals from a JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return [ImprovementGoal.from_dict(item) for item in data]

class TestingSandboxInterface:
    """Interface for sending generated code to the Testing Sandbox."""
    
    @staticmethod
    def submit_candidate(candidate: CodeCandidate, output_path: Union[str, Path]) -> None:
        """Save a code candidate for the Testing Sandbox to evaluate."""
        # Convert Path to string for endswith check
        path_str = str(output_path)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            if path_str.endswith('.json'):
                f.write(candidate.to_json())
            else:
                f.write(candidate.to_text())