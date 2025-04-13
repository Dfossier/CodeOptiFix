"""
Output Formatter module for the Self-Improving AI Assistant Update Generator.

Structures the final output in the specified format (JSON or plain text).
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

import config
import utils
from utils import setup_logging
from interfaces import CodeCandidate

logger = setup_logging(__name__)

class OutputFormatter:
    """Formats the final output of the Update Generator."""
    
    def __init__(self, output_format: Optional[str] = None):
        """
        Initialize the output formatter.
        
        Args:
            output_format: The desired output format (json or text)
        """
        self.logger = logger
        self.output_format = output_format or config.DEFAULT_OUTPUT_FORMAT
        if self.output_format not in ["json", "text"]:
            self.logger.warning(
                f"Invalid output format: {self.output_format}. Using default: json"
            )
            self.output_format = "json"
    
    def format_candidate(self, candidate: CodeCandidate) -> str:
        """
        Format a code candidate in the specified format.
        
        Args:
            candidate: The code candidate to format
            
        Returns:
            Formatted candidate as a string
        """
        self.logger.info(f"Formatting candidate in {self.output_format} format")
        
        if self.output_format == "json":
            return self._format_as_json(candidate)
        else:
            return self._format_as_text(candidate)
    
    def save_candidate(
        self, 
        candidate: CodeCandidate, 
        output_path: Union[str, Path],
        create_dirs: bool = True
    ) -> None:
        """
        Save a formatted candidate to the specified path.
        
        Args:
            candidate: The code candidate to save
            output_path: Path where to save the candidate
            create_dirs: Whether to create parent directories if they don't exist
        """
        output_path = Path(output_path)
        
        # Create parent directories if needed
        if create_dirs:
            os.makedirs(output_path.parent, exist_ok=True)
        
        # Format and save the candidate
        formatted_output = self.format_candidate(candidate)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(formatted_output)
        
        self.logger.info(f"Saved candidate to {output_path}")
    
    def _format_as_json(self, candidate: CodeCandidate) -> str:
        """Format a candidate as JSON."""
        return json.dumps(
            {
                "code": candidate.code,
                "comments": candidate.comments,
                "metadata": candidate.metadata or {}
            },
            indent=2
        )
    
    def _format_as_text(self, candidate: CodeCandidate) -> str:
        """Format a candidate as plain text with delimiters."""
        result = []
        
        # Add code section
        result.append("### CODE START ###")
        result.append(candidate.code)
        result.append("### CODE END ###")
        
        # Add comments section if present
        if candidate.comments:
            result.append("\n### COMMENTS START ###")
            result.append(candidate.comments)
            result.append("### COMMENTS END ###")
        
        # Add metadata section if present
        if candidate.metadata:
            result.append("\n### METADATA START ###")
            result.append(json.dumps(candidate.metadata, indent=2))
            result.append("### METADATA END ###")
        
        return "\n".join(result)

    def format_batch(self, candidates: List[CodeCandidate]) -> str:
        """
        Format a batch of candidates.
        
        Args:
            candidates: List of code candidates to format
            
        Returns:
            Formatted batch as a string
        """
        self.logger.info(f"Formatting batch of {len(candidates)} candidates")
        
        if self.output_format == "json":
            return json.dumps(
                [
                    {
                        "code": c.code,
                        "comments": c.comments,
                        "metadata": c.metadata or {}
                    }
                    for c in candidates
                ],
                indent=2
            )
        else:
            # For text format, separate each candidate with a delimiter
            result = []
            for i, candidate in enumerate(candidates):
                result.append(f"### CANDIDATE {i+1} START ###")
                result.append(self._format_as_text(candidate))
                result.append(f"### CANDIDATE {i+1} END ###")
                result.append("")  # Empty line between candidates
            
            return "\n".join(result)
    
    def save_batch(
        self, 
        candidates: List[CodeCandidate], 
        output_path: Union[str, Path],
        create_dirs: bool = True
    ) -> None:
        """
        Save a batch of formatted candidates to the specified path.
        
        Args:
            candidates: The code candidates to save
            output_path: Path where to save the candidates
            create_dirs: Whether to create parent directories if they don't exist
        """
        output_path = Path(output_path)
        
        # Create parent directories if needed
        if create_dirs:
            os.makedirs(output_path.parent, exist_ok=True)
        
        # Format and save the batch
        formatted_output = self.format_batch(candidates)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(formatted_output)
        
        self.logger.info(f"Saved batch of {len(candidates)} candidates to {output_path}")