"""
Utility functions for the Self-Improving AI Assistant Update Generator.
"""
import logging
import json
import sys
import os
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

import config

def setup_logging(module_name: str) -> logging.Logger:
    """Set up and return a logger for the specified module with log rotation."""
    logger = logging.getLogger(module_name)
    
    # Only configure if no handlers exist to prevent duplicates
    if not logger.handlers:
        log_file = config.LOG_DIR / f"{module_name}.log"
        
        # Use RotatingFileHandler to limit log file size
        try:
            from logging.handlers import RotatingFileHandler
            # Max size: 5MB, keep 3 backup files
            file_handler = RotatingFileHandler(
                log_file, maxBytes=5 * 1024 * 1024, backupCount=3
            )
        except Exception:
            # Fall back to standard FileHandler if RotatingFileHandler unavailable
            file_handler = logging.FileHandler(log_file)
            
        console_handler = logging.StreamHandler(sys.stdout)
        
        # Add timestamp, module name, level, and message to log format
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Set console handler to INFO level to reduce verbosity in terminal
        console_handler.setLevel(logging.DEBUG)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        # Set overall level from config (default to INFO if not specified)
        logger.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))
        
        # Disable propagation to prevent double logging via root logger
        logger.propagate = False
    
    return logger

# Error handling
class UpdateGeneratorError(Exception):
    """Base exception for all Update Generator errors."""
    pass

class CodeAnalysisError(UpdateGeneratorError):
    """Error during code analysis phase."""
    pass

class LLMSynthesisError(UpdateGeneratorError):
    """Error during LLM synthesis phase."""
    pass

class PostProcessingError(UpdateGeneratorError):
    """Error during post-processing phase."""
    pass

class CodeUpdateError(UpdateGeneratorError):
    """Error during code update operations."""
    pass

class GoalProcessingError(UpdateGeneratorError):
    """Exception raised for errors during goal processing."""
    pass

# JSON helpers
def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load and parse a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """Save data to a JSON file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

# File handling
def read_file(file_path: Union[str, Path]) -> str:
    """Read and return the contents of a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def write_file(content: str, file_path: Union[str, Path]) -> None:
    """Write content to a file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)