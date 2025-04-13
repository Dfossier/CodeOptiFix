"""
Outcome Logger for the Self-Improving AI Assistant.

Logs cycle outcomes, goals, and transformation results to disk.
"""
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
import uuid
from datetime import datetime

from utils import setup_logging

logger = setup_logging(__name__)

class OutcomeLogger:
    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = base_path or Path.cwd()
        self.data_dir = self.base_path / "outcome_repository" / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        self.active_cycles = {}
    
    def start_cycle(self, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Start a new improvement cycle and return its ID."""
        cycle_id = str(uuid.uuid4())
        self.active_cycles[cycle_id] = {
            "start_time": datetime.now().isoformat(),
            "metadata": metadata or {},
            "goals": [],
            "transformations": []
        }
        self.logger.info(f"Started cycle {cycle_id}")
        return cycle_id
    
    def log_goal(self, goal: Dict[str, Any], validation: Dict[str, Any], cycle_id: str) -> None:
        """Log a goal and its validation for a cycle."""
        if cycle_id not in self.active_cycles:
            self.logger.error(f"Cannot log goal: Cycle {cycle_id} not found")
            return
        goal_data = {
            "goal": goal,
            "validation": validation,
            "timestamp": datetime.now().isoformat()
        }
        self.active_cycles[cycle_id]["goals"].append(goal_data)
        self.logger.debug(f"Logged goal for cycle {cycle_id}: {goal.get('description', 'unknown')}")

    def log_cycle(self, cycle_data: Dict[str, Any]) -> None:
        """Log the complete cycle data to disk."""
        cycle_id = cycle_data.get("cycle_id")
        if not cycle_id:
            self.logger.error("Cannot log cycle: No cycle_id provided")
            return
        try:
            file_path = self.data_dir / f"cycle_{cycle_id}.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(cycle_data, f, indent=2)
            self.logger.info(f"Logged cycle {cycle_id} to {file_path}")
        except Exception as e:
            self.logger.error(f"Error writing cycle {cycle_id} to disk: {str(e)}")
    
    def end_cycle(self, status: str, summary: Optional[Dict[str, Any]] = None) -> None:
        """End an improvement cycle and save its data."""
        for cycle_id, cycle_data in list(self.active_cycles.items()):
            cycle_data["end_time"] = datetime.now().isoformat()
            cycle_data["status"] = status
            cycle_data["summary"] = summary or {}
            try:
                file_path = self.data_dir / f"cycle_{cycle_id}.json"
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(cycle_data, f, indent=2)
                self.logger.info(f"Ended cycle {cycle_id} with status {status}")
            except Exception as e:
                self.logger.error(f"Error ending cycle {cycle_id}: {str(e)}")
            del self.active_cycles[cycle_id]