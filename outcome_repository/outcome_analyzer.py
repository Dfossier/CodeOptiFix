"""
Outcome Analyzer for the Self-Improving AI Assistant.

Analyzes transformation outcomes to inform future improvements.
"""
import json
from pathlib import Path
from typing import Dict, List, Any
import logging

from utils import setup_logging

logger = setup_logging(__name__)

class OutcomeAnalyzer:
    def __init__(self, base_path: Path = None):
        self.base_path = base_path or Path.cwd()
        self.data_dir = self.base_path / "outcome_repository" / "data"
        self.cache = {}
        self.logger = logger
    
    def refresh_cache(self, clear: bool = False) -> None:
        """Refresh the cache of outcome data."""
        self.logger.debug("Refreshing outcome cache")
        try:
            if clear:
                self.cache.clear()
                self.logger.debug("Cache cleared")
            
            if not self.data_dir.exists():
                self.logger.warning(f"Data directory {self.data_dir} does not exist")
                return
            
            for file_path in self.data_dir.glob("cycle_*.json"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        cycle_data = json.load(f)
                    cycle_id = cycle_data.get("cycle_id")
                    if cycle_id:
                        self.cache[cycle_id] = cycle_data
                    self.logger.debug(f"Loaded cycle {cycle_id} into cache")
                except Exception as e:
                    self.logger.warning(f"Error loading {file_path}: {str(e)}")
            self.logger.info(f"Cached {len(self.cache)} cycles")
        except Exception as e:
            self.logger.error(f"Error refreshing cache: {str(e)}")
    
    def get_transformation_stats(self) -> Dict[str, Any]:
        """Get statistics on transformation outcomes."""
        self.logger.debug("Generating transformation stats")
        stats = {
            "total_cycles": 0,
            "successful_cycles": 0,
            "failed_cycles": 0,
            "transformations": {}
        }
        try:
            if not self.cache:
                self.refresh_cache()
            
            for cycle_data in self.cache.values():
                stats["total_cycles"] += 1
                status = cycle_data.get("status", "unknown")
                if status == "success":
                    stats["successful_cycles"] += 1
                elif status == "error":
                    stats["failed_cycles"] += 1
                
                for transform in cycle_data.get("transformations", []):
                    transform_type = transform.get("goal", {}).get("type", "unknown")
                    if transform_type not in stats["transformations"]:
                        stats["transformations"][transform_type] = {
                            "total": 0,
                            "success": 0,
                            "error": 0
                        }
                    stats["transformations"][transform_type]["total"] += 1
                    if transform.get("status") == "success":
                        stats["transformations"][transform_type]["success"] += 1
                    elif transform.get("status") == "error":
                        stats["transformations"][transform_type]["error"] += 1
            
            self.logger.info(f"Generated stats: {stats['total_cycles']} cycles analyzed")
            return stats
        except Exception as e:
            self.logger.error(f"Error generating stats: {str(e)}")
            return stats