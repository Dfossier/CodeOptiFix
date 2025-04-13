"""
State Persistence Layer for Code State Observatory.

Stores codebase snapshots with tagged metadata for historical comparison,
providing insights into code evolution over time.
"""
import os
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Set, Any, Optional, Union
from datetime import datetime
import logging
import shutil

from utils import setup_logging

logger = setup_logging(__name__)

class StatePersistence:
    """Manages persistent storage of codebase state snapshots."""
    
    def __init__(self, base_path: Optional[Path] = None, storage_path: Optional[Path] = None):
        """Initialize the state persistence layer."""
        self.base_path = base_path or Path.cwd()
        self.storage_path = storage_path or (self.base_path / "observatory" / "data" / "snapshots")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.logger = logger
        self.current_snapshot = None
        
    def create_snapshot(self, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a snapshot of the current codebase state.
        
        Args:
            metadata: Optional metadata to associate with the snapshot
            
        Returns:
            ID of the created snapshot
        """
        self.logger.info("Creating codebase snapshot...")
        
        # Generate snapshot ID (timestamp + random suffix)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_id = f"{timestamp}_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"
        
        # Create snapshot directory
        snapshot_dir = self.storage_path / snapshot_id
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a metadata file
        metadata = metadata or {}
        metadata.update({
            "snapshot_id": snapshot_id,
            "timestamp": timestamp,
            "created_at": datetime.now().isoformat()
        })
        
        with open(snapshot_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        
        # Store file hashes and selective content
        file_data = {}
        
        # Find Python files in the codebase
        python_files = list(self.base_path.glob("**/*.py"))
        
        for file_path in python_files:
            # Skip files that should be ignored
            if self._should_skip_file(str(file_path)):
                continue
                
            # Calculate file hash
            file_hash = self._hash_file(file_path)
            
            # Get relative path
            rel_path = str(file_path.relative_to(self.base_path))
            
            # Store file info
            file_data[rel_path] = {
                "hash": file_hash,
                "size": file_path.stat().st_size,
                "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
            }
            
            # Store complete file content for important files
            # (smaller files or those matching certain patterns)
            if (file_path.stat().st_size < 50 * 1024 or  # Less than 50KB
                any(pattern in rel_path for pattern in ["config", "interface", "main"])):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    file_data[rel_path]["content"] = content
                except Exception as e:
                    self.logger.warning(f"Error reading {rel_path}: {str(e)}")
        
        # Save file data
        with open(snapshot_dir / "files.json", "w", encoding="utf-8") as f:
            json.dump(file_data, f, indent=2)
        
        self.logger.info(f"Created snapshot {snapshot_id} with {len(file_data)} files")
        self.current_snapshot = snapshot_id
        
        return snapshot_id
    
    def _should_skip_file(self, file_path: str) -> bool:
        """Determine if a file should be skipped during analysis."""
        skip_patterns = [
            '.venv/', 
            'site-packages/',
            '.git/',
            '__pycache__/',
            '.pytest_cache/'
        ]
        return any(pattern in file_path for pattern in skip_patterns)
    
    def _hash_file(self, file_path: Path) -> str:
        """Generate a hash of the file contents."""
        with open(file_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def get_snapshot(self, snapshot_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get snapshot data.
        
        Args:
            snapshot_id: ID of the snapshot to get, or None for the latest
            
        Returns:
            Dict containing snapshot data
        """
        # If no snapshot ID is provided, use the latest
        if not snapshot_id:
            snapshots = self.list_snapshots()
            if not snapshots:
                return {}
            snapshot_id = snapshots[0]["id"]
        
        snapshot_dir = self.storage_path / snapshot_id
        
        if not snapshot_dir.exists():
            self.logger.warning(f"Snapshot {snapshot_id} not found")
            return {}
        
        try:
            # Load metadata
            with open(snapshot_dir / "metadata.json", "r", encoding="utf-8") as f:
                metadata = json.load(f)
            
            # Load file data
            with open(snapshot_dir / "files.json", "r", encoding="utf-8") as f:
                files = json.load(f)
            
            return {
                "metadata": metadata,
                "files": files
            }
        except Exception as e:
            self.logger.error(f"Error loading snapshot {snapshot_id}: {str(e)}")
            return {}
    
    def list_snapshots(self) -> List[Dict[str, Any]]:
        """
        List all available snapshots.
        
        Returns:
            List of snapshot metadata
        """
        snapshots = []
        
        for snapshot_dir in self.storage_path.iterdir():
            if snapshot_dir.is_dir():
                metadata_file = snapshot_dir / "metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, "r", encoding="utf-8") as f:
                            metadata = json.load(f)
                        
                        snapshots.append({
                            "id": snapshot_dir.name,
                            "timestamp": metadata.get("timestamp", ""),
                            "created_at": metadata.get("created_at", ""),
                            "metadata": metadata
                        })
                    except Exception as e:
                        self.logger.warning(f"Error loading metadata for {snapshot_dir.name}: {str(e)}")
        
        # Sort by creation time, newest first
        snapshots.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        
        return snapshots
    
    def compare_snapshots(self, snapshot_id1: str, snapshot_id2: str) -> Dict[str, Any]:
        """
        Compare two snapshots to identify changes.
        
        Args:
            snapshot_id1: ID of the first snapshot
            snapshot_id2: ID of the second snapshot
            
        Returns:
            Dict containing comparison results
        """
        # Load snapshots
        snapshot1 = self.get_snapshot(snapshot_id1)
        snapshot2 = self.get_snapshot(snapshot_id2)
        
        if not snapshot1 or not snapshot2:
            return {
                "error": "One or both snapshots not found"
            }
        
        files1 = snapshot1.get("files", {})
        files2 = snapshot2.get("files", {})
        
        # Find added, removed, and modified files
        files_added = [path for path in files2 if path not in files1]
        files_removed = [path for path in files1 if path not in files2]
        files_modified = [
            path for path in files1 
            if path in files2 and files1[path].get("hash") != files2[path].get("hash")
        ]
        
        # Calculate detailed changes for modified files that have content
        detailed_changes = {}
        for path in files_modified:
            if ("content" in files1.get(path, {}) and 
                "content" in files2.get(path, {})):
                detailed_changes[path] = self._diff_content(
                    files1[path]["content"],
                    files2[path]["content"]
                )
        
        return {
            "snapshot1": {
                "id": snapshot_id1,
                "timestamp": snapshot1.get("metadata", {}).get("timestamp", "")
            },
            "snapshot2": {
                "id": snapshot_id2,
                "timestamp": snapshot2.get("metadata", {}).get("timestamp", "")
            },
            "summary": {
                "files_added": len(files_added),
                "files_removed": len(files_removed),
                "files_modified": len(files_modified)
            },
            "details": {
                "files_added": files_added,
                "files_removed": files_removed,
                "files_modified": files_modified
            },
            "detailed_changes": detailed_changes
        }
    
    def _diff_content(self, content1: str, content2: str) -> Dict[str, Any]:
        """
        Generate a simplified diff between two content strings.
        
        Args:
            content1: Original content
            content2: New content
            
        Returns:
            Dict containing diff information
        """
        try:
            import difflib
            
            # Split content into lines
            lines1 = content1.splitlines()
            lines2 = content2.splitlines()
            
            # Calculate diff
            diff = list(difflib.unified_diff(
                lines1, lines2, 
                lineterm="",
                n=3  # Show 3 lines of context
            ))
            
            # Count added and removed lines
            added_lines = len([line for line in diff if line.startswith("+")])
            removed_lines = len([line for line in diff if line.startswith("-")])
            
            return {
                "diff": "\n".join(diff),
                "added_lines": added_lines,
                "removed_lines": removed_lines,
                "changed_ratio": (added_lines + removed_lines) / max(len(lines1), 1)
            }
        except ImportError:
            # Fallback if difflib is not available
            return {
                "length_diff": len(content2) - len(content1),
                "changed": content1 != content2
            }
    
    def get_file_history(self, file_path: Union[str, Path], limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get historical versions of a specific file.
        
        Args:
            file_path: Path to the file
            limit: Maximum number of historical versions to return
            
        Returns:
            List of historical file versions
        """
        if isinstance(file_path, Path):
            file_path = str(file_path.relative_to(self.base_path))
        
        snapshots = self.list_snapshots()
        history = []
        
        for snapshot_data in snapshots[:limit]:
            snapshot_id = snapshot_data["id"]
            snapshot = self.get_snapshot(snapshot_id)
            
            if not snapshot:
                continue
                
            files = snapshot.get("files", {})
            
            if file_path in files:
                file_info = files[file_path]
                history.append({
                    "snapshot_id": snapshot_id,
                    "timestamp": snapshot_data.get("timestamp", ""),
                    "hash": file_info.get("hash", ""),
                    "content": file_info.get("content", None),
                    "size": file_info.get("size", 0),
                    "modified": file_info.get("modified", "")
                })
        
        return history
    
    def clean_old_snapshots(self, max_snapshots: int = 10) -> int:
        """
        Clean up old snapshots to save disk space.
        
        Args:
            max_snapshots: Maximum number of snapshots to keep
            
        Returns:
            Number of snapshots deleted
        """
        snapshots = self.list_snapshots()
        
        if len(snapshots) <= max_snapshots:
            return 0
        
        # Delete oldest snapshots
        snapshots_to_delete = snapshots[max_snapshots:]
        deleted_count = 0
        
        for snapshot in snapshots_to_delete:
            snapshot_id = snapshot["id"]
            snapshot_dir = self.storage_path / snapshot_id
            
            try:
                shutil.rmtree(snapshot_dir)
                deleted_count += 1
                self.logger.info(f"Deleted old snapshot: {snapshot_id}")
            except Exception as e:
                self.logger.error(f"Error deleting snapshot {snapshot_id}: {str(e)}")
        
        return deleted_count