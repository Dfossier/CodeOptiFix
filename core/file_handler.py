# core/file_handler.py
import os
import logging
import re
from pathlib import Path
from datetime import datetime
from tempfile import NamedTemporaryFile
from typing import List, Optional, Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FileHandler:
    """Handles file operations for the code analyzer."""
    
    def __init__(self, base_dir: str, extensions: tuple):
        self.base_dir = Path(base_dir).resolve()
        self.extensions = extensions  # Already a tuple from config
        self.proposals_file = self.base_dir / "proposals.txt"
        self.code_block_re = re.compile(r"```python\n(.*?)\n```", re.DOTALL)
        self.proposed_code_re = re.compile(r"Proposed Code:\n(.*?)(?:====================|$)", re.DOTALL)

    def _read_file(self, file_path: Path) -> str:
        try:
            return file_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return ""

    def _write_file(self, file_path: Path, content: str, append: bool = False) -> bool:
        try:
            mode = "a" if append else "w"
            with open(file_path, mode, encoding="utf-8") as f:
                f.write(content)
            return True
        except Exception as e:
            logger.error(f"Error writing to {file_path}: {e}")
            return False

    def scan(self) -> List[str]:
        """Scan the base directory for code files."""
        code_files = []
        try:
            ignored_dirs = {"venv", ".git", "__pycache__", "node_modules"}
            for root, _, files in os.walk(self.base_dir):
                if any(ignored in Path(root).parts for ignored in ignored_dirs):
                    continue
                for file in files:
                    if file.endswith(self.extensions):  # Works with tuple
                        code_files.append(str(Path(root) / file))
            logger.info(f"Found {len(code_files)} files to analyze")
            return code_files
        except Exception as e:
            logger.error(f"Error scanning directory {self.base_dir}: {e}")
            return []

    def read_file(self, file_path: str) -> str:
        return self._read_file(Path(file_path))

    def write_proposal(self, file_path: str, analysis: str, 
                      assessment: str, suggestions: str, 
                      proposed_code: str) -> None:
        file_path = Path(file_path).resolve()
        content = (
            f"\n=== {file_path} ===\n"
            f"Analysis:\n{analysis}\n\n"
            f"Assessment:\n{assessment}\n\n"
            f"Suggestions:\n{suggestions}\n\n"
            f"Proposed Code:\n```python\n{proposed_code}\n```\n"
            "====================\n"
        )
        if self._write_file(self.proposals_file, content, append=True):
            logger.info(f"Proposal for {file_path} written to {self.proposals_file}")

    def read_latest_proposal(self, file_path: str) -> Optional[str]:
        file_path = Path(file_path).resolve()
        try:
            if not self.proposals_file.exists():
                logger.warning(f"Proposals file does not exist: {self.proposals_file}")
                return None
            
            content = self._read_file(self.proposals_file)
            file_path_escaped = re.escape(str(file_path))
            pattern = rf"=== {file_path_escaped} ===\n(.*?)====================\n"
            matches = re.findall(pattern, content, re.DOTALL)
            
            if not matches:
                logger.warning(f"No proposal found for file: {file_path}")
                return None
                
            latest_section = matches[-1]
            
            code_block_match = self.code_block_re.search(latest_section)
            if code_block_match:
                return code_block_match.group(1).strip()
                
            code_match = self.proposed_code_re.search(latest_section)
            if code_match:
                content = code_match.group(1).strip()
                inner_code_match = self.code_block_re.search(content)
                if inner_code_match:
                    return inner_code_match.group(1).strip()
                return content
                
            logger.warning(f"Proposal for {file_path} lacks clear code block. Manual review recommended.")
            return None
        except Exception as e:
            logger.error(f"Error reading proposal for {file_path}: {e}")
            return None

    def apply_proposals(self, file_path: str, proposed_code: str) -> bool:
        file_path = Path(file_path).resolve()
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = file_path.with_name(f"{file_path.stem}_{timestamp}.bak")
            
            last_backup = max((f for f in file_path.parent.glob(f"{file_path.stem}_*.bak")), 
                             key=lambda x: x.stat().st_mtime, default=None)
            if last_backup and file_path.stat().st_mtime > last_backup.stat().st_mtime:
                logger.warning(f"{file_path} modified since last backup. Creating new backup.")
            
            original_content = self._read_file(file_path)
            if not self._write_file(backup_path, original_content):
                return False
                
            with NamedTemporaryFile(mode='w', encoding='utf-8', delete=False) as tmp:
                tmp.write(proposed_code)
                tmp_path = Path(tmp.name)
            os.replace(tmp_path, file_path)
            
            logger.info(f"Applied changes to {file_path} (backup at {backup_path})")
            return True
        except Exception as e:
            logger.error(f"Error applying proposals to {file_path}: {e}")
            return False

    def save_proposed_version(self, file_path: str, proposed_code: str) -> str:
        file_path = Path(file_path).resolve()
        try:
            stem = file_path.stem
            suffix = file_path.suffix
            parent = file_path.parent
            version_path = parent / f"{stem}_optimized{suffix}"
            
            if self._write_file(version_path, proposed_code):
                logger.info(f"Saved proposed version to {version_path}")
                return str(version_path)
            return ""
        except Exception as e:
            logger.error(f"Error saving proposed version for {file_path}: {e}")
            return ""

    def get_latest_backup(self, file_path: str) -> Optional[Path]:
        file_path = Path(file_path).resolve()
        backups = sorted(file_path.parent.glob(f"{file_path.stem}_*.bak"), 
                        key=lambda x: x.stat().st_mtime, reverse=True)
        return backups[0] if backups else None

    def rollback(self, file_path: str) -> bool:
        file_path = Path(file_path).resolve()
        try:
            backup_path = self.get_latest_backup(file_path)
            if not backup_path:
                logger.warning(f"No backup found for {file_path}")
                return False
                
            backup_content = self._read_file(backup_path)
            if self._write_file(file_path, backup_content):
                backup_path.unlink()
                logger.info(f"Rolled back {file_path} from {backup_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error rolling back {file_path}: {e}")
            return False