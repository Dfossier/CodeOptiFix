import argparse
from config import HOME_DIR

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the CodeOptiFix CLI."""
    parser = argparse.ArgumentParser(description="CodeOptiFix: AI-driven package optimization.")
    parser.add_argument("--dir", default=str(HOME_DIR), help="Directory to analyze")
    parser.add_argument("--apply", action="store_true", help="Apply proposed changes")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without applying")
    return parser.parse_args()