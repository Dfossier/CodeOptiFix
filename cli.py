import argparse
import logging
import sys
import asyncio
from pathlib import Path
from typing import Dict
from core.model import ModelFactory
from core.analyzer import CodeAnalyzer
from core.file_handler import FileHandler
from config import config_manager  # Using ConfigManager

logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the CodeOptiFix CLI."""
    parser = argparse.ArgumentParser(description="CodeOptiFix: AI-driven package optimization.")
    parser.add_argument("--dir", default=str(config_manager.get("HOME_DIR")), help="Directory to analyze")
    parser.add_argument("--apply", action="store_true", help="Apply proposed changes")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without applying")
    return parser.parse_args()

async def run_cli_async(args: argparse.Namespace, model, file_handler: FileHandler, analyzer: CodeAnalyzer) -> None:
    """Run the CLI workflow asynchronously."""
    try:
        test_prompt = "Ping"
        logger.info("Testing API responsiveness")
        model.generate(test_prompt)
        
        files = [Path(f) for f in file_handler.scan()]
        logger.info(f"Scanned files: {[str(f) for f in files]}")
        if not files:
            logger.warning(f"No files found in {args.dir}")
            print(f"No files found in {args.dir}")
            return

        if file_handler.proposals_file.exists() and not args.apply and not args.dry_run:
            file_handler.proposals_file.unlink()
            logger.info(f"Cleared previous proposals: {file_handler.proposals_file}")

        await analyzer.process_package(file_handler, [str(f) for f in files])  # Direct await, no asyncio.to_thread
        proposed_files = analyzer.read_latest_proposal(file_handler)

        if not proposed_files:
            logger.warning("No valid proposals generated")
            print("No valid proposals generated")
            return

        if args.dry_run:
            print("\nDry run: Proposed changes:")
            for filename, code in proposed_files.items():
                print(f"\n# {filename}\n{code}")
            return

        if args.apply:
            for filename, code in proposed_files.items():
                success = file_handler.apply_proposals(filename, code)
                status = "✅ Applied" if success else "❌ Failed"
                print(f"{status} changes to {filename}")

        logger.info(f"Package analysis complete: {file_handler.proposals_file}")
        print(f"\nAnalysis complete. Review proposals in {file_handler.proposals_file}")
    except Exception as e:
        logger.error(f"Execution error: {e}", exc_info=True)
        print(f"Error: {e}")
        sys.exit(1)

def run_cli(args=None) -> None:
    """Main entry point for the CLI."""
    args = args or parse_args()
    if not config_manager.get("DEEPSEEK_API_KEY"):
        logger.warning("DeepSeek API key not found in .env file")
        print("DeepSeek API key not found. Please create a .env file with DEEPSEEK_API_KEY.")
        sys.exit(1)
    
    try:
        model = ModelFactory.create_model({
            'model_type': config_manager.get('MODEL_TYPE'),
            'api_key': config_manager.get('DEEPSEEK_API_KEY')
        })
        file_handler = FileHandler(args.dir, config_manager.get('SUPPORTED_EXTENSIONS'))
        analyzer = CodeAnalyzer(model, config_manager.get('PROMPTS'))
        asyncio.run(run_cli_async(args, model, file_handler, analyzer))
    except Exception as e:
        logger.error(f"CLI execution error: {e}", exc_info=True)
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_cli()