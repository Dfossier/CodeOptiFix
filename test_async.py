import logging
logger = logging.getLogger(__name__)

"""
Simple test script to verify async functionality is working correctly.
"""
import asyncio
from pathlib import Path

async def main():
    """Test simple async functionality."""
    logger.info('Starting async test...')
    logger.info('Simulating async operation...')
    await asyncio.sleep(1)
    base_path = Path.cwd()
    files = list(base_path.glob('*.py'))
    logger.info('{}', (f'Found {len(files)} Python files in the current directory',))
    if files:
        logger.info('{}', (f'First 5 files: {[f.name for f in files[:5]]}',))
    logger.info('Test completed successfully!')
if __name__ == '__main__':
    asyncio.run(main())