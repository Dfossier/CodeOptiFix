"""
Configuration settings for the Self-Improving AI Assistant Update Generator.
"""
import os
from pathlib import Path
import platform
import logging

logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    env_path = Path(__file__).resolve().parent / '.env'
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        logger.info(f'Loaded environment variables from {env_path}')
    else:
        logger.info(f'No .env file found at {env_path}')
except ImportError:
    logger.info('python-dotenv not installed. Environment variables will be loaded from system only.')
except Exception as e:
    logger.info(f'Error loading .env file: {e}')

IS_WSL = 'microsoft-standard' in platform.uname().release.lower()
IS_WINDOWS = platform.system() == 'Windows'
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / 'outputs'
LOG_DIR = BASE_DIR / 'logs'
OUTPUT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)
LLM_MODEL = os.environ.get('CODEOPTIFIX_LLM_MODEL', 'deepseek-chat')
LLM_TEMPERATURE = float(os.environ.get('CODEOPTIFIX_LLM_TEMP', '0.2'))
LLM_MAX_TOKENS = int(os.environ.get('CODEOPTIFIX_LLM_MAX_TOKENS', '4096'))
LLM_RETRY_COUNT = 3
API_KEY = os.environ.get('DEEPSEEK_API_KEY', '')
API_URL = os.environ.get('DEEPSEEK_API_URL', 'https://api.deepseek.com')
TIMEOUT_SECONDS = 60
BATCH_SIZE = 10
DEFAULT_OUTPUT_FORMAT = 'json'
LOG_LEVEL = os.environ.get('CODEOPTIFIX_LOG_LEVEL', 'INFO')