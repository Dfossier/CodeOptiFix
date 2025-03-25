# config.py
import os
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

"""Supported file extensions and base directory"""
SUPPORTED_EXTENSIONS = (".py", ".js", ".cpp")
HOME_DIR = Path(__file__).parent

"""Deepseek API configuration"""
MODEL_TYPE = os.getenv("MODEL_TYPE", "deepseek")
MODEL_ID = os.getenv("MODEL_ID", "deepseek-reasoner")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
if MODEL_TYPE == "deepseek" and not DEEPSEEK_API_KEY:
    raise ValueError("DEEPSEEK_API_KEY is required when using deepseek model type")

"""General model settings"""
MAX_TOKENS = 4096

"""Local model configuration"""
LOCAL_MODEL_ID = "TheBloke/deepseek-coder-6.7B-base-AWQ"
LOCAL_MODEL_PATH = Path("/mnt/c/Users/dfoss/Desktop/LocalAIModels/deepseek-coder-6.7B-base-AWQ")

try:
    with open(HOME_DIR / "prompts.json", "r") as f:
        PROMPTS = json.load(f)
except (IOError, json.JSONDecodeError) as e:
    raise ValueError(f"Failed to load prompts: {str(e)}")

required_keys = {"analyze", "assess", "propose"}
missing_keys = required_keys - PROMPTS.keys()
if missing_keys:
    raise ValueError(f"PROMPTS missing required keys: {missing_keys}")