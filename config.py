# config.py
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv
import os
import json

class ConfigManager:
    def __init__(self, env_file: str = ".env"):
        load_dotenv(env_file)
        self.home_dir = Path(__file__).parent
        self.defaults = {
            "MODEL_TYPE": "deepseek",
            "MODEL_ID": "deepseek-reasoner",
            "DEEPSEEK_API_KEY": "",
            "DEEPSEEK_API_BASE": "https://api.deepseek.com",
            "MAX_TOKENS": 8192,
            "SUPPORTED_EXTENSIONS": (".py", ".js", ".cpp"),
            "LOCAL_MODEL_ID": "TheBloke/deepseek-coder-6.7B-base-AWQ",
            "LOCAL_MODEL_PATH": self.home_dir / "deepseek-coder-6.7B-base-AWQ",
            "HOME_DIR": self.home_dir  # Added HOME_DIR explicitly
        }
        self.config: Dict[str, Any] = {}

    def load(self) -> None:
        for key, default in self.defaults.items():
            self.config[key] = os.getenv(key, default)
        self._validate()
        self.config["PROMPTS"] = self._load_prompts()

    def _validate(self) -> None:
        if self.config["MODEL_TYPE"] == "deepseek" and not self.config["DEEPSEEK_API_KEY"]:
            raise ValueError("DEEPSEEK_API_KEY required for deepseek model")

    def _load_prompts(self) -> Dict[str, str]:
        prompts_file = self.home_dir / "prompts.json"
        try:
            with open(prompts_file, "r") as f:
                prompts = json.load(f)
            required = {"analyze", "assess", "propose"}
            if missing := required - set(prompts.keys()):
                raise ValueError(f"Missing prompt keys: {missing}")
            return prompts
        except Exception as e:
            raise ValueError(f"Failed to load prompts: {e}")

    def get(self, key: str) -> Any:
        return self.config.get(key, self.defaults.get(key))

config_manager = ConfigManager()
config_manager.load()