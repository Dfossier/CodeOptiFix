# core/model.py
from abc import ABC, abstractmethod
from typing import Dict, Type
import requests
import logging

logger = logging.getLogger(__name__)

class BaseModel(ABC):
    """Abstract base class for all models."""
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass

class ModelRegistry:
    """Registry for model types."""
    _models: Dict[str, Type[BaseModel]] = {}

    @classmethod
    def register(cls, model_type: str):
        def decorator(model_cls: Type[BaseModel]):
            cls._models[model_type] = model_cls
            return model_cls
        return decorator

    @classmethod
    def create(cls, model_type: str, config: Dict) -> BaseModel:
        if model_type not in cls._models:
            raise ValueError(f"Unknown model type: {model_type}")
        # Only pass relevant config keys to the model constructor
        model_config = {k: v for k, v in config.items() if k != 'model_type'}
        return cls._models[model_type](**model_config)

class DeepSeekModel(BaseModel):
    """DeepSeek API model implementation."""
    def __init__(self, api_key: str, api_base: str = "https://api.deepseek.com"):
        self.api_key = api_key
        self.api_base = api_base
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def generate(self, prompt: str) -> str:
        """Generate text using the DeepSeek API."""
        payload = {
            "model": "deepseek-chat",  # Updated model name (adjust if needed)
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 8192,
            "temperature": 0.7
        }
        try:
            response = requests.post(f"{self.api_base}/v1/chat/completions", json=payload, headers=self.headers)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"DeepSeek API error: {e}")
            raise

# Register the DeepSeekModel
ModelRegistry.register("deepseek")(DeepSeekModel)

class ModelFactory:
    """Factory to create model instances."""
    @staticmethod
    def create_model(config: Dict) -> BaseModel:
        model_type = config.get("model_type", "deepseek").lower()
        return ModelRegistry.create(model_type, config)