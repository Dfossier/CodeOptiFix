# core/model.py
import os
import logging
from abc import ABC, abstractmethod
from config import (
    MODEL_TYPE, MODEL_ID, LOCAL_MODEL_ID, LOCAL_MODEL_PATH,
    DEEPSEEK_API_KEY, DEEPSEEK_API_BASE, MAX_TOKENS
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BaseModel(ABC):
    """Abstract base class for model implementations."""
    
    @abstractmethod
    def generate(self, prompt):
        """Generate text based on the given prompt."""
        pass

class MockLocalModel(BaseModel):
    """Mock local model implementation for testing."""
    
    def __init__(self):
        logger.warning("Using mock local model for testing")
        logger.warning("For actual local model inference, install vllm package")
    
    def generate(self, prompt):
        """Generate a mock response based on the given prompt."""
        return "[This is a mock response. Install vllm for local model inference.]"

class DeepSeekModel(BaseModel):
    """DeepSeek API model implementation using OpenAI client."""
    
    def __init__(self, dir_path=None):
        self.api_key = DEEPSEEK_API_KEY
        self.api_base = DEEPSEEK_API_BASE
        self.model_id = MODEL_ID
        self.system_prompt = "You are a helpful AI assistant that provides code solutions."
        
        if not self.api_key:
            raise ValueError("DeepSeek API key not found. Set DEEPSEEK_API_KEY in .env.")
        
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)
            logger.info(f"Using DeepSeek API with model: {self.model_id}")
            self.using_openai_sdk = True
        except ImportError:
            import httpx
            self.httpx = httpx
            logger.info("OpenAI SDK not available, falling back to HTTPX")
            logger.info(f"Using DeepSeek API with model: {self.model_id}")
            self.using_openai_sdk = False
    
    def generate(self, prompt):
        if self.using_openai_sdk:
            return self._generate_with_openai_sdk(prompt)
        return self._generate_with_httpx(prompt)
    
    def _build_api_messages(self, prompt):
        """Build standardized API messages."""
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
    
    def _generate_with_openai_sdk(self, prompt):
        from openai import OpenAIError
        import json
        for attempt in range(2):  # Retry once
            try:
                messages = self._build_api_messages(prompt)
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    max_tokens=MAX_TOKENS,
                    timeout=60  # Explicit 60s timeout
                )
                return response.choices[0].message.content
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error on attempt {attempt + 1}: {e}")
                if attempt == 1:
                    raise ValueError(f"Failed to parse DeepSeek response: {e}")
            except OpenAIError as e:
                logger.error(f"OpenAI SDK error on attempt {attempt + 1}: {e}")
                if attempt == 1:
                    raise ValueError(f"DeepSeek API failed: {e}")
            except Exception as e:
                logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
                if attempt == 1:
                    raise ValueError(f"DeepSeek API call failed: {e}")
        return "Error: Max retries reached"
    
    def _generate_with_httpx(self, prompt):
        try:
            url = f"{self.api_base}/chat/completions"
            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
            payload = {
                "model": self.model_id.split('/')[-1] if '/' in self.model_id else self.model_id,
                "messages": self._build_api_messages(prompt),
                "max_tokens": MAX_TOKENS
            }
            timeout = self.httpx.Timeout(connect=10.0, read=60.0)
            with self.httpx.Client(timeout=timeout) as client:
                logger.info(f"Sending request to DeepSeek API: {url}")
                response = client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Error calling DeepSeek API directly: {e}")
            return f"Error: {e}"

try:
    from vllm import LLM, SamplingParams
    HAS_VLLM = True
    
    class LocalModel(BaseModel):
        """Local model implementation using vLLM."""
        
        def __init__(self, dir_path=None):
            weights_exist = False
            if LOCAL_MODEL_PATH.exists():
                weights_exist = any(f.suffix in {'.safetensors', '.bin'} for f in LOCAL_MODEL_PATH.glob('*'))
            model_location = LOCAL_MODEL_PATH if weights_exist else LOCAL_MODEL_ID
            logger.info(f"Loading local model from: {model_location}")
            
            self.llm = LLM(
                model=str(model_location),
                dtype="float16",
                gpu_memory_utilization=0.7,
                max_model_len=4096
            )
            self.sampling_params = SamplingParams(
                temperature=0.2,
                max_tokens=2048,
                repetition_penalty=1.1
            )
        
        def generate(self, prompt):
            outputs = self.llm.generate([prompt], self.sampling_params)
            return outputs[0].outputs[0].text
            
except ImportError:
    HAS_VLLM = False

class CodeModel:
    """Factory class that returns the appropriate model based on configuration."""
    
    def __init__(self, dir_path=None):
        model_type = MODEL_TYPE.lower()
        
        if model_type == "local":
            if HAS_VLLM:
                self.model = LocalModel(dir_path)
            else:
                logger.info("vLLM not available, using mock model")
                self.model = MockLocalModel()
        else:
            self.model = DeepSeekModel(dir_path)
        
        self.dir_path = dir_path
    
    def generate(self, prompt):
        return self.model.generate(prompt)