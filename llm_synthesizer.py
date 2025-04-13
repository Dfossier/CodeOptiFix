"""
LLM Synthesizer module for the Self-Improving AI Assistant Update Generator.

Generates candidate code improvements using a Large Language Model.
"""
import json
import os
import time
import datetime
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import asyncio
from openai import AsyncOpenAI

import config
import utils
from utils import LLMSynthesisError, setup_logging
from interfaces import ImprovementGoal, CodeCandidate

logger = setup_logging(__name__)

class LLMSynthesizer:
    """Generates code improvements using LLM."""
    
    def __init__(self):
        """Initialize the LLM synthesizer."""
        self.logger = logger
        self.model = config.LLM_MODEL
        self.temperature = config.LLM_TEMPERATURE
        self.max_tokens = config.LLM_MAX_TOKENS
        self.retry_count = config.LLM_RETRY_COUNT
        
        self.logger.info(f"Initialized LLM Synthesizer with model {self.model}")
        self.logger.debug(f"Config LLM_MODEL value: {config.LLM_MODEL}")
    
    async def generate_candidates(
        self, 
        goal: ImprovementGoal, 
        context: Dict[str, Any],
        num_candidates: int = 1
    ) -> List[CodeCandidate]:
        """
        Generate code improvement candidates for a given goal.
        
        Args:
            goal: The improvement goal
            context: Context information about the code to improve
            num_candidates: Number of candidates to generate
            
        Returns:
            List of CodeCandidate objects
        """
        self.logger.info(
            f"Generating {num_candidates} candidates for goal: {goal.description}"
        )
        
        prompt = self._build_prompt(goal, context)
        tasks = [self._generate_single_candidate(prompt, i) for i in range(num_candidates)]
        candidates = await asyncio.gather(*tasks)
        
        self.logger.info(f"Generated {len(candidates)} candidates")
        return candidates
    
    async def _generate_single_candidate(
        self, prompt: str, candidate_index: int
    ) -> CodeCandidate:
        """Generate a single code candidate."""
        adjusted_temp = self.temperature + (candidate_index * 0.05)
        
        for attempt in range(self.retry_count):
            try:
                self.logger.debug(
                    f"Attempt {attempt+1}/{self.retry_count} for candidate {candidate_index}"
                )
                
                response = await self._call_llm(prompt, adjusted_temp)
                parsed = self._parse_llm_response(response)
                
                return CodeCandidate(
                    code=parsed["code"],
                    comments=parsed.get("comments", ""),
                    metadata={
                        "model": self.model,
                        "temperature": adjusted_temp,
                        "attempt": attempt + 1,
                        "timestamp": time.time()
                    }
                )
                
            except Exception as e:
                self.logger.warning(
                    f"Error generating candidate {candidate_index}, "
                    f"attempt {attempt+1}: {str(e)}"
                )
                if attempt == self.retry_count - 1:
                    raise LLMSynthesisError(
                        f"Failed to generate candidate after {self.retry_count} attempts: {str(e)}"
                    )
        
        raise LLMSynthesisError("Unexpected error in candidate generation")
    
    def _build_prompt(self, goal: ImprovementGoal, context: Dict[str, Any]) -> str:
        """Build a prompt for the LLM based on the goal and context."""
        target_module = context.get("module_info", {})
        target_function = context.get("function_info", {})
        
        prompt = [
            "You are an expert Python developer tasked with improving code.",
            "Your goal is to generate improved code based on the following requirements:",
            f"GOAL: {goal.description}"
        ]
        
        if goal.performance_target:
            prompt.append(f"PERFORMANCE TARGET: {goal.performance_target}")
        
        if target_module:
            prompt.append("\n## MODULE CONTEXT")
            prompt.append(f"Module: {target_module.get('module_path', 'unknown')}")
            imports = target_module.get("imports", [])
            if imports:
                prompt.append("\n### IMPORTS")
                for imp in imports:
                    if imp["type"] == "import":
                        prompt.append(f"import {imp['name']}")
                    else:
                        prompt.append(f"from {imp['module']} import {imp['name']}")
        
        if target_function:
            prompt.append("\n## FUNCTION TO IMPROVE")
            if "ast_node" in target_function:
                start_line = target_function["ast_node"].lineno
                end_line = target_function.get("complexity", {}).get("line_count", 0)
                if "content" in target_module and start_line <= len(target_module["content"].split("\n")):
                    func_code = "\n".join(target_module["content"].split("\n")[start_line-1:end_line])
                    prompt.append(f"```python\n{func_code}\n```")
            complexity = target_function.get("complexity", {})
            if complexity:
                prompt.append("\n### COMPLEXITY METRICS")
                prompt.append(f"- Cyclomatic Complexity: {complexity.get('cyclomatic_complexity', 'N/A')}")
                prompt.append(f"- Line Count: {complexity.get('line_count', 'N/A')}")
                prompt.append(f"- Return Count: {complexity.get('return_count', 'N/A')}")
                prompt.append(f"- Branch Count: {complexity.get('branch_count', 'N/A')}")
        
        prompt.append("\n## INSTRUCTIONS")
        prompt.append("1. Analyze the current code and identify areas for improvement.")
        prompt.append("2. Generate improved code that addresses the specified goal.")
        prompt.append("3. Explain your changes and reasoning.")
        prompt.append("\n## RESPONSE FORMAT")
        prompt.append("```python")
        prompt.append("# Improved code here")
        prompt.append("```")
        prompt.append("\nExplanation of changes:")
        prompt.append("- Point 1")
        prompt.append("- Point 2")
        
        return "\n".join(prompt)
    
    async def _call_llm(self, prompt: str, temperature: float) -> str:
        """
        Make an API call to the LLM API.
        
        Args:
            prompt: The prompt to send to the LLM
            temperature: The temperature setting for generation
            
        Returns:
            The LLM response text
        
        Raises:
            LLMSynthesisError: If there is an error calling the LLM
        """
        self.logger.info(f"Calling LLM API with temperature {temperature}")
        api_url = getattr(config, 'API_URL', 'https://api.deepseek.com')
        api_key = getattr(config, 'API_KEY', None)
        self.logger.debug(f"API base URL: {api_url}")
        self.logger.debug(f"Model: {self.model}")
        
        # Only log key preview at DEBUG level
        if self.logger.level <= logging.DEBUG and api_key:
            self.logger.debug(f"API key preview: {api_key[:4]}...{api_key[-4:] if api_key else 'None'}")
            
        if not api_key:
            self.logger.warning("No API key provided, using demo response")
            return self._get_demo_response(prompt)
        
        try:
            client = AsyncOpenAI(api_key=api_key, base_url=api_url)
            data = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert Python developer specialized in optimizing and improving code. Generate clear, efficient, and well-documented code improvements. Focus on specific improvements requested by the user. Your output should include well-formatted Python code blocks with ```python tags and separate explanation of changes."
                    },
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature,
                "max_tokens": self.max_tokens,
                "stream": False,
                "top_p": 0.95,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0
            }
            
            # Only log full payload at DEBUG level
            if self.logger.level <= logging.DEBUG:
                self.logger.debug("Full request payload: [omitted for brevity]")
                
            self.logger.info("Sending request to API...")
            response = await client.chat.completions.create(**data)
            self.logger.info("API request successful")
            
            # Move detailed response logging to DEBUG level
            self.logger.debug(f"Response type: {type(response)}")
            
            # Only preview content at INFO level
            self.logger.info(f"Response content preview: {response.choices[0].message.content[:150]}...")
            return response
            
        except asyncio.TimeoutError:
            raise LLMSynthesisError("API request timed out")
        except Exception as e:
            raise LLMSynthesisError(f"Error connecting to DeepSeek API: {str(e)}")
    
    def _parse_llm_response(self, response) -> Dict[str, str]:
        """Parse the LLM response to extract code and comments."""
        try:
            if hasattr(response, 'choices') and response.choices:
                content = response.choices[0].message.content.strip()
                self.logger.debug(f"Raw response content: {content[:100]}...")  # Log first 100 chars
            else:
                # Handle case where response is a string (like in demo mode)
                content = response if isinstance(response, str) else ""
                self.logger.debug("Processing raw string response")
            
            # Extract code block between ```python and ```
            code_match = re.search(r'```python\s*(.*?)\s*```', content, re.DOTALL)
            if not code_match:
                self.logger.warning(f"No code block found in LLM response. Response preview: {content[:150]}...")
                # Return a minimal valid response if no code block is found
                return {
                    "code": "# No valid code found in response\npass",
                    "comments": "Failed to extract code from LLM response"
                }
            
            code = code_match.group(1).strip()
            
            # Extract explanation after "Explanation of changes:"
            explanation_match = re.search(r'Explanation of changes:(.*)', content, re.DOTALL)
            comments = explanation_match.group(1).strip() if explanation_match else ""
            
            return {
                "code": code,
                "comments": comments
            }
        except Exception as e:
            self.logger.error(f"Error parsing LLM response: {str(e)}")
            # Return a fallback response to prevent crashing
            return {
                "code": "# Error parsing response\npass",
                "comments": f"Error parsing LLM response: {str(e)}"
            }
    
    def _get_demo_response(self, prompt: str) -> str:
        """
        Get a demonstration response when API key is not available.
        This is just for testing purposes.
        """
        goal_info = ""
        if "GOAL:" in prompt:
            goal_parts = prompt.split("GOAL:")
            if len(goal_parts) > 1:
                goal_info = goal_parts[1].split("\n")[0].strip()
        
        return f"""```python
# Improved implementation addressing: {goal_info}
def improved_function(a, b, c):
    \"\"\"
    Improved implementation with better performance and reliability.
    
    Args:
        a: First parameter
        b: Second parameter
        c: Denominator (must be non-zero)
        
    Returns:
        Calculated result with optimizations applied
    \"\"\"
    # Optimize calculation
    result = a + b
    
    # Add error handling for division by zero
    if c == 0:
        logger.warning("Division by zero prevented")
        return result
    
    # Apply weekend optimization if applicable
    if datetime.datetime.now().weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        logger.info("Applying weekend adjustment")
        return result * 0.9
    
    return result / c
```

Explanation of changes:
- Added proper docstring with type information and parameter descriptions
- Implemented weekend detection using datetime module
- Added logging instead of print statements for error cases
- Improved error handling for the division by zero case
- Made code more maintainable with descriptive comments
"""