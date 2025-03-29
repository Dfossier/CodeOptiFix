# core/suggestion_generator.py
import logging  # Added import
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)  # Define logger for this module

class SuggestionGenerator:
    """Generates suggestions and formats prompts for code analysis."""
    def __init__(self, model, prompts: Dict[str, str], token_limit: int):
        self.model = model
        self.prompts = prompts
        self.token_limit = token_limit
        self.impact_keywords = {"validation": 3, "complexity": 2, "performance": 2, "cli": 1}

    def format_prompt(self, task: str, code: str, assessment: str = None, suggestions: str = None) -> str:
        """Format prompt for the given task."""
        try:
            prompt = self.prompts[task]
            if task == "sanitize":
                if not assessment:
                    raise ValueError("Assessment required for sanitize")
                formatted = prompt.replace("##ASSESSMENT##", assessment).replace("##CODE##", code)
                if suggestions:
                    ranked = self._rank_suggestions(suggestions)
                    formatted = formatted.replace("##SUGGESTIONS##", "\n".join(f"- {s} (Score: {score})" for s, score in ranked))
            elif task == "propose":
                if not suggestions:
                    raise ValueError("Suggestions required for propose")
                formatted = prompt.replace("##SUGGESTIONS##", suggestions).replace("##CODE##", code)
            else:
                formatted = prompt.replace("##CODE##", code)
            logger.debug(f"Formatted prompt for {task} ({len(formatted)} chars, ~{len(formatted)//4} tokens): {formatted[:500]}...")
            return self.truncate_prompt(formatted)
        except Exception as e:
            logger.error(f"Prompt formatting failed for {task}: {e}")
            raise

    def truncate_prompt(self, prompt: str) -> str:
        """Truncate prompt if it exceeds token limit."""
        if len(prompt) > self.token_limit:
            logger.warning(f"Prompt too long ({len(prompt)}), truncating to {self.token_limit}")
            return prompt[:self.token_limit]
        return prompt

    def _rank_suggestions(self, suggestions: str) -> List[Tuple[str, int]]:
        """Rank suggestions based on impact keywords."""
        lines = suggestions.splitlines()
        ranked = []
        for line in lines:
            if line.strip():
                score = sum(self.impact_keywords.get(kw, 1) for kw in self.impact_keywords if kw in line.lower())
                ranked.append((line.strip(), score))
        return sorted(ranked, key=lambda x: x[1], reverse=True)