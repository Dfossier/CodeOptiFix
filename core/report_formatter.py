from typing import Dict, List

class ReportFormatter:
    """Formats analysis reports for output."""
    
    def __init__(self, base_dir: str):
        self.base_dir = base_dir

    def format(self, analysis: str, assessment: str, suggestions: str, repairs: Dict[str, List[str]], 
               proposed_files: Dict[str, str], metrics: Dict[str, float]) -> str:
        # Remove duplicates from suggestions
        suggestions = "\n".join(sorted(set(suggestions.splitlines()), key=lambda x: suggestions.find(x)))
        content = (
            f"\n=== Package: {self.base_dir} ===\n"
            f"Analysis:\n{analysis.strip()}\n\n"
            f"Assessment:\n{assessment.strip()}\n\n"
            f"Suggestions:\n{suggestions.strip()}\n\n"
            f"Repairs Applied:\n"
            f"- Missing Components: {', '.join(repairs['missing']) or 'None'}\n"
            f"- Unresolved Imports: {', '.join(repairs['unresolved']) or 'None'}\n\n"
            f"Self-Performance Metrics:\n"
            f"- Analysis Time: {metrics['analysis_time']:.2f}s\n"
            f"- Token Usage: {metrics['token_usage']}\n"
            f"- Success Rate: {metrics['success_rate']:.2%}\n\n"
            "Proposed Code:\n"
        )
        for filename, code in proposed_files.items():
            content += f"# {filename}\n{code.strip()}\n\n"
        content += "====================\n"
        return content