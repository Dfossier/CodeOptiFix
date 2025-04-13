"""
Goal Processor for Natural Language Goal Descriptions.

Processes high-level natural language goal descriptions into structured improvement goals.
Analyzes goal descriptions to extract sub-goals and creates tracking mechanisms.
"""
import logging
import json
import hashlib
import re
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

import config
from utils import setup_logging, GoalProcessingError
from interfaces import ImprovementGoal
from llm_synthesizer import LLMSynthesizer

logger = setup_logging(__name__)

class GoalProcessor:
    """Processes natural language goal descriptions into structured improvement goals."""
    
    def __init__(self, base_path: Optional[Path] = None):
        """
        Initialize the goal processor.
        
        Args:
            base_path: Base path for storing processed goals
        """
        self.logger = logger
        self.base_path = base_path or Path.cwd()
        self.goals_dir = self.base_path / "processed_goals"
        self.goals_dir.mkdir(exist_ok=True)
        self.llm_synthesizer = LLMSynthesizer()
        
    async def process_text_goal(self, text_file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Process a text file containing a natural language goal description.
        
        Args:
            text_file_path: Path to the text file with goal description
            
        Returns:
            Dict containing processed goal information
        """
        self.logger.info(f"Processing goal from text file: {text_file_path}")
        
        # Read the text file
        try:
            with open(text_file_path, 'r', encoding='utf-8') as f:
                goal_text = f.read().strip()
        except Exception as e:
            self.logger.error(f"Error reading goal file {text_file_path}: {str(e)}")
            raise ValueError(f"Failed to read goal file: {str(e)}")
            
        if not goal_text:
            self.logger.error(f"Empty goal file: {text_file_path}")
            raise ValueError("Goal file is empty")
            
        # Generate a unique ID for this goal
        goal_id = self._generate_goal_id(goal_text)
        
        # Analyze the goal text to extract structured information
        structured_goals = await self._analyze_goal_text(goal_text, goal_id)
        
        # Create a tracking record for this goal
        timestamp = datetime.now().isoformat()
        goal_record = {
            "goal_id": goal_id,
            "timestamp": timestamp,
            "original_text": goal_text,
            "structured_goals": structured_goals,
            "status": "pending",
            "attempts": 0,
            "sub_goals_completed": 0,
            "sub_goals_total": len(structured_goals),
            "logs": []
        }
        
        # Save the goal record
        output_path = self.goals_dir / f"goal_{goal_id}_{timestamp.replace(':', '-')}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(goal_record, f, indent=2)
            
        self.logger.info(f"Processed goal saved to {output_path}")
        return goal_record
    
    async def _analyze_goal_text(self, goal_text: str, goal_id: str) -> List[Dict[str, Any]]:
        """
        Analyze natural language goal text to extract structured sub-goals.
        
        Args:
            goal_text: Natural language goal description
            goal_id: Unique ID for this goal
            
        Returns:
            List of structured sub-goals
        """
        self.logger.info("Analyzing goal text to extract structured sub-goals")
        
        # Create a prompt for the LLM to extract structured goals
        prompt = self._create_analysis_prompt(goal_text)
        
        try:
            # Use the LLM synthesizer to analyze the goal text
            # Create a simplified context to avoid confusing the LLM with too much code context
            context = {
                "goal_text": goal_text,
                "goal_id": goal_id
            }
            
            # Generate a single candidate (we only need one analysis)
            candidates = await self.llm_synthesizer.generate_candidates(
                goal=ImprovementGoal(
                    target_module="",
                    description="Analyze natural language goal and extract structured sub-goals"
                ),
                context=context,
                num_candidates=1
            )
            
            if not candidates:
                self.logger.warning("No candidates generated for goal analysis")
                return self._create_default_subgoals(goal_text, goal_id)
                
            candidate = candidates[0]
            structured_goals = self._parse_candidate_output(candidate.code, goal_id)
            
            # If parsing failed, create default sub-goals
            if not structured_goals:
                self.logger.warning("Failed to parse candidate output for structured goals")
                return self._create_default_subgoals(goal_text, goal_id)
                
            return structured_goals
            
        except Exception as e:
            self.logger.error(f"Error analyzing goal text: {str(e)}")
            return self._create_default_subgoals(goal_text, goal_id)
    
    def _create_analysis_prompt(self, goal_text: str) -> str:
        """Create a prompt for the LLM to analyze the goal text."""
        return f"""
You are an expert code analysis system. Your task is to analyze a high-level goal description
and break it down into specific, actionable sub-goals that can be implemented by a code improvement system.

HIGH-LEVEL GOAL:
{goal_text}

INSTRUCTIONS:
1. Analyze the high-level goal and identify the key components or steps needed to achieve it.
2. Break down the goal into 3-5 specific, actionable sub-goals.
3. For each sub-goal, determine:
   - A clear description of what needs to be implemented
   - The target module or file that would need to be modified (if unclear, use a placeholder)
   - The priority level (1-5, where 5 is highest priority)

Please output your analysis as a Python list of dictionaries with the following format:
```python
[
    {{
        "description": "Detailed description of sub-goal 1",
        "target_module": "likely_module_name.py",  # Best guess at which module would implement this
        "priority": 3,  # Priority from 1-5
        "type": "feature_addition"  # One of: feature_addition, bug_fix, performance_improvement, refactoring
    }},
    # More sub-goals...
]
```

Focus on being specific and actionable. Each sub-goal should be something that could be implemented
independently as a single improvement to the codebase.
"""
    
    def _parse_candidate_output(self, candidate_output: str, goal_id: str) -> List[Dict[str, Any]]:
        """
        Parse the LLM output to extract structured sub-goals.
        
        Args:
            candidate_output: Output from the LLM
            goal_id: Unique ID for this goal
            
        Returns:
            List of structured sub-goals
        """
        try:
            # Try to extract Python code from the output
            code_match = re.search(r'\[\s*{.*}\s*\]', candidate_output, re.DOTALL)
            if not code_match:
                self.logger.warning("No list of dictionaries found in LLM output")
                return []
                
            code_str = code_match.group(0)
            
            # Execute the code in a safe context to get the list of dictionaries
            # This approach is safer than using eval()
            local_vars = {}
            try:
                exec(f"result = {code_str}", {"__builtins__": {}}, local_vars)
                structured_goals = local_vars.get("result", [])
            except Exception as e:
                self.logger.error(f"Error executing extracted code: {str(e)}")
                return []
                
            # Validate and enhance the extracted goals
            validated_goals = []
            for i, goal in enumerate(structured_goals):
                if not isinstance(goal, dict):
                    continue
                    
                # Ensure required fields are present
                if "description" not in goal or not goal["description"]:
                    continue
                    
                # Add defaults for missing fields
                if "target_module" not in goal or not goal["target_module"]:
                    goal["target_module"] = "unknown.py"
                    
                if "priority" not in goal or not isinstance(goal["priority"], int):
                    goal["priority"] = 3
                    
                if "type" not in goal:
                    goal["type"] = "feature_addition"
                    
                # Add a sub-goal ID
                goal["sub_goal_id"] = f"{goal_id}_{i:02d}"
                
                validated_goals.append(goal)
                
            return validated_goals
            
        except Exception as e:
            self.logger.error(f"Error parsing candidate output: {str(e)}")
            return []
    
    def _create_default_subgoals(self, goal_text: str, goal_id: str) -> List[Dict[str, Any]]:
        """
        Create default sub-goals when analysis fails.
        
        Args:
            goal_text: Original goal text
            goal_id: Unique ID for this goal
            
        Returns:
            List of default sub-goals
        """
        self.logger.info("Creating default sub-goals")
        
        # Create a single default sub-goal based on the original text
        return [
            {
                "description": goal_text,
                "target_module": "unknown.py",
                "priority": 3,
                "type": "feature_addition",
                "sub_goal_id": f"{goal_id}_00"
            }
        ]
    
    def _generate_goal_id(self, goal_text: str) -> str:
        """Generate a unique ID for a goal based on its text content."""
        # Create a hash of the goal text
        hash_obj = hashlib.md5(goal_text.encode())
        # Use the first 8 characters of the hash
        return hash_obj.hexdigest()[:8]
    
    def create_improvement_goals(self, goal_record: Dict[str, Any]) -> List[ImprovementGoal]:
        """
        Convert processed sub-goals into ImprovementGoal objects.
        
        Args:
            goal_record: Processed goal record
            
        Returns:
            List of ImprovementGoal objects
        """
        self.logger.info(f"Creating improvement goals for goal ID: {goal_record['goal_id']}")
        
        improvement_goals = []
        
        for sub_goal in goal_record.get("structured_goals", []):
            goal = ImprovementGoal(
                target_module=sub_goal.get("target_module", "unknown.py"),
                description=sub_goal.get("description", "No description"),
                priority=sub_goal.get("priority", 3)
            )
            improvement_goals.append(goal)
            
        self.logger.info(f"Created {len(improvement_goals)} improvement goals")
        return improvement_goals
    
    def update_goal_status(self, goal_id: str, sub_goal_id: Optional[str] = None, 
                          status: Optional[str] = None, logs: Optional[List[str]] = None) -> bool:
        """
        Update the status and logs for a goal or sub-goal.
        
        Args:
            goal_id: ID of the goal to update
            sub_goal_id: Optional ID of the specific sub-goal to update
            status: Optional new status for the goal
            logs: Optional log entries to add
            
        Returns:
            True if update was successful, False otherwise
        """
        self.logger.info(f"Updating goal status for goal ID: {goal_id}")
        
        # Find the goal record file
        goal_files = list(self.goals_dir.glob(f"goal_{goal_id}_*.json"))
        if not goal_files:
            self.logger.error(f"No goal record found for goal ID: {goal_id}")
            return False
            
        goal_file = goal_files[0]  # Use the first matching file
        
        try:
            # Load the goal record
            with open(goal_file, 'r', encoding='utf-8') as f:
                goal_record = json.load(f)
                
            # Update the goal record
            if sub_goal_id:
                # Update a specific sub-goal
                for sub_goal in goal_record.get("structured_goals", []):
                    if sub_goal.get("sub_goal_id") == sub_goal_id:
                        if status:
                            sub_goal["status"] = status
                        if logs:
                            sub_goal["logs"] = sub_goal.get("logs", []) + logs
                        break
                
                # Update overall completion count
                if status == "completed":
                    goal_record["sub_goals_completed"] = goal_record.get("sub_goals_completed", 0) + 1
            else:
                # Update the overall goal
                if status:
                    goal_record["status"] = status
                if logs:
                    goal_record["logs"] = goal_record.get("logs", []) + logs
                    
                # Increment attempt count if status is "attempt"
                if status == "attempt":
                    goal_record["attempts"] = goal_record.get("attempts", 0) + 1
                    
            # Save the updated goal record
            with open(goal_file, 'w', encoding='utf-8') as f:
                json.dump(goal_record, f, indent=2)
                
            self.logger.info(f"Goal status updated successfully for goal ID: {goal_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating goal status: {str(e)}")
            return False
            
    async def analyze_error_logs(self, goal_id: str, scan_system_logs: bool = True) -> List[Dict[str, Any]]:
        """
        Analyze error logs for a specific goal to extract insights for next attempt.
        
        Args:
            goal_id: ID of the goal to analyze logs for
            scan_system_logs: Whether to also scan system log files
            
        Returns:
            List of insights extracted from the logs
        """
        self.logger.info(f"Analyzing error logs for goal ID: {goal_id}")
        
        all_insights = []
        
        # Part 1: Analyze logs from the goal record
        goal_files = list(self.goals_dir.glob(f"goal_{goal_id}_*.json"))
        if goal_files:
            goal_file = goal_files[0]  # Use the first matching file
            
            try:
                # Load the goal record
                with open(goal_file, 'r', encoding='utf-8') as f:
                    goal_record = json.load(f)
                    
                # Extract all logs
                goal_logs = goal_record.get("logs", [])
                for sub_goal in goal_record.get("structured_goals", []):
                    goal_logs.extend(sub_goal.get("logs", []))
                    
                # Filter for error logs
                error_logs = [log for log in goal_logs if "error" in log.lower() or "fail" in log.lower()]
                
                if error_logs:
                    # Create insights from goal logs
                    for log in error_logs:
                        all_insights.append({
                            "log": log,
                            "source": "goal_record",
                            "insight": "Error detected in goal execution"
                        })
                    self.logger.info(f"Extracted {len(error_logs)} insights from goal logs")
                else:
                    self.logger.info(f"No error logs found in goal record for goal ID: {goal_id}")
            except Exception as e:
                self.logger.error(f"Error analyzing goal logs: {str(e)}")
        else:
            self.logger.warning(f"No goal record found for goal ID: {goal_id}")
        
        # Part 2: Scan system logs if requested
        if scan_system_logs:
            self.logger.info("Scanning system log files for errors")
            system_insights = self._scan_system_logs()
            if system_insights:
                all_insights.extend(system_insights)
                self.logger.info(f"Extracted {len(system_insights)} insights from system logs")
            else:
                self.logger.info("No relevant errors found in system logs")
        
        # Part 3: Use LLM to generate insights from the collected error logs
        if all_insights and len(all_insights) > 0:
            enhanced_insights = await self._enhance_insights_with_llm(all_insights, goal_id)
            return enhanced_insights
        
        return all_insights
    
    def _scan_system_logs(self) -> List[Dict[str, Any]]:
        """
        Scan system log files for error messages.
        
        Returns:
            List of insights from system logs
        """
        insights = []
        logs_dir = self.base_path / "logs"
        
        if not logs_dir.exists() or not logs_dir.is_dir():
            self.logger.warning(f"Logs directory not found: {logs_dir}")
            return insights
        
        # Scan all log files in the logs directory
        for log_file in logs_dir.glob("*.log"):
            try:
                self.logger.debug(f"Scanning log file: {log_file}")
                with open(log_file, 'r', encoding='utf-8') as f:
                    log_content = f.readlines()
                
                # Extract error lines
                for line in log_content:
                    # Look for ERROR or WARNING level logs
                    if " ERROR " in line or " WARNING " in line:
                        # Extract module and message
                        parts = line.strip().split(" - ", 3)
                        if len(parts) >= 3:
                            timestamp = parts[0]
                            module = parts[1]
                            level_and_message = parts[2]
                            
                            insights.append({
                                "log": line.strip(),
                                "timestamp": timestamp,
                                "module": module,
                                "level_message": level_and_message,
                                "source": str(log_file.name),
                                "insight": "System log error detected"
                            })
            except Exception as e:
                self.logger.error(f"Error scanning log file {log_file}: {str(e)}")
        
        return insights
    
    async def _enhance_insights_with_llm(self, insights: List[Dict[str, Any]], goal_id: str) -> List[Dict[str, Any]]:
        """
        Use the LLM to generate more meaningful insights from error logs.
        
        Args:
            insights: Raw insights from logs
            goal_id: ID of the goal being analyzed
            
        Returns:
            Enhanced insights with LLM-generated analysis
        """
        try:
            # If there are too many insights, sample a representative set
            sample_insights = insights[:10] if len(insights) > 10 else insights
            
            # Create a context for the LLM
            context = {
                "insights": sample_insights,
                "goal_id": goal_id
            }
            
            # Create a prompt for analysis
            prompt = self._create_log_analysis_prompt(sample_insights)
            
            # Generate a single candidate
            candidates = await self.llm_synthesizer.generate_candidates(
                goal=ImprovementGoal(
                    target_module="",
                    description="Analyze error logs and extract insights"
                ),
                context=context,
                num_candidates=1
            )
            
            if not candidates:
                self.logger.warning("No candidates generated for log analysis")
                return insights
                
            candidate = candidates[0]
            
            # Parse the response to extract enhanced insights
            enhanced_insights = self._parse_enhanced_insights(candidate.code, insights)
            if enhanced_insights:
                self.logger.info(f"Generated {len(enhanced_insights)} enhanced insights")
                return enhanced_insights
                
        except Exception as e:
            self.logger.error(f"Error enhancing insights with LLM: {str(e)}")
        
        # If enhancement fails, return the original insights
        return insights
    
    def _create_log_analysis_prompt(self, insights: List[Dict[str, Any]]) -> str:
        """Create a prompt for the LLM to analyze error logs."""
        logs_text = "\n".join([f"- {insight['log']}" for insight in insights])
        
        return f"""
You are an expert log analysis system. Your task is to analyze error logs from a code improvement system
and provide meaningful insights and recommendations based on the patterns you observe.

ERROR LOGS:
{logs_text}

INSTRUCTIONS:
1. Analyze the error logs and identify common patterns or recurring issues
2. Determine the root causes of the errors when possible
3. Suggest specific adjustments or improvements to avoid these errors in the future
4. Provide concrete recommendations for the next improvement attempt

Please output your analysis as a Python list of dictionaries with the following format:
```python
[
    {{
        "error_pattern": "Description of the error pattern",
        "probable_cause": "Likely cause of this type of error",
        "recommendation": "Specific recommendation to address this issue",
        "priority": 3  # Priority from 1-5
    }},
    # More insights...
]
```

Focus on being specific and actionable. Your recommendations should be concrete steps that can be taken
to improve the success rate of future code transformations.
"""
    
    def _parse_enhanced_insights(self, candidate_output: str, original_insights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Parse the LLM output to extract enhanced insights.
        
        Args:
            candidate_output: Output from the LLM
            original_insights: Original insights from logs
            
        Returns:
            Enhanced insights with LLM analysis
        """
        try:
            # Try to extract Python code from the output
            code_match = re.search(r'\[\s*{.*}\s*\]', candidate_output, re.DOTALL)
            if not code_match:
                self.logger.warning("No list of dictionaries found in LLM output")
                return original_insights
                
            code_str = code_match.group(0)
            
            # Execute the code in a safe context to get the list of dictionaries
            local_vars = {}
            try:
                exec(f"result = {code_str}", {"__builtins__": {}}, local_vars)
                enhanced_insights = local_vars.get("result", [])
            except Exception as e:
                self.logger.error(f"Error executing extracted code: {str(e)}")
                return original_insights
                
            # Validate and combine with original insights
            result = []
            for i, insight in enumerate(enhanced_insights):
                if not isinstance(insight, dict):
                    continue
                    
                # Create a new enhanced insight
                enhanced = {
                    "error_pattern": insight.get("error_pattern", "Unknown pattern"),
                    "probable_cause": insight.get("probable_cause", "Unknown cause"),
                    "recommendation": insight.get("recommendation", "No recommendation"),
                    "priority": insight.get("priority", 3),
                    "llm_enhanced": True
                }
                
                # Add the original log that this enhancement is based on (if available)
                if i < len(original_insights):
                    enhanced["original_log"] = original_insights[i].get("log", "")
                    enhanced["source"] = original_insights[i].get("source", "unknown")
                
                result.append(enhanced)
                
            return result
            
        except Exception as e:
            self.logger.error(f"Error parsing enhanced insights: {str(e)}")
            return original_insights

# Add this to the utils module if not present
class GoalProcessingError(Exception):
    """Exception raised for errors during goal processing."""
    pass