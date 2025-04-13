"""
Sample scheduler module for demonstrating the Update Generator.
"""
import datetime
import time
from typing import Any, Callable, Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)

class Task:
    """Represents a scheduled task."""
    
    def __init__(
        self, 
        name: str, 
        function: Callable, 
        args: List[Any] = None, 
        kwargs: Dict[str, Any] = None,
        interval: int = 60,
        start_time: Optional[datetime.datetime] = None,
        end_time: Optional[datetime.datetime] = None
    ):
        """
        Initialize a scheduled task.
        
        Args:
            name: Name of the task
            function: Function to execute
            args: Positional arguments for the function
            kwargs: Keyword arguments for the function
            interval: Execution interval in seconds
            start_time: Optional start time for the task
            end_time: Optional end time after which the task will not execute
        """
        self.name = name
        self.function = function
        self.args = args or []
        self.kwargs = kwargs or {}
        self.interval = interval
        self.start_time = start_time
        self.end_time = end_time
        self.last_run = None
        self.next_run = None
        
        # Calculate the next run time
        self._calculate_next_run()
    
    def _calculate_next_run(self) -> None:
        """Calculate the next time this task should run."""
        if self.last_run:
            self.next_run = self.last_run + datetime.timedelta(seconds=self.interval)
        else:
            if self.start_time and self.start_time > datetime.datetime.now():
                self.next_run = self.start_time
            else:
                self.next_run = datetime.datetime.now()
    
    def should_run(self) -> bool:
        """Check if the task should run now."""
        now = datetime.datetime.now()
        
        # Check if the task has expired
        if self.end_time and now > self.end_time:
            return False
        
        # Check if it's time to run
        return self.next_run <= now
    
    def execute(self) -> Any:
        """Execute the task and update timing information."""
        start_time = time.time()
        
        try:
            result = self.function(*self.args, **self.kwargs)
            logger.info(f"Task '{self.name}' executed successfully")
            return result
        except Exception as e:
            logger.error(f"Error executing task '{self.name}': {str(e)}")
            raise
        finally:
            end_time = time.time()
            duration = end_time - start_time
            logger.debug(f"Task '{self.name}' execution time: {duration:.4f}s")
            
            self.last_run = datetime.datetime.now()
            self._calculate_next_run()


class Scheduler:
    """Simple task scheduler."""
    
    def __init__(self):
        """Initialize the scheduler."""
        self.tasks = []
        self.running = False
    
    def add_task(self, task: Task) -> None:
        """Add a task to the scheduler."""
        self.tasks.append(task)
        logger.info(f"Added task '{task.name}' to scheduler")
    
    def remove_task(self, task_name: str) -> bool:
        """Remove a task from the scheduler by name."""
        for i, task in enumerate(self.tasks):
            if task.name == task_name:
                self.tasks.pop(i)
                logger.info(f"Removed task '{task_name}' from scheduler")
                return True
        
        logger.warning(f"Task '{task_name}' not found in scheduler")
        return False
    
    def schedule_task(
        self, 
        name: str, 
        function: Callable, 
        args: List[Any] = None, 
        kwargs: Dict[str, Any] = None,
        interval: int = 60,
        start_time: Optional[datetime.datetime] = None,
        end_time: Optional[datetime.datetime] = None
    ) -> Task:
        """
        Schedule a new task.
        
        Args:
            name: Name of the task
            function: Function to execute
            args: Positional arguments for the function
            kwargs: Keyword arguments for the function
            interval: Execution interval in seconds
            start_time: Optional start time for the task
            end_time: Optional end time after which the task will not execute
            
        Returns:
            The created Task object
        """
        task = Task(
            name=name,
            function=function,
            args=args,
            kwargs=kwargs,
            interval=interval,
            start_time=start_time,
            end_time=end_time
        )
        
        self.add_task(task)
        return task
    
    def run_pending(self) -> int:
        """
        Run all pending tasks.
        
        Returns:
            Number of tasks executed
        """
        count = 0
        for task in self.tasks:
            if task.should_run():
                logger.debug(f"Running task '{task.name}'")
                task.execute()
                count += 1
        
        return count
    
    def run(self, interval: float = 1.0) -> None:
        """
        Run the scheduler in a loop.
        
        Args:
            interval: Sleep interval between checks in seconds
        """
        self.running = True
        logger.info("Starting scheduler")
        
        try:
            while self.running:
                self.run_pending()
                time.sleep(interval)
        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user")
            self.running = False
        except Exception as e:
            logger.error(f"Error in scheduler: {str(e)}")
            self.running = False
            raise
        
        logger.info("Scheduler stopped")
    
    def stop(self) -> None:
        """Stop the scheduler."""
        self.running = False