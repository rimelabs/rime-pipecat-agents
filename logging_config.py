"""
Logging configuration module for Rime Pipecat Agent.
Handles incremental log file creation and proper log formatting.
"""

import os
import glob
from datetime import datetime
from loguru import logger
import sys


class LoggingManager:
    """Manages logging configuration with incremental file naming."""
    
    def __init__(self, log_dir="logs", base_filename="rime_agent_log"):
        self.log_dir = log_dir
        self.base_filename = base_filename
        self.current_log_file = None
        
    def setup_logging(self):
        """Set up logging with incremental file naming."""
        # Create logs directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Find the next available log file number
        log_number = self._get_next_log_number()
        
        # Create the log filename with timestamp and number
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_log_file = os.path.join(
            self.log_dir, 
            f"{self.base_filename}_{log_number:03d}_{timestamp}.txt"
        )
        
        # Remove default logger
        logger.remove()
        
        # Add console output (optional - can be disabled by setting level to ERROR)
        logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                   "<level>{message}</level>",
            level="INFO"
        )
        
        # Add file output with detailed formatting
        logger.add(
            self.current_log_file,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
            level="DEBUG",
            rotation="100 MB",  # Rotate when file gets too large
            retention="30 days",  # Keep logs for 30 days
            compression="zip",  # Compress old logs
            backtrace=True,  # Include full traceback in logs
            diagnose=True   # Include variable values in tracebacks
        )
        
        # Log the startup information
        logger.info("=" * 80)
        logger.info(f"Rime Pipecat Agent Starting - Session {log_number}")
        logger.info(f"Log file: {self.current_log_file}")
        logger.info(f"Timestamp: {datetime.now().isoformat()}")
        logger.info("=" * 80)
        
        return self.current_log_file
    
    def _get_next_log_number(self):
        """Find the next available log number."""
        # Look for existing log files matching our pattern
        pattern = os.path.join(self.log_dir, f"{self.base_filename}_*.txt")
        existing_files = glob.glob(pattern)
        
        if not existing_files:
            return 1
        
        # Extract numbers from existing files
        numbers = []
        for file_path in existing_files:
            filename = os.path.basename(file_path)
            # Extract number from filename like "rime_agent_log_001_20231009_143022.txt"
            try:
                parts = filename.replace(f"{self.base_filename}_", "").split("_")
                if parts and parts[0].isdigit():
                    numbers.append(int(parts[0]))
            except (ValueError, IndexError):
                continue
        
        return max(numbers) + 1 if numbers else 1
    
    def log_session_end(self):
        """Log session end information."""
        if logger:
            logger.info("=" * 80)
            logger.info("Rime Pipecat Agent Session Ended")
            logger.info(f"End timestamp: {datetime.now().isoformat()}")
            logger.info("=" * 80)
    
    def get_current_log_file(self):
        """Get the current log file path."""
        return self.current_log_file


# Global logging manager instance
_logging_manager = None


def setup_application_logging():
    """Set up application logging. Call this at the start of your application."""
    global _logging_manager  # pylint: disable=global-statement
    _logging_manager = LoggingManager()
    return _logging_manager.setup_logging()


def log_session_end():
    """Log session end. Call this when the application is shutting down."""
    if _logging_manager:
        _logging_manager.log_session_end()


def get_current_log_file():
    """Get the current log file path."""
    if _logging_manager:
        return _logging_manager.get_current_log_file()
    return None
