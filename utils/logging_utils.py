import logging
import logging.handlers
from typing import Optional


class TrainingLogger:
    def __init__(
        self,
        log_file_path: str,
        name: Optional[str] = None,
        log_level: int = logging.INFO,
        verbose: Optional[bool] = False,
    ):
        """Initialize the logger object

        Args:
            log_file_path (str): Path to the log file
            name (str, optional): Name of the logger. Defaults to None.
            log_level (int, optional): Log level. Defaults to logging.INFO.
            verbose (bool, optional): Whether to print the logs to stdout. Defaults to False.
        """
        self.log_file_path = log_file_path
        if name is None:
            name = __name__
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        self.messages = []
        self.verbose = verbose

        self._setup_file_handler()

    def _setup_file_handler(self):
        """Set up the file handler for logging"""
        try:
            file_handler = logging.handlers.RotatingFileHandler(
                self.log_file_path, maxBytes=1024 * 1024, backupCount=5
            )
            # maxBytes = 1024 * 1024 = 1 MB and backupCount = 5 means that at most 5 files will be created
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        except Exception as e:
            print(f"Error setting up log file handler: {str(e)}")

    def log_info(self, message: str):
        """Log an info message"""
        self.logger.info(message)
        self.messages.append(message)
        self.print_last_message() if self.verbose else None

    def log_warning(self, message: str):
        """Log a warning message"""
        self.logger.warning(message)
        self.messages.append(message)
        self.print_last_message() if self.verbose else None

    def log_error(self, message: str):
        """Log an error message"""
        self.logger.error(message)
        self.messages.append(message)
        self.print_last_message() if self.verbose else None

    def log_critical(self, message: str):
        """Log a critical message"""
        self.logger.critical(message)
        self.messages.append(message)
        self.print_last_message() if self.verbose else None

    def print_last_message(self):
        """Print the last message in the log file"""
        if self.messages:
            print(self.messages[-1])
        else:
            print("No messages logged.")
