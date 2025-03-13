import sys
from loguru import logger
from typing import Optional
from pathlib import Path


class LoggerFactory:
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, log_dir: Optional[str] = None):
        if not LoggerFactory._initialized:
            self.setup_logger(log_dir)
            LoggerFactory._initialized = True

    def setup_logger(self, log_dir: Optional[str] = None):
        # Remove default handler
        logger.remove()

        # Set up log directory
        if log_dir is None:
            log_dir = "logs"  # default path
        
        # Create log directory if it doesn't exist
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        # Add console handler
        logger.add(
            sys.stdout,
            colorize=True,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level="INFO"
        )

        # Add file handler
        log_file = log_path / "app.log"
        logger.add(
            str(log_file),
            rotation="500 MB",
            retention="10 days",
            compression="zip",
            level="DEBUG",
            enqueue=True
        )

    def get_logger(self, name: Optional[str] = None):
        return logger.bind(context=name) if name else logger