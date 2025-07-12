"""
Logging configuration for LocalRAG
"""

import sys
from pathlib import Path

from loguru import logger

from core.config import settings


def setup_logging():
    """Configure application logging"""
    # Remove default logger
    logger.remove()
    
    # Console logging
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    
    logger.add(
        sys.stdout,
        format=console_format,
        level=settings.log_level,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )
    
    # File logging
    if settings.log_file_path:
        log_path = Path(settings.log_file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_format = (
            "{time:YYYY-MM-DD HH:mm:ss} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{message}"
        )
        
        logger.add(
            log_path,
            format=file_format,
            level=settings.log_level,
            rotation=settings.log_rotation,
            retention=settings.log_retention,
            compression="gz",
            backtrace=True,
            diagnose=True,
        )
    
    # Performance/metrics logging
    if settings.enable_metrics:
        metrics_path = log_path.parent / "metrics.log" if settings.log_file_path else None
        if metrics_path:
            logger.add(
                metrics_path,
                format="{time} | {level} | {message}",
                level="INFO",
                rotation="1 day",
                retention="7 days",
                filter=lambda record: "metrics" in record["extra"],
            )
    
    logger.info("Logging configured successfully")


def get_logger(name: str):
    """Get a logger instance for a specific module"""
    return logger.bind(name=name)


# Performance and metrics logging helpers
def log_performance(operation: str, duration: float, **kwargs):
    """Log performance metrics"""
    logger.bind(metrics=True).info(
        f"PERFORMANCE | {operation} | duration={duration:.3f}s | {kwargs}"
    )


def log_api_request(method: str, path: str, status_code: int, duration: float):
    """Log API request metrics"""
    logger.bind(metrics=True).info(
        f"API | {method} {path} | {status_code} | {duration:.3f}s"
    )


def log_database_query(query_type: str, duration: float, rows_affected: int = None):
    """Log database query metrics"""
    extra = f"rows={rows_affected}" if rows_affected is not None else ""
    logger.bind(metrics=True).info(
        f"DATABASE | {query_type} | {duration:.3f}s | {extra}"
    )


def log_model_inference(model_name: str, input_size: int, duration: float):
    """Log model inference metrics"""
    logger.bind(metrics=True).info(
        f"MODEL | {model_name} | input_size={input_size} | {duration:.3f}s"
    )