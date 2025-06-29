import sys
import traceback
from typing import Optional, Dict, Any
from datetime import datetime
from pathlib import Path
import json
from app.logging_config import setup_logging

logger = setup_logging("error_handler")

class CustomException(Exception):
    """Base class for custom exceptions with detailed error tracking"""
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.timestamp = datetime.now().isoformat()
        self.traceback = traceback.format_exc()
        
        # Get the caller's frame
        frame = sys._getframe(1)
        self.file_name = frame.f_code.co_filename
        self.line_number = frame.f_lineno
        self.function_name = frame.f_code.co_name
        
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary format"""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp,
            "location": {
                "file": self.file_name,
                "line": self.line_number,
                "function": self.function_name
            },
            "traceback": self.traceback
        }

    def log_error(self):
        """Log the error with full details"""
        error_dict = self.to_dict()
        logger.error(
            f"Error occurred:\n"
            f"Code: {error_dict['error_code']}\n"
            f"Message: {error_dict['message']}\n"
            f"Location: {error_dict['location']['file']}:{error_dict['location']['line']} "
            f"in {error_dict['location']['function']}\n"
            f"Details: {json.dumps(error_dict['details'], indent=2)}\n"
            f"Traceback:\n{error_dict['traceback']}"
        )

def handle_exception(exc: Exception, error_code: Optional[str] = None) -> Dict[str, Any]:
    """
    Handle any exception and convert it to a standardized format
    
    Args:
        exc: The exception to handle
        error_code: Optional error code to categorize the error
        
    Returns:
        Dict containing error details
    """
    if isinstance(exc, CustomException):
        exc.log_error()
        return exc.to_dict()
    
    # For non-custom exceptions, create a custom exception
    custom_exc = CustomException(
        message=str(exc),
        error_code=error_code or "UNKNOWN_ERROR",
        details={"original_exception": exc.__class__.__name__}
    )
    custom_exc.log_error()
    return custom_exc.to_dict()

# Example usage of custom exceptions
class ValidationError(CustomException):
    """Raised when input validation fails"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "VALIDATION_ERROR", details)

class APIError(CustomException):
    """Raised when API calls fail"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "API_ERROR", details)

class DatabaseError(CustomException):
    """Raised when database operations fail"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "DATABASE_ERROR", details) 