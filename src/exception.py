import sys
import traceback
from src.logger import logging

def format_error_message(error: Exception, error_detail=sys) -> str:
    """
    Extracts detailed error information including filename, line number,
    and error message.

    Args:
        error (Exception): The exception object.
        error_detail (module): Typically the sys module to get exc_info.

    Returns:
        str: A formatted error message.
    """
    exc_type, exc_value, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename if exc_tb else "<unknown>"

    return (
        f"Error occurred in script [{file_name}] "
        f"at line [{exc_tb.tb_lineno if exc_tb else 'unknown'}]: "
        f"{str(error)}"
    )


class CustomException(Exception):
    """
    Custom Exception class that provides detailed error information.
    """

    def __init__(self, error: Exception, error_detail=sys):
        super().__init__(str(error))
        self.error_message = format_error_message(error, error_detail)

    def __str__(self) -> str:
        return self.error_message
