import traceback
import sys

class CustomException(Exception):
    def __init__(self, error_message, error_detail=None):
        self.error_message = self.get_detailed_error_message(error_message, error_detail)
        super().__init__(self.error_message)

    @staticmethod
    def get_detailed_error_message(error_message, error_detail=None):
        exc_type, exc_value, exc_tb = sys.exc_info()
        if exc_tb:
            file_name = exc_tb.tb_frame.f_code.co_filename
            line_number = exc_tb.tb_lineno
            return (f"Error in {file_name}, line {line_number}: {error_message} | "
                    f"Original: {exc_value}")
        else:
            return f"Error: {error_message}"

    def __str__(self):
        return self.error_message
