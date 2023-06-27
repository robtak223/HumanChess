
class CustomException(Exception):
    """
    Custom Exception class for model error handling
    """
    def __init__(self, message):
        super().__init__(message)

class InputException(CustomException):
    """
    Handles an error with function input format
    """
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)
