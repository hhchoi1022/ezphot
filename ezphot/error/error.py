
class PlatesolveError(Exception):
    """Custom exception raised when plate solving fails."""

    def __init__(self, message, filepath=None, error_code=None):
        super().__init__(message)
        self.filepath = filepath
        self.error_code = error_code

    def __str__(self):
        parts = [super().__str__()]
        if self.filepath:
            parts.append(f"[File: {self.filepath}]")
        if self.error_code is not None:
            parts.append(f"[Code: {self.error_code}]")
        return ' '.join(parts)
    
class MaskingError(Exception):
    """Custom exception raised when Masking fails."""

    def __init__(self, message, filepath=None, error_code=None):
        super().__init__(message)
        self.filepath = filepath
        self.error_code = error_code

    def __str__(self):
        parts = [super().__str__()]
        if self.filepath:
            parts.append(f"[File: {self.filepath}]")
        if self.error_code is not None:
            parts.append(f"[Code: {self.error_code}]")
        return ' '.join(parts)
