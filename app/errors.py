class AppError(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class BadRequestError(AppError):
    pass


class ExtractionConfidenceError(AppError):
    def __init__(self, low_confidence_fields: list[str]):
        self.low_confidence_fields = low_confidence_fields
        super().__init__("Extraction confidence is too low")


class TranscriptionError(AppError):
    pass


class DatabaseUnavailableError(AppError):
    pass


class DatabaseError(AppError):
    pass
