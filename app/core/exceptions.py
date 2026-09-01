class LowConfidenceExtractionError(Exception):
    """
    Raised when one or more required fields could not be confidently
    extracted from the transcript. Carries field-level detail so the
    API layer can return HTTP 422 with specifics.
    """
    def __init__(self, missing_fields: list[str]):
        self.missing_fields = missing_fields
        super().__init__(f"Low confidence extraction for fields: {missing_fields}")