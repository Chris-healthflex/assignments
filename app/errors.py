from typing import Any, Dict, List, Optional


class PipelineError(Exception):
    """A failure the API reports with a status code and field-level details."""

    def __init__(
        self,
        code: str,
        message: str,
        status_code: int = 500,
        details: Optional[List[Dict[str, Any]]] = None,
    ):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or []
