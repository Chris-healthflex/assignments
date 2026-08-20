from __future__ import annotations
import json
import os
from pathlib import Path

DEFAULT_FIXTURE = Path("tests/fixtures/stub_extraction.json")


class StubStructuredLLM:
    def __init__(self, output_model: type) -> None:
        self._output_model = output_model

    def invoke(self, _messages: object) -> object:
        fixture = Path(os.environ.get("STUB_EXTRACTION_PATH", DEFAULT_FIXTURE))
        if fixture.is_file():
            return self._output_model.model_validate(
                json.loads(fixture.read_text(encoding="utf-8"))
            )
        return self._output_model()
