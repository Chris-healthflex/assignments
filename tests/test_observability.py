import json
import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.observability import (
    JsonFormatter,
    RequestContextMiddleware,
    install_request_id_factory,
    log_context,
    request_id_var,
)

# The factory is what binds a record to its request; install it for the suite
# the way `configure_logging()` does at startup.
install_request_id_factory()


@pytest.fixture
def logger_at_info(caplog):
    """A logger that actually builds records, so `extra=` collisions surface.

    The default pytest level lets `logger.info(...)` short-circuit before
    `makeRecord` runs, which is exactly how a bad `extra=` key hides.
    """
    caplog.set_level(logging.INFO)
    return logging.getLogger("app.test")


def format_one(record: logging.LogRecord) -> dict:
    return json.loads(JsonFormatter().format(record))


class TestJsonFormatter:
    def test_emits_json_with_the_standard_envelope(self, logger_at_info, caplog):
        logger_at_info.info("hello")
        payload = format_one(caplog.records[-1])

        assert payload["message"] == "hello"
        assert payload["level"] == "INFO"
        assert payload["logger"] == "app.test"
        assert "ts" in payload and "request_id" in payload

    def test_context_fields_are_flattened_into_the_payload(self, logger_at_info, caplog):
        logger_at_info.info("done", extra=log_context(status=200, duration_ms=12.5))
        payload = format_one(caplog.records[-1])

        assert payload["status"] == 200
        assert payload["duration_ms"] == 12.5
        assert "_ctx" not in payload

    @pytest.mark.parametrize("reserved", ["filename", "module", "name", "args"])
    def test_context_field_may_shadow_a_logrecord_attribute(
        self, logger_at_info, caplog, reserved
    ):
        """`extra={"filename": ...}` raises KeyError; log_context must not."""
        logger_at_info.info("upload", extra=log_context(**{reserved: "session.wav"}))
        payload = format_one(caplog.records[-1])

        assert payload[reserved] == "session.wav"
        assert payload["message"] == "upload"

    @pytest.mark.parametrize("envelope_key", ["message", "level", "request_id"])
    def test_a_context_field_cannot_corrupt_the_envelope(
        self, logger_at_info, caplog, envelope_key
    ):
        logger_at_info.info("real message", extra=log_context(**{envelope_key: "spoofed"}))
        payload = format_one(caplog.records[-1])

        assert payload[envelope_key] != "spoofed"
        # ...but the caller's value is moved aside, not thrown away.
        assert payload[f"{envelope_key}_"] == "spoofed"
        assert payload["message"] == "real message"

    def test_raw_extra_still_lands_in_the_payload(self, logger_at_info, caplog):
        """uvicorn passes `extra=` directly; we shouldn't drop those fields."""
        logger_at_info.info("access", extra={"color_message": "x"})
        assert format_one(caplog.records[-1])["color_message"] == "x"

    def test_the_id_is_bound_when_the_record_is_made_not_when_it_is_written(
        self, logger_at_info, caplog
    ):
        token = request_id_var.set("abc123")
        try:
            logger_at_info.info("in request")
        finally:
            request_id_var.reset(token)

        # Formatting happens here, well after the request context is gone.
        assert format_one(caplog.records[-1])["request_id"] == "abc123"

    def test_exceptions_are_serialised_rather_than_swallowed(self, logger_at_info, caplog):
        try:
            raise ValueError("transcription blew up")
        except ValueError:
            logger_at_info.exception("failed")

        assert "transcription blew up" in format_one(caplog.records[-1])["exception"]

    def test_unserialisable_values_do_not_break_the_log_line(self, logger_at_info, caplog):
        logger_at_info.info("odd", extra=log_context(path=object()))
        assert "path" in format_one(caplog.records[-1])


def build_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(RequestContextMiddleware)

    @app.get("/ok")
    def ok():
        return {"request_id": request_id_var.get()}

    @app.get("/boom")
    def boom():
        raise RuntimeError("nope")

    return app


class TestRequestContextMiddleware:
    def test_generates_a_request_id_and_echoes_it(self):
        response = TestClient(build_app()).get("/ok")

        assert response.status_code == 200
        assert response.headers["X-Request-ID"]
        # The handler saw the same id that came back on the response.
        assert response.json()["request_id"] == response.headers["X-Request-ID"]

    def test_honours_a_client_supplied_id_so_traces_join_up(self):
        response = TestClient(build_app()).get(
            "/ok", headers={"X-Request-ID": "trace-from-caller"}
        )

        assert response.headers["X-Request-ID"] == "trace-from-caller"
        assert response.json()["request_id"] == "trace-from-caller"

    def test_logs_the_completed_request_with_its_timing(self, caplog):
        caplog.set_level(logging.INFO)
        TestClient(build_app()).get("/ok")

        record = next(r for r in caplog.records if r.message == "request completed")
        payload = format_one(record)
        assert payload["method"] == "GET"
        assert payload["path"] == "/ok"
        assert payload["status"] == 200
        assert payload["duration_ms"] >= 0

    def test_logs_and_re_raises_a_failing_request(self, caplog):
        caplog.set_level(logging.INFO)
        client = TestClient(build_app(), raise_server_exceptions=False)

        assert client.get("/boom").status_code == 500

        record = next(r for r in caplog.records if r.message == "request failed")
        payload = format_one(record)
        assert payload["path"] == "/boom"
        assert "RuntimeError" in payload["exception"]
        # The id is still bound while the failure is logged, not reset early.
        assert payload["request_id"] != "-"

    def test_the_context_var_does_not_leak_between_requests(self):
        client = TestClient(build_app())

        first = client.get("/ok").json()["request_id"]
        second = client.get("/ok").json()["request_id"]

        assert first != second
        assert request_id_var.get() == "-"
