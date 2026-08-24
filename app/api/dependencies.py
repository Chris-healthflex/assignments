"""Shared request helpers for the API."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from fastapi import HTTPException


def parse_date(value: Optional[str], field: str) -> Optional[datetime]:
    """Parse an ISO date/datetime query param into an aware UTC datetime."""
    if value is None:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid '{field}': expected ISO date like 2026-08-24 or 2026-08-24T10:00:00",
        )
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt
