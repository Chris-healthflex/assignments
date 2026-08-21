from app.guardrails.source_match import fuzzy_source_match, normalize_text
from app.guardrails.date_validator import is_valid_date_string
from app.guardrails.numeric_validator import is_numeric

def test_fuzzy_match_numbers():
    transcript = "patient had pain three days ago"
    value = "3 days"
    assert fuzzy_source_match(value, transcript) >= 70

def test_normalize_numbers():
    assert "three" in normalize_text("3")

def test_date_valid():
    assert is_valid_date_string("2024-01-15")
    assert is_valid_date_string("2 weeks")
    assert not is_valid_date_string("not a date")

def test_numeric():
    assert is_numeric("45")
    assert is_numeric("45 degrees")
    assert not is_numeric("mild")