"""Reusable coercion helpers that enforce the two hard schema constraints.

The output contract requires that every string field is a string (never null) and
every array field is an array (even with a single item). The extraction model
does not reliably honour that on its own, so it is enforced here in the schema
rather than left to the prompt.
"""

from typing import Annotated

from pydantic import BeforeValidator


def _empty_if_none(v):
    """Turn null into an empty string, and coerce non-strings to str."""
    if v is None:
        return ""
    return str(v)


# A string field that accepts null and stores "" instead. Used for every string
# field in the assessment schema, replacing what would otherwise be the same
# validator duplicated across each section model.
EmptyStr = Annotated[str, BeforeValidator(_empty_if_none)]


def as_list(v):
    """Coerce a value into a list.

    The model routinely returns `null` for an empty section, or a bare object
    when there is exactly one item. Both become valid arrays here.
    """
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return list(v)
    return [v]


def as_object(v):
    """Turn a null section into an empty section instead of a hard error.

    Every field on the nested section models has a default, so an empty dict
    validates into a fully-populated section of empty strings.
    """
    return {} if v is None else v
