"""JEV error envelope and conversion from Pydantic errors."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

STRIPPED_LOC_PREFIXES = {"body"}


class ErrorDetail(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str
    message: str


class ErrorResponse(BaseModel):
    """Body returned for every rejected request."""

    model_config = ConfigDict(extra="forbid")

    error: Literal["VALIDATION_ERROR", "NOT_FOUND", "CONFLICT", "BAD_REQUEST", "INTERNAL_ERROR"]
    details: list[ErrorDetail] = []


class JEVError(Exception):
    """Validation failure raised outside FastAPI body parsing (e.g. after merging defaults)."""

    def __init__(self, details: list[ErrorDetail]):
        super().__init__("; ".join(f"{d.field}: {d.message}" for d in details))
        self.details = details


def details_from_errors(
    errors: list[dict], prefix: tuple[str | int, ...] = ()
) -> list[ErrorDetail]:
    """Convert Pydantic/FastAPI error dicts into field-path details like ``params.top_n``."""
    out: list[ErrorDetail] = []
    for err in errors:
        loc = [*prefix, *err.get("loc", ())]
        if err.get("type") == "json_invalid":
            loc = []  # loc holds a character offset, not a field
        elif loc and loc[0] in STRIPPED_LOC_PREFIXES:
            loc = loc[1:]
        field = ".".join(str(p) for p in loc) or "body"
        message = str(err.get("msg", "Invalid value")).removeprefix("Value error, ")
        out.append(ErrorDetail(field=field, message=message))
    return out


def jev_error_from(exc: ValidationError, prefix: tuple[str | int, ...] = ()) -> JEVError:
    return JEVError(details_from_errors(exc.errors(), prefix))
