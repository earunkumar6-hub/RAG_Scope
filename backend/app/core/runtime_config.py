"""Persisted runtime configuration (pipeline defaults + guardrails), editable via the API."""

import json
import logging
import threading
from pathlib import Path

from pydantic import ValidationError

from app.schemas.config import RuntimeConfig

logger = logging.getLogger(__name__)


class RuntimeConfigStore:
    """Thread-safe JSON-file-backed holder of the current ``RuntimeConfig``."""

    def __init__(self, path: Path):
        self._path = path
        self._lock = threading.Lock()
        self._config = self._load()

    def _load(self) -> RuntimeConfig:
        """Load the persisted config; an unreadable/outdated file is set aside, not fatal."""
        if not self._path.exists():
            return RuntimeConfig()
        try:
            return RuntimeConfig.model_validate(json.loads(self._path.read_text("utf-8")))
        except (json.JSONDecodeError, ValidationError) as exc:
            backup = self._path.with_suffix(".invalid.json")
            self._path.replace(backup)
            logger.warning(
                "invalid runtime config moved aside; using defaults",
                extra={"backup": str(backup), "error": str(exc)},
            )
            return RuntimeConfig()

    def get(self) -> RuntimeConfig:
        with self._lock:
            return self._config.model_copy(deep=True)

    def set(self, config: RuntimeConfig) -> RuntimeConfig:
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(config.model_dump_json(indent=2), "utf-8")
            tmp.replace(self._path)
            self._config = config.model_copy(deep=True)
            return self._config.model_copy(deep=True)
