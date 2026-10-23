"""Engines by name, built on first use, so the bench can switch models without a restart.

A heavy engine (a model of gigabytes) can be marked as such: at most `max_heavy` heavy engines stay loaded, and
loading another one unloads the oldest (16 GB of memory is shared with the LLM).
"""
from __future__ import annotations

import gc
import logging
import threading
import time
from collections.abc import Callable
from typing import Any

log = logging.getLogger("orb.speech")


class EngineError(RuntimeError):
    """An engine that works through another service failed. The message says why, in plain words, with no secret."""


class Registry:
    def __init__(self, factories: dict[str, Callable[[], Any]], default: str, *, heavy: frozenset[str] = frozenset(),
                 max_heavy: int = 1, meta: dict[str, dict] | None = None):
        if default not in factories:
            raise ValueError(f"The default engine {default!r} is not one of {', '.join(factories)}.")
        self._factories, self.default, self._heavy, self._max_heavy = dict(factories), default, heavy, max_heavy
        self.meta = meta or {}  # facts per engine that are known without loading it (voices, styles, languages)
        self._loaded: dict[str, Any] = {}
        self._order: list[str] = []  # heavy engines, oldest first
        self._lock = threading.Lock()

    @classmethod
    def single(cls, engine: Any) -> "Registry":
        """One engine that is already built (tests and fakes)."""
        name = getattr(engine, "name", "default")
        reg = cls({name: lambda: engine}, name)
        reg._loaded[name] = engine
        return reg

    @property
    def names(self) -> list[str]:
        return list(self._factories)

    @property
    def loaded(self) -> list[str]:
        return list(self._loaded)

    def resolve(self, name: str | None) -> str:
        name = name or self.default
        if name not in self._factories:
            raise ValueError(f"Unknown engine {name!r}. Engines: {', '.join(self._factories)}.")
        return name

    def get(self, name: str | None = None) -> Any:
        """The engine, built now if needed (slow for a large model: call it in a worker thread)."""
        name = self.resolve(name)
        with self._lock:
            engine = self._loaded.get(name)
            if engine is None:
                if name in self._heavy:
                    while len(self._order) >= self._max_heavy:
                        self._unload(self._order.pop(0))
                t0 = time.perf_counter()
                engine = self._factories[name]()
                self._loaded[name] = engine
                if name in self._heavy:
                    self._order.append(name)
                log.info("engine %s loaded in %.1f s", name, time.perf_counter() - t0)
            return engine

    def peek(self, name: str | None = None) -> Any | None:
        """The engine if it is loaded, else None. It never builds one."""
        return self._loaded.get(self.resolve(name))

    def _unload(self, name: str) -> None:
        engine = self._loaded.pop(name, None)
        if engine is not None and hasattr(engine, "close"):
            engine.close()
        del engine
        gc.collect()
        log.info("engine %s unloaded", name)
