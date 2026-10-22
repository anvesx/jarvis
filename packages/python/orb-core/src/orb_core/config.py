"""Settings from the process environment, with .env files as a fallback.

Order of precedence: a variable that is already set in the process wins, then the first .env file that sets it.
Secret values are never logged, printed or returned by `describe`. Only their names and "set" or "missing" are.
"""
from __future__ import annotations

import os
from pathlib import Path

SECRET_HINTS = ("KEY", "TOKEN", "SECRET", "PASSWORD")


def is_secret(name: str) -> bool:
    upper = name.upper()
    return any(h in upper for h in SECRET_HINTS)


def parse_env_text(text: str) -> dict[str, str]:
    """KEY=VALUE lines. Blank lines and # comments are skipped. An `export ` prefix and quotes are removed."""
    out = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            out[key] = value
    return out


def load_env_files(*paths: str | os.PathLike | None) -> list[str]:
    """Copy variables from .env files into os.environ. A variable that is already set is not changed.

    Returns the names that were set, never the values. A missing file is skipped.
    """
    loaded = []
    for p in paths:
        if not p:
            continue
        path = Path(p).expanduser()
        if not path.is_file():
            continue
        for key, value in parse_env_text(path.read_text()).items():
            if value and key not in os.environ:
                os.environ[key] = value
                loaded.append(key)
    return loaded


def bootstrap(root: str | os.PathLike | None = None) -> list[str]:
    """Load the repo .env, then the file that ORB_SECRETS_FILE or JEV_ENV_FILE names."""
    root = Path(root) if root else repo_root()
    extra = os.environ.get("ORB_SECRETS_FILE") or os.environ.get("JEV_ENV_FILE")
    return load_env_files(root / ".env", extra)


def env(name: str, default: str | None = None, required: bool = False) -> str | None:
    value = os.environ.get(name, "").strip()
    if value:
        return value
    if required:
        raise RuntimeError(f"{name} is not set. Set it in the environment or in the repo .env file.")
    return default


def describe(names: list[str]) -> dict[str, str]:
    """For a health page: "set" or "missing" for each name. It never includes a value."""
    return {n: ("set" if os.environ.get(n, "").strip() else "missing") for n in names}


def repo_root() -> Path:
    """The orb repo root: ORB_ROOT, or the nearest parent folder that has go.work."""
    if os.environ.get("ORB_ROOT"):
        return Path(os.environ["ORB_ROOT"]).expanduser()
    here = Path.cwd().resolve()
    for p in (here, *here.parents):
        if (p / "go.work").is_file():
            return p
    return here
