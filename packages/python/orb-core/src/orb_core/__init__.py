"""Shared parts of every orb agent.

- config: settings from the environment and .env files. Secret values are never printed.
- jev: the reader. Jev (TypeSafe System One) answers typed questions about text.
- ollama: a streaming client for a local LLM.
- agent: the one interface that every agent and every channel uses.
- web: load the readable text of a web page, with a guard against private addresses.
"""

__version__ = "0.1.0"
