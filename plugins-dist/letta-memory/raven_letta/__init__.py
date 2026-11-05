"""Letta memory backend for Raven.

The package body imports cheaply on purpose: plugin discovery imports this
module when it resolves the ``raven.plugins`` entry point, and the registry
may health-check a backend without ever starting it. The Letta wire and the
contract surface live in :mod:`raven_letta.backend`.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
