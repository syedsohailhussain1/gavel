"""Vendored snake-immortal engine (provably-safe Hamiltonian policies)."""
try:
    from .policies import POLICIES
except ImportError:  # pragma: no cover - script-dir fallback
    from policies import POLICIES

__all__ = ["POLICIES"]
