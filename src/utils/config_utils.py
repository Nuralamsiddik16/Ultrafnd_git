# src/utils/config_utils.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml  # requires PyYAML
    _HAS_YAML = True
except Exception:
    _HAS_YAML = False


def _to_dict(x: Any) -> Dict[str, Any]:
    if x is None:
        return {}
    if isinstance(x, dict):
        return x
    # Some YAMLs load as None or lists accidentally; normalize to empty dict
    return {}


class ConfigManager:
    """
    Tiny, dependency-tolerant YAML loader with caching and safe fallbacks.

    Usage:
        cfg = ConfigManager().load_config("configs/model_configs/fusion.yaml")
        hidden = int(cfg.get("hidden_dim", 512))
    """
    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}

    def load_config(self, path: str, defaults: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Load a YAML into a plain dict with safe fallbacks.
        - If file is missing or PyYAML isn't available, returns `defaults` or {}.
        - Caches by absolute path to avoid repeated disk reads.
        """
        p = Path(path)
        # Allow relative to repo root or cwd
        if not p.exists():
            # Try relative to project root (two levels up from this file)
            repo_root = Path(__file__).resolve().parents[2]
            alt = repo_root / path
            if alt.exists():
                p = alt

        apath = str(p.resolve()) if p.exists() else str(Path(path))

        # Return cached if present
        if apath in self._cache:
            cfg = self._cache[apath]
            return self._merge_defaults(cfg, defaults)

        # Try to load
        cfg: Dict[str, Any] = {}
        if p.exists() and p.is_file() and _HAS_YAML:
            try:
                with open(p, "r", encoding="utf-8") as f:
                    loaded = yaml.safe_load(f)
                cfg = _to_dict(loaded)
            except Exception:
                cfg = {}
        else:
            # Missing file or no YAML support → empty dict
            cfg = {}

        # Cache and merge defaults
        self._cache[apath] = cfg
        return self._merge_defaults(cfg, defaults)

    @staticmethod
    def _merge_defaults(cfg: Dict[str, Any], defaults: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not defaults:
            return cfg
        out = dict(defaults)
        out.update(cfg or {})
        return out


def load_yaml(path: str, defaults: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Convenience function if you don't want to instantiate ConfigManager.
    """
    return ConfigManager().load_config(path, defaults=defaults)
