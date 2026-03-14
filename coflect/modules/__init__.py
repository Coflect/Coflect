"""Coflect pluggable module namespace."""

import coflect.modules.hilt as _hilt  # noqa: F401
from coflect.modules.base import ModuleSpec
from coflect.modules.registry import get_module, list_modules, register_module

__all__ = ["ModuleSpec", "register_module", "get_module", "list_modules"]
