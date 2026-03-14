"""HILT module registration."""

from coflect.modules.base import ModuleSpec
from coflect.modules.registry import register_module

HILT_MODULE = ModuleSpec(
    name="hilt",
    description="Human In Loop Training module with async XAI worker.",
    backends=("torch", "tensorflow"),
)

register_module(HILT_MODULE)

__all__ = ["HILT_MODULE"]
