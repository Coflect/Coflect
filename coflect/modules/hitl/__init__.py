"""HITL module registration."""

from coflect.modules.base import ModuleSpec
from coflect.modules.registry import register_module

HITL_MODULE = ModuleSpec(
    name="hitl",
    description="Human-in-the-loop training module with async XAI worker.",
    backends=("torch", "tensorflow"),
)

register_module(HITL_MODULE)

__all__ = ["HITL_MODULE"]
