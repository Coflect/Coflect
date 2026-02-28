"""Backend interfaces, implementations, and default registry bindings."""

from coflect.backends.base import BackendAdapter
from coflect.backends.jax_backend import JaxAdapter
from coflect.backends.registry import get_backend_factory, list_backends, register_backend
from coflect.backends.tensorflow_backend import TensorFlowAdapter

TorchAdapter: type[BackendAdapter] | None

try:
    from coflect.backends.torch_backend import TorchAdapter as _TorchAdapter
except Exception:  # pragma: no cover - optional dependency guard
    TorchAdapter = None
else:
    TorchAdapter = _TorchAdapter
    # Torch is the default backend shipped in the initial release line.
    register_backend("torch", _TorchAdapter)

# TensorFlow adapter can be instantiated when optional dependencies are installed.
register_backend("tensorflow", TensorFlowAdapter)

__all__ = [
    "BackendAdapter",
    "TensorFlowAdapter",
    "JaxAdapter",
    "register_backend",
    "get_backend_factory",
    "list_backends",
]
if TorchAdapter is not None:
    __all__.append("TorchAdapter")
