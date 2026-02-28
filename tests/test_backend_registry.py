from coflect.backends import get_backend_factory, list_backends


def test_torch_backend_registered() -> None:
    # Guard retained for environments where torch import can fail at runtime.
    if "torch" not in list_backends():
        return
    factory = get_backend_factory("torch")
    assert factory.__name__ == "TorchAdapter"


def test_tensorflow_backend_registered() -> None:
    factory = get_backend_factory("tensorflow")
    assert factory.__name__ == "TensorFlowAdapter"
