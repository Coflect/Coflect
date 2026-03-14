from coflect.modules import get_module, list_modules


def test_hilt_module_registered() -> None:
    assert "hilt" in list_modules()
    hilt = get_module("hilt")
    assert hilt.name == "hilt"
    assert "torch" in hilt.backends
    assert "tensorflow" in hilt.backends
