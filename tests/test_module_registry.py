from coflect.modules import get_module, list_modules


def test_hitl_module_registered() -> None:
    assert "hitl" in list_modules()
    hitl = get_module("hitl")
    assert hitl.name == "hitl"
    assert "torch" in hitl.backends
    assert "tensorflow" in hitl.backends
