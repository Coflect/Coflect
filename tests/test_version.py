import coflect


def test_version_present() -> None:
    assert isinstance(coflect.__version__, str)
    assert coflect.__version__
