from coflect.modules.hitl.common.instruction_parser import parse_instruction


def test_parse_strength_delta_percent() -> None:
    parsed = parse_instruction("increase focus by 20%")
    assert parsed.strength is None
    assert abs(parsed.strength_delta - 0.2) < 1e-8


def test_parse_absolute_strength_and_center_roi() -> None:
    parsed = parse_instruction("focus=0.35 and center")
    assert parsed.strength is not None
    assert abs(parsed.strength - 0.35) < 1e-8
    assert parsed.roi_norm == (0.25, 0.25, 0.75, 0.75)


def test_parse_coordinate_roi() -> None:
    parsed = parse_instruction("x0=0.1 y0=0.2 x1=0.9 y1=0.8")
    assert parsed.roi_norm == (0.1, 0.2, 0.9, 0.8)


def test_parse_xywh_roi() -> None:
    parsed = parse_instruction("x=0.2 y=0.1 w=0.4 h=0.5")
    assert parsed.roi_norm is not None
    x0, y0, x1, y1 = parsed.roi_norm
    assert abs(x0 - 0.2) < 1e-8
    assert abs(y0 - 0.1) < 1e-8
    assert abs(x1 - 0.6) < 1e-8
    assert abs(y1 - 0.6) < 1e-8


def test_parse_disable_focus() -> None:
    parsed = parse_instruction("disable focus now")
    assert parsed.strength == 0.0
