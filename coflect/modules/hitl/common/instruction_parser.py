"""Deterministic parser for free-text HITL instructions.

The goal is pragmatic flexibility with bounded behavior: parse a small command
language from natural text without introducing nondeterministic model calls.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

_ABS_RE = re.compile(r"(?:strength|lambda|focus)\s*[:=]?\s*([0-9]*\.?[0-9]+)")
_DELTA_RE = re.compile(r"(increase|decrease|raise|lower)\s+(?:focus|strength|lambda)?(?:\s*by)?\s*([0-9]*\.?[0-9]+)?")
_COORD_RE = re.compile(
    r"x0\s*[:=]\s*([0-9]*\.?[0-9]+)\D+"
    r"y0\s*[:=]\s*([0-9]*\.?[0-9]+)\D+"
    r"x1\s*[:=]\s*([0-9]*\.?[0-9]+)\D+"
    r"y1\s*[:=]\s*([0-9]*\.?[0-9]+)"
)
_RECT_RE = re.compile(
    r"x\s*[:=]\s*([0-9]*\.?[0-9]+)\D+"
    r"y\s*[:=]\s*([0-9]*\.?[0-9]+)\D+"
    r"w\s*[:=]\s*([0-9]*\.?[0-9]+)\D+"
    r"h\s*[:=]\s*([0-9]*\.?[0-9]+)"
)

_ROI_KEYWORDS: dict[str, tuple[float, float, float, float]] = {
    "center": (0.25, 0.25, 0.75, 0.75),
    "centre": (0.25, 0.25, 0.75, 0.75),
    "middle": (0.25, 0.25, 0.75, 0.75),
    "left": (0.0, 0.1, 0.5, 0.9),
    "right": (0.5, 0.1, 1.0, 0.9),
    "top": (0.1, 0.0, 0.9, 0.5),
    "bottom": (0.1, 0.5, 0.9, 1.0),
    "top-left": (0.0, 0.0, 0.5, 0.5),
    "top left": (0.0, 0.0, 0.5, 0.5),
    "top-right": (0.5, 0.0, 1.0, 0.5),
    "top right": (0.5, 0.0, 1.0, 0.5),
    "bottom-left": (0.0, 0.5, 0.5, 1.0),
    "bottom left": (0.0, 0.5, 0.5, 1.0),
    "bottom-right": (0.5, 0.5, 1.0, 1.0),
    "bottom right": (0.5, 0.5, 1.0, 1.0),
}


def _clamp01(x: float) -> float:
    """Clamp numeric value into [0.0, 1.0].

    Example:
        >>> _clamp01(1.3)
        1.0
    """
    return max(0.0, min(1.0, x))


@dataclass(frozen=True)
class ParsedInstruction:
    """Structured policy update parsed from instruction text."""

    strength: float | None = None
    strength_delta: float = 0.0
    roi_norm: tuple[float, float, float, float] | None = None
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Convert parsed instruction to JSON-safe dict.

        Example:
            >>> ParsedInstruction(strength=0.2).to_dict()[\"strength\"]
            0.2
        """
        return asdict(self)


def parse_instruction(text: str) -> ParsedInstruction:
    """Parse free text into deterministic strength/ROI adjustments."""
    if not text.strip():
        return ParsedInstruction()

    msg = " ".join(text.lower().split())
    notes: list[str] = []
    strength: float | None = None
    delta = 0.0
    roi_norm: tuple[float, float, float, float] | None = None

    if any(kw in msg for kw in ("disable focus", "stop focus", "reset focus", "turn off focus")):
        strength = 0.0
        notes.append("focus_reset")

    m_abs = _ABS_RE.search(msg)
    if m_abs:
        raw = float(m_abs.group(1))
        if raw > 1.0:
            raw = raw / 100.0
        strength = _clamp01(raw)
        notes.append("strength_absolute")

    m_delta = _DELTA_RE.search(msg)
    if m_delta:
        direction = m_delta.group(1)
        mag_raw = m_delta.group(2)
        mag = float(mag_raw) if mag_raw is not None else 0.1
        if mag > 1.0:
            mag = mag / 100.0
        mag = _clamp01(mag)
        sign = 1.0 if direction in {"increase", "raise"} else -1.0
        delta += sign * mag
        notes.append("strength_delta")
    else:
        if "increase focus" in msg or "raise focus" in msg:
            delta += 0.1
            notes.append("strength_delta_default")
        if "decrease focus" in msg or "lower focus" in msg:
            delta -= 0.1
            notes.append("strength_delta_default")

    m_coords = _COORD_RE.search(msg)
    if m_coords:
        x0, y0, x1, y1 = (float(v) for v in m_coords.groups())
        if max(x0, y0, x1, y1) > 1.0:
            notes.append("roi_coords_ignored_non_normalized")
        else:
            x0c, x1c = sorted((_clamp01(x0), _clamp01(x1)))
            y0c, y1c = sorted((_clamp01(y0), _clamp01(y1)))
            if x1c > x0c and y1c > y0c:
                roi_norm = (x0c, y0c, x1c, y1c)
                notes.append("roi_coordinates")
    elif (m_rect := _RECT_RE.search(msg)) is not None:
        x, y, w, h = (float(v) for v in m_rect.groups())
        x0c, y0c = _clamp01(x), _clamp01(y)
        x1c, y1c = _clamp01(x + w), _clamp01(y + h)
        if x1c > x0c and y1c > y0c:
            roi_norm = (x0c, y0c, x1c, y1c)
            notes.append("roi_xywh")

    if roi_norm is None:
        if (
            "ignore border" in msg
            or "ignore borders" in msg
            or "ignore edges" in msg
            or "ignore background" in msg
            or "focus object" in msg
        ):
            roi_norm = (0.15, 0.15, 0.85, 0.85)
            notes.append("roi_ignore_border")
        else:
            for key, roi in _ROI_KEYWORDS.items():
                if key in msg:
                    roi_norm = roi
                    notes.append(f"roi_{key}")
                    break

    return ParsedInstruction(
        strength=strength,
        strength_delta=delta,
        roi_norm=roi_norm,
        notes=tuple(notes),
    )
