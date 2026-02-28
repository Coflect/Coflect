"""LiveCAM attribution helpers for the XAI worker.

Implemented methods:
- livecam: class-activation map on a configurable CNN layer
- smoothgrad: smoothed absolute input-gradient saliency
- consensus: mean blend of livecam and smoothgrad
"""

from __future__ import annotations

import io

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _robust_normalize_01(x: np.ndarray, lo_q: float = 1.0, hi_q: float = 99.0) -> np.ndarray:
    """Normalize array to [0, 1] using robust quantiles."""
    lo = float(np.percentile(x, lo_q))
    hi = float(np.percentile(x, hi_q))
    if hi <= lo + 1e-8:
        lo = float(x.min())
        hi = float(x.max())
        if hi <= lo + 1e-8:
            return np.zeros_like(x, dtype=np.float32)
    y = (x - lo) / (hi - lo + 1e-8)
    return np.clip(y, 0.0, 1.0).astype(np.float32, copy=False)


def _to_uint8_img(chw: torch.Tensor) -> np.ndarray:
    """Convert CHW tensor to uint8 HWC image for overlay rendering."""
    x = chw.detach().cpu().float().numpy().astype(np.float32, copy=False)
    x = _robust_normalize_01(x, lo_q=1.0, hi_q=99.0)
    x = (x * 255.0).clip(0, 255).astype(np.uint8)
    return np.transpose(x, (1, 2, 0)).copy()


def _heat_to_rgb(heatmap_hw: np.ndarray) -> np.ndarray:
    """Map normalized heatmap to a high-contrast RGB colormap."""
    h = _normalize_heatmap(heatmap_hw).astype(np.float32, copy=False)
    r = np.clip(1.5 - np.abs(4.0 * h - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * h - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * h - 1.0), 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def _focus_bbox(heatmap_hw: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return bbox around high-activation region (x0, y0, x1, y1)."""
    h = _normalize_heatmap(heatmap_hw)
    threshold = max(0.70, float(np.quantile(h, 0.97)))
    mask = h >= threshold
    min_area = max(6, int(mask.size * 0.002))
    if int(mask.sum()) < min_area:
        return None
    ys, xs = np.nonzero(mask)
    if xs.size == 0 or ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _draw_bbox(img_hwc: np.ndarray, bbox: tuple[int, int, int, int] | None, color: tuple[int, int, int]) -> np.ndarray:
    """Draw a thin rectangle on image and return a copy."""
    if bbox is None:
        return img_hwc
    out = img_hwc.copy()
    h, w = out.shape[:2]
    x0, y0, x1, y1 = bbox
    x0 = max(0, min(w - 1, x0))
    x1 = max(0, min(w - 1, x1))
    y0 = max(0, min(h - 1, y0))
    y1 = max(0, min(h - 1, y1))
    if x1 <= x0 or y1 <= y0:
        return out

    t = 1
    out[y0 : min(h, y0 + t), x0 : x1 + 1] = color
    out[max(0, y1 - t + 1) : y1 + 1, x0 : x1 + 1] = color
    out[y0 : y1 + 1, x0 : min(w, x0 + t)] = color
    out[y0 : y1 + 1, max(0, x1 - t + 1) : x1 + 1] = color
    return out


def overlay_heatmap(base_hwc: np.ndarray, heatmap_hw: np.ndarray, alpha: float = 0.08) -> np.ndarray:
    """Blend heatmap while preserving the original image context.

    Low-activation pixels stay close to original image.
    High-activation pixels receive stronger heat overlay.
    """
    h = _normalize_heatmap(heatmap_hw)
    # Emphasize only strong responses so the whole image is still readable.
    focus = np.clip((h - 0.55) / 0.45, 0.0, 1.0)
    heat_rgb = _heat_to_rgb(h)
    base = base_hwc.astype(np.float32, copy=False) / 255.0
    alpha_map = np.clip(alpha * focus + 0.70 * focus, 0.0, 0.85)[..., None]
    out = base * (1.0 - alpha_map) + heat_rgb * alpha_map
    return (out * 255.0).clip(0, 255).astype(np.uint8)


def make_input_png(x_chw: torch.Tensor) -> bytes:
    """Render the normalized original input image as PNG."""
    base = _to_uint8_img(x_chw)
    im = Image.fromarray(base)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def _normalize_heatmap(hm: torch.Tensor | np.ndarray) -> np.ndarray:
    """Normalize heatmap to [0, 1]."""
    if isinstance(hm, torch.Tensor):
        x = hm.detach().float().cpu().numpy()
    else:
        x = hm.astype(np.float32)
    x = x - x.min()
    x = x / (x.max() + 1e-8)
    return x


def _heatmap_agreement(a_hw: np.ndarray, b_hw: np.ndarray) -> float:
    """Return heuristic agreement score in [0, 1] between two heatmaps."""
    a = _normalize_heatmap(a_hw).reshape(-1)
    b = _normalize_heatmap(b_hw).reshape(-1)
    if a.size == 0 or b.size == 0:
        return 0.0
    corr = float(np.corrcoef(a, b)[0, 1]) if a.size > 1 else 0.0
    if np.isnan(corr):
        corr = 0.0
    corr01 = 0.5 * (max(-1.0, min(1.0, corr)) + 1.0)
    a_mask = a >= 0.6
    b_mask = b >= 0.6
    inter = float(np.logical_and(a_mask, b_mask).sum())
    union = float(np.logical_or(a_mask, b_mask).sum())
    iou = inter / union if union > 0 else 0.0
    return float(max(0.0, min(1.0, 0.5 * corr01 + 0.5 * iou)))


def gradcam_resnet(model: torch.nn.Module, x: torch.Tensor, target_class: int, layer_name: str = "layer4") -> np.ndarray:
    """
    Minimal Grad-CAM for ResNet. Returns heatmap HxW in [0, 1] (upsampled to input resolution).
    """
    acts: dict[str, torch.Tensor] = {}
    grads: dict[str, torch.Tensor] = {}

    layer = getattr(model, layer_name)

    def fwd_hook(_, __, output):
        """Capture forward activations for CAM weighting.

        Example:
            >>> # registered via layer.register_forward_hook(fwd_hook)
        """
        acts["v"] = output

    def bwd_hook(_, _grad_in, grad_out):
        """Capture backward gradients for CAM weighting.

        Example:
            >>> # registered via layer.register_full_backward_hook(bwd_hook)
        """
        grads["v"] = grad_out[0]

    h1 = layer.register_forward_hook(fwd_hook)
    h2 = layer.register_full_backward_hook(bwd_hook)

    model.zero_grad(set_to_none=True)
    logits = model(x)  # 1xC
    score = logits[:, target_class].sum()
    score.backward()

    h1.remove()
    h2.remove()

    A = acts["v"]          # 1xCxH'xW'
    dA = grads["v"]        # 1xCxH'xW'
    w = dA.mean(dim=(2, 3), keepdim=True)  # 1xCx1x1
    cam = (w * A).sum(dim=1, keepdim=True)  # 1x1xH'xW'
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)
    cam = cam[0, 0]
    return _normalize_heatmap(cam)


def saliency_absgrad(model: torch.nn.Module, x: torch.Tensor, target_class: int) -> np.ndarray:
    """Compute absolute gradient saliency and channel-reduce to HxW."""
    x = x.detach().clone().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    logits = model(x)
    score = logits[:, target_class].sum()
    score.backward()
    g = x.grad.detach().abs().mean(dim=1)[0]
    return _normalize_heatmap(g)


def smoothgrad_saliency(
    model: torch.nn.Module,
    x: torch.Tensor,
    target_class: int,
    samples: int = 8,
    noise_std: float = 0.15,
) -> np.ndarray:
    """Compute SmoothGrad saliency by averaging noisy gradient maps."""
    base = x.detach()
    scale = (base.max() - base.min()).detach()
    sigma = (noise_std * scale).item() + 1e-6
    acc = np.zeros((x.shape[2], x.shape[3]), dtype=np.float32)
    for _ in range(max(1, samples)):
        noise = torch.randn_like(base) * sigma
        hm = saliency_absgrad(model, base + noise, target_class=target_class)
        acc += hm
    return _normalize_heatmap(acc / float(max(1, samples)))


def select_heatmap(model: torch.nn.Module, x: torch.Tensor, target_class: int, method: str = "livecam") -> np.ndarray:
    """Select and compute heatmap for an attribution method."""
    key = method.strip().lower()
    if key in {"gradcam", "livecam"}:
        return gradcam_resnet(model, x, target_class=target_class)
    if key == "smoothgrad":
        return smoothgrad_saliency(model, x, target_class=target_class)
    if key in {"consensus", "hybrid"}:
        cam = gradcam_resnet(model, x, target_class=target_class)
        sgrad = smoothgrad_saliency(model, x, target_class=target_class)
        return _normalize_heatmap(0.5 * cam + 0.5 * sgrad)
    raise ValueError(f"Unknown attribution method: {method}")


def make_overlay_png(
    model: torch.nn.Module,
    x_chw: torch.Tensor,
    target_class: int,
    method: str = "livecam",
) -> bytes:
    """Render an attribution overlay PNG for a single CHW input tensor."""
    model.eval()
    with torch.enable_grad():
        x = x_chw.unsqueeze(0)  # 1x3xHxW
        heat = select_heatmap(model, x, target_class=target_class, method=method)

    base = _to_uint8_img(x_chw)
    bbox = _focus_bbox(heat)
    out = overlay_heatmap(base, heat)
    out = _draw_bbox(out, bbox=bbox, color=(255, 255, 0))

    im = Image.fromarray(out)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def make_overlay_png_with_meta(
    model: torch.nn.Module,
    x_chw: torch.Tensor,
    target_class: int,
    method: str = "livecam",
) -> tuple[bytes, dict[str, object]]:
    """Render overlay and emit attribution quality metadata."""
    model.eval()
    with torch.enable_grad():
        x = x_chw.unsqueeze(0)
        h_gradcam = gradcam_resnet(model, x, target_class=target_class)
        h_sgrad = smoothgrad_saliency(model, x, target_class=target_class)
        if method.strip().lower() in {"gradcam", "livecam"}:
            h_selected = h_gradcam
        elif method.strip().lower() == "smoothgrad":
            h_selected = h_sgrad
        else:
            h_selected = _normalize_heatmap(0.5 * h_gradcam + 0.5 * h_sgrad)
        agreement = _heatmap_agreement(h_gradcam, h_sgrad)

    base = _to_uint8_img(x_chw)
    bbox = _focus_bbox(h_selected)
    out = overlay_heatmap(base, h_selected)
    out = _draw_bbox(out, bbox=bbox, color=(255, 255, 0))
    im = Image.fromarray(out)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    meta: dict[str, object] = {"xai_agreement": agreement}
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        h, w = h_selected.shape
        meta["focus_bbox"] = [
            float(x0 / max(1, w)),
            float(y0 / max(1, h)),
            float((x1 + 1) / max(1, w)),
            float((y1 + 1) / max(1, h)),
        ]
    else:
        meta["focus_bbox"] = None
    return buf.getvalue(), meta
