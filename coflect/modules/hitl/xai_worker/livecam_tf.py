"""TensorFlow LiveCAM attribution helpers for HITL XAI worker."""

from __future__ import annotations

import io
from typing import Any

import numpy as np
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


def _normalize(hm: np.ndarray) -> np.ndarray:
    """Normalize heatmap values into [0, 1].

    Example:
        >>> _normalize(np.array([[0.0, 2.0]], dtype=np.float32)).tolist()
        [[0.0, 1.0]]
    """
    x = hm.astype(np.float32, copy=False)
    x = x - x.min()
    x = x / (x.max() + 1e-8)
    return x


def _to_uint8_hwc(x_hwc: np.ndarray) -> np.ndarray:
    """Convert float image tensor to uint8 HWC for visualization.

    Example:
        >>> _to_uint8_hwc(np.zeros((2, 2, 3), dtype=np.float32)).dtype
        dtype('uint8')
    """
    x = x_hwc.astype(np.float32, copy=False)
    x = _robust_normalize_01(x, lo_q=1.0, hi_q=99.0)
    return (x * 255.0).clip(0, 255).astype(np.uint8)


def _heat_to_rgb(heatmap_hw: np.ndarray) -> np.ndarray:
    """Map normalized heatmap to a high-contrast RGB colormap."""
    h = _normalize(heatmap_hw).astype(np.float32, copy=False)
    r = np.clip(1.5 - np.abs(4.0 * h - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * h - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * h - 1.0), 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def _focus_bbox(heatmap_hw: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return bbox around high-activation region (x0, y0, x1, y1)."""
    h = _normalize(heatmap_hw)
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
    """Blend heatmap while preserving the original image context."""
    h = _normalize(heatmap_hw)
    focus = np.clip((h - 0.55) / 0.45, 0.0, 1.0)
    heat_rgb = _heat_to_rgb(h)
    base = base_hwc.astype(np.float32, copy=False) / 255.0
    alpha_map = np.clip(alpha * focus + 0.70 * focus, 0.0, 0.85)[..., None]
    out = base * (1.0 - alpha_map) + heat_rgb * alpha_map
    return (out * 255.0).clip(0, 255).astype(np.uint8)


def make_input_png_tf(x_input: Any) -> bytes:
    """Render the normalized original input image as PNG."""
    x_arr = np.asarray(x_input, dtype=np.float32)
    if x_arr.ndim == 3 and x_arr.shape[0] == 3:
        x_hwc = np.transpose(x_arr, (1, 2, 0))
    elif x_arr.ndim == 3 and x_arr.shape[-1] == 3:
        x_hwc = x_arr
    else:
        raise ValueError("Expected CHW or HWC 3-channel input")
    base = _to_uint8_hwc(x_hwc)
    im = Image.fromarray(base)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def _agreement(a_hw: np.ndarray, b_hw: np.ndarray) -> float:
    """Return blended correlation/IoU agreement in [0, 1].

    Example:
        >>> round(_agreement(np.ones((2, 2), dtype=np.float32), np.ones((2, 2), dtype=np.float32)), 3)
        1.0
    """
    a = _normalize(a_hw).reshape(-1)
    b = _normalize(b_hw).reshape(-1)
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


def gradcam_tf(model: Any, x_nhwc: Any, target_class: int, layer_name: str = "conv3") -> np.ndarray:
    """Compute TF Grad-CAM heatmap resized to input spatial shape.

    Example:
        >>> # heat = gradcam_tf(model, x_nhwc, target_class=1)
    """
    import tensorflow as tf

    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output],
    )
    with tf.GradientTape() as tape:
        conv_out, logits = grad_model(x_nhwc, training=False)
        score = logits[:, target_class]
    grads = tape.gradient(score, conv_out)
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.reduce_sum(weights * conv_out, axis=-1)
    cam = tf.nn.relu(cam)
    cam = tf.image.resize(cam[..., tf.newaxis], size=(x_nhwc.shape[1], x_nhwc.shape[2]), method="bilinear")
    return _normalize(cam[0, :, :, 0].numpy())


def smoothgrad_tf(model: Any, x_nhwc: Any, target_class: int, samples: int = 8, noise_std: float = 0.15) -> np.ndarray:
    """Compute SmoothGrad saliency by averaging noisy gradients.

    Example:
        >>> # heat = smoothgrad_tf(model, x_nhwc, target_class=1, samples=4)
    """
    import tensorflow as tf

    base = tf.convert_to_tensor(x_nhwc)
    scale = tf.reduce_max(base) - tf.reduce_min(base)
    sigma = tf.cast(noise_std * scale + 1e-6, base.dtype)
    acc = np.zeros((int(x_nhwc.shape[1]), int(x_nhwc.shape[2])), dtype=np.float32)
    for _ in range(max(1, samples)):
        noise = tf.random.normal(shape=tf.shape(base), stddev=sigma)
        x = tf.identity(base + noise)
        with tf.GradientTape() as tape:
            tape.watch(x)
            logits = model(x, training=False)
            score = logits[:, target_class]
        grad = tape.gradient(score, x)
        sal = tf.reduce_mean(tf.abs(grad), axis=-1)[0]
        acc += _normalize(sal.numpy())
    return _normalize(acc / float(max(1, samples)))


def _select_heatmap(method: str, h_gradcam: np.ndarray, h_sgrad: np.ndarray) -> np.ndarray:
    """Pick the final heatmap based on configured attribution method.

    Example:
        >>> _select_heatmap(\"gradcam\", np.ones((2, 2), dtype=np.float32), np.zeros((2, 2), dtype=np.float32)).shape
        (2, 2)
    """
    key = method.strip().lower()
    if key in {"gradcam", "livecam"}:
        return h_gradcam
    if key == "smoothgrad":
        return h_sgrad
    if key in {"consensus", "hybrid"}:
        return _normalize(0.5 * h_gradcam + 0.5 * h_sgrad)
    raise ValueError(f"Unknown attribution method: {method}")


def make_overlay_png_tf_with_meta(
    model: Any,
    x_input: Any,
    target_class: int,
    method: str = "consensus",
) -> tuple[bytes, dict[str, object]]:
    """Render TF attribution overlay and return metadata."""
    import tensorflow as tf

    x_arr = np.asarray(x_input, dtype=np.float32)
    if x_arr.ndim == 3 and x_arr.shape[0] == 3:
        x_hwc = np.transpose(x_arr, (1, 2, 0))
    elif x_arr.ndim == 3 and x_arr.shape[-1] == 3:
        x_hwc = x_arr
    else:
        raise ValueError("Expected CHW or HWC 3-channel input")

    x_nhwc = tf.convert_to_tensor(x_hwc[np.newaxis, ...], dtype=tf.float32)
    h_gradcam = gradcam_tf(model, x_nhwc, target_class=target_class)
    h_sgrad = smoothgrad_tf(model, x_nhwc, target_class=target_class)
    heat = _select_heatmap(method, h_gradcam, h_sgrad)
    agreement = _agreement(h_gradcam, h_sgrad)

    base = _to_uint8_hwc(x_hwc)
    bbox = _focus_bbox(heat)
    out = overlay_heatmap(base, heat)
    out = _draw_bbox(out, bbox=bbox, color=(255, 255, 0))
    im = Image.fromarray(out)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    meta: dict[str, object] = {"xai_agreement": agreement}
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        h, w = heat.shape
        meta["focus_bbox"] = [
            float(x0 / max(1, w)),
            float(y0 / max(1, h)),
            float((x1 + 1) / max(1, w)),
            float((y1 + 1) / max(1, h)),
        ]
    else:
        meta["focus_bbox"] = None
    return buf.getvalue(), meta
