"""TensorFlow model builders for HITL module."""

from __future__ import annotations

from typing import Any


def build_tf_cnn(num_classes: int = 10, image_size: int = 64) -> Any:
    """Build a compact CNN with a named final conv layer for Grad-CAM."""
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise RuntimeError("TensorFlow is required; install with `pip install coflect[tensorflow]`") from exc

    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name="image")
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu", name="conv1")(inputs)
    x = tf.keras.layers.MaxPool2D(pool_size=2, name="pool1")(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu", name="conv2")(x)
    x = tf.keras.layers.MaxPool2D(pool_size=2, name="pool2")(x)
    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu", name="conv3")(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    logits = tf.keras.layers.Dense(num_classes, name="logits")(x)
    return tf.keras.Model(inputs=inputs, outputs=logits, name="coflect_hitl_cnn")
