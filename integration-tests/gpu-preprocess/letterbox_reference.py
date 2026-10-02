#!/usr/bin/env python3
"""NumPy reference for the YOLO26 letterbox contract (spec R-2).

Mirrors deploy/object_detection/yolo26/ensemble/generate_pipeline.py:
  rw = W*640 // max(W,H), rh = H*640 // max(W,H)
  bilinear resize, half-pixel centers, no antialiasing, rounded back to uint8
  centered paste on a 640x640 canvas with fill 0
  RGB, x/255, CHW float32

The pad offset rounding is a parameter until spike T-3 measures DALI's rule.
"""

from __future__ import annotations

import numpy as np

CANVAS = 640


def letterbox_geometry(height: int, width: int, canvas: int = CANVAS, pad_rounding: str = "floor"):
    """Return (resized_h, resized_w, top, left) for one image."""
    extent = max(height, width)
    resized_w = width * canvas // extent
    resized_h = height * canvas // extent
    if resized_w == 0 or resized_h == 0:
        # Aspect ratios beyond canvas:1 collapse one side to zero pixels; every
        # variant rejects these inputs instead of resizing to an empty image.
        raise ValueError(f"degenerate letterbox for {height}x{width}")
    if pad_rounding == "floor":
        top = (canvas - resized_h) // 2
        left = (canvas - resized_w) // 2
    elif pad_rounding == "ceil":
        top = (canvas - resized_h + 1) // 2
        left = (canvas - resized_w + 1) // 2
    else:
        raise ValueError(f"unknown pad rounding: {pad_rounding}")
    return resized_h, resized_w, top, left


def _source_axis(out_size: int, in_size: int):
    scale = in_size / out_size
    source = (np.arange(out_size, dtype=np.float64) + 0.5) * scale - 0.5
    source = np.clip(source, 0.0, in_size - 1)
    low = np.floor(source).astype(np.int64)
    high = np.minimum(low + 1, in_size - 1)
    weight = (source - low).astype(np.float32)
    return low, high, weight


def resize_bilinear_half_pixel(image: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Bilinear resize of an HxWxC uint8 image; returns float32 before rounding."""
    in_h, in_w = image.shape[:2]
    y0, y1, wy = _source_axis(out_h, in_h)
    x0, x1, wx = _source_axis(out_w, in_w)
    src = image.astype(np.float32)
    top = src[y0][:, x0] * (1 - wx)[None, :, None] + src[y0][:, x1] * wx[None, :, None]
    bottom = src[y1][:, x0] * (1 - wx)[None, :, None] + src[y1][:, x1] * wx[None, :, None]
    return top * (1 - wy)[:, None, None] + bottom * wy[:, None, None]


def letterbox(rgb: np.ndarray, canvas: int = CANVAS, pad_rounding: str = "floor") -> np.ndarray:
    """RGB HxWx3 uint8 -> float32 [3, canvas, canvas] per spec R-2."""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("expected an HxWx3 uint8 RGB image")
    height, width = rgb.shape[:2]
    resized_h, resized_w, top, left = letterbox_geometry(height, width, canvas, pad_rounding)
    resized = resize_bilinear_half_pixel(rgb, resized_h, resized_w)
    resized = np.clip(np.rint(resized), 0, 255).astype(np.float32)
    output = np.zeros((canvas, canvas, 3), dtype=np.float32)
    output[top : top + resized_h, left : left + resized_w] = resized
    return np.ascontiguousarray(output.transpose(2, 0, 1) / 255.0, dtype=np.float32)
