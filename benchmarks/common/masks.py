"""Packed-mask output validation shared by the segmentation benchmark checkers.

The mask paths write one PGM per detection alongside the result JSON. Masks are
emitted either bbox-local or full-frame; both are accepted, and a bbox-local mask is
expanded into frame coordinates when it has to be compared against a full-frame one.
"""

from __future__ import annotations

import cv2
import numpy as np


def load_mask(json_path, detection, frame_shape=None):
    mask_path = json_path.parent / detection["mask_file"]
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"could not read mask artifact: {mask_path}")
    mask = mask != 0
    if not np.count_nonzero(mask):
        raise ValueError(f"empty mask artifact: {mask_path}")
    x, y, width, height = detection["bbox"]
    if mask.shape == (height, width):
        if frame_shape is None:
            return mask
        expanded = np.zeros(frame_shape, dtype=bool)
        if y + height > frame_shape[0] or x + width > frame_shape[1]:
            raise ValueError(f"bbox-local mask exceeds frame: {mask_path}")
        expanded[y : y + height, x : x + width] = mask
        return expanded
    if frame_shape is not None and mask.shape != frame_shape:
        raise ValueError(f"full-frame mask shape mismatch: {mask_path}")
    return mask


def mask_overlap_for(reference_path, candidate_path):
    """Build an overlap callable bound to the two result files being compared."""

    def mask_overlap(ref, cand):
        ref_mask = load_mask(reference_path, ref)
        cand_mask = load_mask(candidate_path, cand, ref_mask.shape)
        intersection = int(np.count_nonzero(ref_mask & cand_mask))
        union = int(np.count_nonzero(ref_mask | cand_mask))
        return intersection / union if union else 0.0

    return mask_overlap
