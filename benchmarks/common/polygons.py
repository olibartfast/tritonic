"""Polygon-output validation shared by the segmentation benchmark checkers.

The GPU polygon path emits convex hull rings per instance, in bbox-relative image
coordinates, with an explicit point-count that must agree with the ring data. These
helpers both *validate* that structure and rasterize it, so the same code that proves
the output is well-formed also produces the pixel set used for CPU-vs-GPU agreement.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

# A ring covering essentially the whole box is the signature of a degenerate trace
# rather than a real instance.
MAX_FILL_RATIO = 0.98


def signed_area(ring):
    return 0.5 * sum(
        x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(ring, ring[1:] + ring[:1])
    )


def point_in_ring(point, ring):
    px, py = point
    inside = False
    previous = ring[-1]
    for current in ring:
        ax, ay = current
        bx, by = previous
        if (ay > py) != (by > py) and px < (bx - ax) * (py - ay) / (by - ay) + ax:
            inside = not inside
        previous = current
    return inside


def validate_ring(ring, bbox, expected_sign, name):
    if len(ring) < 3:
        raise ValueError(f"{name} has fewer than three points")
    x, y, width, height = bbox
    for point in ring:
        if len(point) != 2 or not all(math.isfinite(value) for value in point):
            raise ValueError(f"{name} contains an invalid point")
        px, py = point
        if px < x or px > x + width or py < y or py > y + height:
            raise ValueError(f"{name} point lies outside its bounding box")
    area = signed_area(ring)
    if area == 0 or math.copysign(1.0, area) != expected_sign:
        raise ValueError(f"{name} has invalid winding or zero area")
    for index, current in enumerate(ring):
        previous = ring[index - 1]
        following = ring[(index + 1) % len(ring)]
        turn = (current[0] - previous[0]) * (following[1] - current[1]) - (
            current[1] - previous[1]
        ) * (following[0] - current[0])
        if turn * expected_sign < 0:
            raise ValueError(f"{name} is not convex")


def polygon_pixels(detection):
    """Validate a detection's polygon output and return its filled pixel set."""
    bbox = detection["bbox"]
    x, y, width, height = bbox
    if width <= 0 or height <= 0:
        raise ValueError("invalid detection bounding box")
    polygons = detection.get("polygons", [])
    if not polygons or detection.get("polygon_count") != len(polygons):
        raise ValueError("missing or inconsistent polygon output")

    actual_point_count = 0
    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon_index, polygon in enumerate(polygons):
        exterior = polygon.get("exterior", [])
        holes = polygon.get("holes", [])
        validate_ring(exterior, bbox, 1.0, f"polygon {polygon_index} exterior")
        actual_point_count += len(exterior)
        for hole_index, hole in enumerate(holes):
            validate_ring(hole, bbox, -1.0, f"polygon {polygon_index} hole {hole_index}")
            if not point_in_ring(hole[0], exterior):
                raise ValueError("polygon hole is not contained by its exterior")
            actual_point_count += len(hole)

        exterior_points = np.asarray(
            [[point[0] - x, point[1] - y] for point in exterior], dtype=np.int32
        )
        cv2.fillPoly(mask, [exterior_points], 1)
        for hole in holes:
            hole_points = np.asarray(
                [[point[0] - x, point[1] - y] for point in hole], dtype=np.int32
            )
            cv2.fillPoly(mask, [hole_points], 0)

    if detection.get("polygon_point_count") != actual_point_count:
        raise ValueError("polygon_point_count does not match polygon data")
    filled = int(np.count_nonzero(mask))
    fill_ratio = filled / (width * height)
    if not filled or fill_ratio >= MAX_FILL_RATIO:
        raise ValueError("empty or near-solid garbage polygon output")
    ys, xs = np.nonzero(mask)
    return set(zip(xs + x, ys + y))


def polygon_overlap(ref, cand):
    ref_pixels = polygon_pixels(ref)
    cand_pixels = polygon_pixels(cand)
    return len(ref_pixels & cand_pixels) / len(ref_pixels | cand_pixels)
