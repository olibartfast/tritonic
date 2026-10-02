"""Synthetic fixtures for benchmark paths that only misbehave under load.

The stock fixtures (bus, horses, person, mug) each yield 10-50 anchors above the
GPU plugin's confidence gate -- comfortably under the 100-detection cap -- so a
postprocessor that truncates its candidate list cannot be distinguished from a
correct one by running them. Tiling one of them produces a scene dense enough to
cross the cap and expose the difference.

The fixture is generated rather than committed: it is exactly derivable from
bus.jpg, and because the checks it feeds are differential (CPU path versus GPU
path on the *same* file) any JPEG encoder variation cancels out.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# 2x2 takes bus.jpg from 50 to ~164 anchors above the 0.5 gate. Larger grids shrink
# each instance until scores fall away again, so this is the useful density.
CROWD_GRID = 2
CROWD_QUALITY = 95


def make_crowded_fixture(source, destination, grid=CROWD_GRID):
    """Tile `source` into a grid and write it to `destination`, returning the path."""
    source, destination = Path(source), Path(destination)
    image = cv2.imread(str(source))
    if image is None:
        raise ValueError(f"could not read fixture source: {source}")
    tiled = np.tile(image, (grid, grid, 1))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(
        str(destination), tiled, [cv2.IMWRITE_JPEG_QUALITY, CROWD_QUALITY]
    ):
        raise ValueError(f"could not write crowded fixture: {destination}")
    return destination


# High-resolution fixtures expose resolution-dependent costs (JPEG decode, request
# bytes, resize input size) that the sub-megapixel stock fixtures hide. They are
# stretched from bus.jpg rather than committed: the comparison is between
# preprocessing paths on the *same* file, so the content only needs to be
# deterministic and realistic, not undistorted.
RESOLUTION_FIXTURES = {"bus_1080p": (1920, 1080), "bus_2160p": (3840, 2160)}
RESOLUTION_QUALITY = 90


def make_resolution_fixture(source, destination, size, quality=RESOLUTION_QUALITY):
    """Resize `source` to `size` (width, height) and write a JPEG to `destination`."""
    source, destination = Path(source), Path(destination)
    image = cv2.imread(str(source))
    if image is None:
        raise ValueError(f"could not read fixture source: {source}")
    resized = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), resized, [cv2.IMWRITE_JPEG_QUALITY, quality]):
        raise ValueError(f"could not write resolution fixture: {destination}")
    return destination
