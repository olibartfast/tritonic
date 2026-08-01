"""Shared CPU-vs-GPU agreement checks for the Tritonic benchmark harnesses.

Every harness compares a GPU path against the CPU reference for the same fixture.
The comparison is the same in all of them: reduce each side to a canonical set of
detections, require the counts to match, pair them up by class and box overlap,
then bound the per-pair disagreement. Only the *shape* overlap differs -- polygons
for the polygon paths, packed masks for the mask paths, nothing for plain
detection -- so that part is injected.

Keeping one implementation matters: the YOLO11 harness previously carried a
partial copy whose validators were never called, so a GPU path that dropped half
its detections still reported a clean run.
"""

from __future__ import annotations

import math
import statistics

# Two detections of the same class overlapping this much are the same object.
# Used to collapse near-duplicates before counting, so a harmless extra split
# box does not look like a count mismatch.
CANONICAL_IOU = 0.9


def bbox_iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union else 0.0


def canonical(detections, iou_threshold=CANONICAL_IOU):
    """Highest-scoring representative of each distinct object, score-ordered."""
    kept = []
    for detection in sorted(detections, key=lambda item: item["score"], reverse=True):
        if any(
            detection["class_id"] == prior["class_id"]
            and bbox_iou(detection["bbox"], prior["bbox"]) >= iou_threshold
            for prior in kept
        ):
            continue
        kept.append(detection)
    return kept


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)]


def timing_summary_many(documents):
    result = {}
    for key in ("preprocess", "infer", "postprocess", "total"):
        values = [
            sample[key] for document in documents for sample in document["samples_ms"]
        ]
        result[key] = {
            "mean_ms": statistics.fmean(values),
            "median_ms": statistics.median(values),
            "p95_ms": percentile(values, 0.95),
        }
    return result


def timing_summary(document):
    return timing_summary_many([document])


def require_model_family(document, accepted, source):
    """Guard that a per-run document came from the model family we think it did.

    `accepted` is a set so a harness can keep honouring result files written
    before the family naming was corrected.
    """
    family = document.get("model_family")
    if family not in accepted:
        expected = " or ".join(sorted(accepted))
        raise ValueError(f"wrong model family in {source}: {family!r}, expected {expected}")


def compare(
    reference_doc,
    candidate_doc,
    min_box_iou,
    max_score_delta,
    overlap=None,
    min_overlap=0.0,
    overlap_name="overlap",
):
    """Pair a candidate path's detections against the reference and bound the drift.

    `overlap`, when given, is called as overlap(ref, cand) and must return a
    ratio in [0, 1] -- polygon or mask agreement for the segmentation harnesses.

    Raises ValueError on any disagreement; returns the per-match metrics so the
    caller can record them.
    """
    reference = canonical(reference_doc["detections"])
    candidate = canonical(candidate_doc["detections"])
    if len(reference) != len(candidate):
        raise ValueError(
            f"canonical detection count mismatch: {len(reference)} != {len(candidate)}"
        )
    remaining = set(range(len(candidate)))
    matches = []
    for ref in reference:
        choices = [
            index for index in remaining if candidate[index]["class_id"] == ref["class_id"]
        ]
        if not choices:
            raise ValueError(f"missing class {ref['class_id']}")
        index = max(
            choices, key=lambda item: bbox_iou(ref["bbox"], candidate[item]["bbox"])
        )
        cand = candidate[index]
        remaining.remove(index)
        box = bbox_iou(ref["bbox"], cand["bbox"])
        score_delta = abs(ref["score"] - cand["score"])
        # Key order matches the pre-refactor output so existing summary.json files
        # stay byte-identical.
        match = {"class_id": ref["class_id"], "box_iou": box}
        failures = []
        if box < min_box_iou:
            failures.append(f"box_iou={box:.6f}")
        if overlap is not None:
            ratio = overlap(ref, cand)
            match[overlap_name] = ratio
            if ratio < min_overlap:
                failures.append(f"{overlap_name}={ratio:.6f}")
        match["score_delta"] = score_delta
        if score_delta > max_score_delta:
            failures.append(f"score_delta={score_delta:.6f}")
        if failures:
            raise ValueError(
                f"semantic mismatch class={ref['class_id']} " + " ".join(failures)
            )
        matches.append(match)
    return matches
