"""Sparse, local evidence that a changed picture or a known result graphic stayed on screen."""

from __future__ import annotations

import math

import cv2
import numpy as np

from clipper import config, media


def result_reference(settings):
    """Load the operator's exact crop. A missing or unsuitable reference must fail visibly."""
    spec = settings["result_card"]
    if spec["template"] is None and spec["roi"] is None:
        return None
    box = spec["roi"]
    if (not spec["template"] or not isinstance(box, (list, tuple)) or len(box) != 4
            or not all(type(v) in (float, int) and math.isfinite(v) for v in box)):
        raise ValueError("result_card needs a template and normalized [x, y, width, height]")
    x, y, w, h = box
    if not (0 <= x < x+w <= 1 and 0 <= y < y+h <= 1 and 0 < spec["similarity"] <= 1):
        raise ValueError("invalid result_card ROI or similarity")
    path = config.project_root() / spec["template"]
    try:
        ref = cv2.cvtColor(media.read_image(path), cv2.COLOR_BGR2GRAY)
    except (OSError, cv2.error, media.MediaError) as exc:
        raise ValueError(f"Cannot read result_card template {path}") from exc
    if min(ref.shape) < 3 or ref.std() < 5:
        raise ValueError("result_card template needs a visible, contrasting pattern")
    return ref


class PictureHold:
    """Keep only the before/after pair for the most recent change on the gameplay grid."""

    def __init__(self, cfg, n):
        self.cfg = cfg
        self.until = np.zeros(n, np.float64)
        self.before = self.after = None
        self.start = 0

    def observe(self, i, pixels, previous, hop, sample_end):
        def distance(left, right):
            return np.mean(cv2.absdiff(left, right) > 20)

        if self.after is not None:
            if (distance(pixels, self.after) <= self.cfg["quiet_still_fraction"]
                    and distance(pixels, self.before) >= self.cfg["quiet_change_fraction"]):
                if sample_end - self.start * hop >= self.cfg["hold_s"]:
                    self.until[self.start] = sample_end
                return
            transient_return = (i * hop - self.start * hop < self.cfg["hold_s"]
                                and distance(pixels, self.before) <= self.cfg["quiet_still_fraction"])
            self.before = self.after = None
            if transient_return:
                return
        if previous is not None and distance(pixels, previous) >= self.cfg["quiet_change_fraction"]:
            self.start, self.before, self.after = i, previous, pixels


def hard_cuts(histograms, indices, threshold, n):
    """A new histogram must differ on both sides. A one-sample flash and its return are excluded."""
    cuts = np.zeros(n, dtype=bool)
    h = np.asarray(histograms)
    if len(h) >= 4:
        left, right = h[1:-2], h[2:-1]
        distance = lambda a, b: np.abs(a - b).sum(axis=1) / 2
        strong = np.minimum.reduce([distance(left, right), distance(h[:-3], right), distance(left, h[3:])])
        cuts[np.asarray(indices[2:-1])] = strong >= threshold
    return cuts


def confirmed_card_samples(visible, indices, hop, hold_s):
    """Interpolate a card run only after separate decoded frames confirm persistence."""
    timeline = np.zeros_like(visible)
    if not indices:
        return timeline
    edges = np.diff(np.r_[False, visible[indices], False].astype(np.int8))
    for left, right in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        start, last_seen = indices[left], indices[right - 1]
        if (last_seen - start) * hop >= hold_s:
            stop = indices[right] if right < len(indices) else len(visible)
            timeline[start:stop] = True
    return timeline


def card_appearances(visible, hop, duration, hold_s):
    """Keep new, persistent appearances. A HUD already visible at the first sample is background."""
    edges = np.diff(np.r_[False, visible, False].astype(np.int8))
    return {int(start): float(min(duration, stop * hop))
            for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))
            if start > 0 and stop - start >= 2 and min(duration, stop * hop) - start * hop >= hold_s}
