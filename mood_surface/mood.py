"""Collapse six emotion scores per paragraph into one mood value in [0, 1].

The spectrum orders the emotions from heavy to light; a paragraph's mood is
where it sits on that line. `banded` gives each emotion its own band so the
height is readable as an emotion; `softmax` is a weighted mean and stays near
the middle unless one emotion dominates.
"""

import numpy as np

SPECTRUM = ["sadness", "danger", "confusion", "curiosity", "wonder", "humor"]


def normalize(scores, kind="rank"):
    """Make the emotions comparable, per column, over the whole book."""
    x = np.asarray(scores, dtype=float)
    if kind == "rank":
        out = np.empty_like(x)
        for j in range(x.shape[1]):
            order = np.argsort(np.argsort(x[:, j]))
            out[:, j] = order / max(len(x) - 1, 1)
        return out
    if kind == "zscore":
        return (x - x.mean(0)) / np.where(x.std(0) > 1e-12, x.std(0), 1.0)
    if kind == "minmax":
        rng = np.where(x.ptp(0) > 1e-12, x.ptp(0), 1.0)
        return (x - x.min(0)) / rng
    raise ValueError(f"unknown norm {kind!r}")


def mood(scores, emotions, order=SPECTRUM, blend="banded", norm="rank",
         temp=0.15):
    """Per-paragraph mood in [0, 1], and the spectrum positions for the z axis."""
    order = [e for e in order if e in emotions]
    order += [e for e in emotions if e not in order]
    positions = np.linspace(0, 1, len(order))
    cols = [emotions.index(e) for e in order]
    v = normalize(scores[:, cols], norm)          # (n, k)
    win = np.argmax(v, axis=1)

    if blend == "softmax":
        w = np.exp((v - v.max(1, keepdims=True)) / temp)
        z = (w * positions).sum(1) / w.sum(1)
    elif blend == "dominant":
        z = positions[win]
    elif blend == "banded":
        # winner's band, plus its winning margin ranked inside that band
        srt = np.sort(v, axis=1)
        margin = srt[:, -1] - srt[:, -2]
        step = positions[1] - positions[0]
        z = positions[win].astype(float)
        for k in range(len(order)):
            sel = win == k
            if sel.sum() > 1:
                q = np.argsort(np.argsort(margin[sel])) / (sel.sum() - 1)
                z[sel] += (q - 0.5) * step * 0.9
    else:
        raise ValueError(f"unknown blend {blend!r}")

    lo, hi = z.min(), z.max()
    z = (z - lo) / (hi - lo if hi > lo else 1.0)
    return z, order, positions, win
