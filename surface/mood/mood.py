"""Collapse six emotion scores into one mood value on a heavy -> light spectrum.

This is the single definition of "mood" in the repository. The mood surface
(this folder) uses `mood()` per paragraph; the arc-on-surface scripts
(`arc_on_surface/arc_emotion_axis.py` and its grids) use `normalizer()` and
`softmax_position()` per grid cell of six fitted surfaces.

The spectrum orders the emotions from heavy to light; a paragraph's mood is
where it sits on that line. Before blending, each emotion is normalized over
the whole book, because the raw emotions are not comparable: some sit at 0.4
all book, others near 0.
"""

import numpy as np

SPECTRUM = ["sadness", "danger", "confusion", "curiosity", "wonder", "humor"]
BLENDS = ("banded", "softmax", "dominant", "project", "pc1")


def normalizer(scores, kind="rank"):
    """A per-emotion transform, calibrated on the book's scores for that emotion.

        rank    a value's percentile in this emotion's distribution -- baseline-
                and scale-free, spreads the most (the default).
        zscore  standardize to this emotion's own mean and sd.
        minmax  linear to [0, 1] over this emotion's observed range.

    The scores are quantized, so ties are common. Under `rank` tied scores get
    their average rank: two paragraphs with the same score always get the same
    value, whatever their position in the book.

    Returns a function, so the calibration from the paragraphs can be applied
    to anything else (for example a grid of fitted surface heights).
    """
    x = np.asarray(scores, dtype=np.float64)
    if kind == "rank":
        xs, counts = np.unique(x, return_counts=True)
        if len(xs) == 1:
            return lambda v: np.full(np.shape(v), 0.5)
        midrank = np.cumsum(counts) - (counts + 1) / 2.0   # 0-based average rank
        qs = midrank / (len(x) - 1)
        return lambda v: np.interp(np.asarray(v, dtype=np.float64), xs, qs)
    if kind == "zscore":
        m, s = x.mean(), x.std() or 1.0
        return lambda v: (np.asarray(v, dtype=np.float64) - m) / s
    if kind == "minmax":
        lo, rng = x.min(), (x.max() - x.min()) or 1.0
        return lambda v: (np.asarray(v, dtype=np.float64) - lo) / rng
    raise ValueError(f"unknown norm {kind!r}")


def normalize(scores, kind="rank"):
    """Normalize every column of an (n, k) score matrix against itself."""
    x = np.asarray(scores, dtype=np.float64)
    return np.column_stack([normalizer(x[:, j], kind)(x[:, j])
                            for j in range(x.shape[1])])


def softmax_position(norm_values, positions, temp):
    """Spectrum position from already-normalized values, softmax-weighted.

    The weights are a softmax over emotions of their normalized prominence, so
    the height swings toward whichever emotion is locally elevated. `temp` is
    the temperature: small snaps to the dominant emotion, large averages
    (flat). `norm_values` is (k, ...) and `positions` (k,). NaN cells (outside
    a support mask) get no weight; an all-NaN point comes back NaN.
    """
    v = np.asarray(norm_values, dtype=np.float64)
    shape = (-1,) + (1,) * (v.ndim - 1)
    z = np.where(np.isfinite(v), v, -np.inf)
    z = z - np.nanmax(np.where(np.isfinite(z), z, np.nan), axis=0, keepdims=True)
    w = np.exp(z / temp)
    wsum = w.sum(axis=0)
    return ((positions.reshape(shape) * w).sum(axis=0)
            / np.where(wsum > 0, wsum, np.nan))


def mood(scores, emotions, order=SPECTRUM, blend="banded", norm="rank",
         temp=0.15):
    """Per-paragraph mood in [0, 1], and the spectrum positions for the z axis.

    Five ways of putting six emotions on one height:

        banded    each emotion gets its own band of the axis, centred on its
                  spectrum position; height within the band is how clearly it
                  wins (its margin over the runner-up, ranked within the band).
                  Readable as an emotion and keeps intensity. The default.
        softmax   the softmax-weighted mean of spectrum positions. Readable,
                  but a mean: unless one emotion dominates it sits near the
                  middle, and a higher `temp` flattens it further.
        dominant  the position of the winning emotion. Every height is a named
                  emotion, but a step function with cliffs between plateaus.
        project   the normalized scores dotted with a centred spectrum
                  (sadness -1 ... humor +1): a sum, not a mean, so intensity
                  survives and nothing pulls it to the middle.
        pc1       the first principal component of the normalized scores: the
                  axis the emotions themselves vary along most, sign-aligned
                  so up stays toward humor.

    Returns (mood, order, positions, winner) where `winner` is each paragraph's
    dominant emotion as an index into `order`.
    """
    order = [e for e in order if e in emotions]
    order += [e for e in emotions if e not in order]
    positions = np.linspace(0, 1, len(order))
    cols = [emotions.index(e) for e in order]
    v = normalize(np.asarray(scores)[:, cols], norm)      # (n, k)
    win = np.argmax(v, axis=1)

    if blend == "softmax":
        z = softmax_position(v.T, positions, temp)
    elif blend == "dominant":
        z = positions[win].astype(float)
    elif blend == "banded":
        srt = np.sort(v, axis=1)
        margin = srt[:, -1] - srt[:, -2]
        step = positions[1] - positions[0] if len(positions) > 1 else 1.0
        z = positions[win].astype(float)
        for k in range(len(order)):
            sel = win == k
            if sel.sum() > 1:
                q = np.argsort(np.argsort(margin[sel])) / (sel.sum() - 1)
                z[sel] += (q - 0.5) * step * 0.9
    elif blend == "project":
        signed = np.linspace(-1.0, 1.0, len(order))
        z = v @ signed
    elif blend == "pc1":
        A = v - v.mean(axis=0, keepdims=True)
        _, _, Vt = np.linalg.svd(A, full_matrices=False)
        axis = Vt[0]
        if np.dot(axis, np.linspace(-1.0, 1.0, len(order))) < 0:
            axis = -axis
        z = A @ axis
    else:
        raise ValueError(f"unknown blend {blend!r}; choose from {BLENDS}")

    lo, hi = z.min(), z.max()
    z = (z - lo) / (hi - lo if hi > lo else 1.0)
    return z, order, positions, win
