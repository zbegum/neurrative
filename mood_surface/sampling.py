"""Pick which paragraphs the arc runs through. All return sorted indices."""

import numpy as np


def pick(kind, coords, mood, chapters, n):
    """Dispatch to a strategy by name."""
    if kind == "chapter_medoids":
        return chapter_medoids(coords, chapters)
    if kind == "uniform":
        return uniform(coords, n)
    if kind == "douglas_peucker":
        return douglas_peucker(mood, n)
    if kind == "farthest_point":
        return farthest_point(coords, n)
    raise ValueError(f"unknown sampling {kind!r}")


def _medoid(coords, members):
    P = coords[members]
    D = np.linalg.norm(P[:, None] - P[None], axis=2)
    return members[int(D.sum(1).argmin())]


def chapter_medoids(coords, chapters):
    """One central real paragraph per chapter."""
    return np.array(sorted(_medoid(coords, np.where(chapters == c)[0])
                           for c in np.unique(chapters)))


def uniform(coords, n):
    return np.unique(np.linspace(0, len(coords) - 1, n).astype(int))


def douglas_peucker(mood, keep):
    """Fewest paragraphs that preserve the shape of mood over reading order."""
    n = len(mood)
    t = np.linspace(0, 1, n)
    P = np.column_stack([t, mood])
    chosen, segs = {0, n - 1}, [(0, n - 1)]
    while len(chosen) < keep and segs:
        best = None
        for a, b in segs:
            if b - a < 2:
                continue
            d = P[b] - P[a]
            nn = np.linalg.norm(d)
            dev = (np.abs(np.cross(d / nn, P[a + 1:b] - P[a])) if nn else
                   np.linalg.norm(P[a + 1:b] - P[a], axis=1))
            i = int(dev.argmax()) + a + 1
            if best is None or dev.max() > best[0]:
                best = (dev.max(), i, a, b)
        if best is None:
            break
        _, i, a, b = best
        chosen.add(i)
        segs = [s for s in segs if s != (a, b)] + [(a, i), (i, b)]
    return np.array(sorted(chosen))


def farthest_point(coords, n):
    """Spread points across the plane so the arc visits every region."""
    sel = [int(np.linalg.norm(coords - coords.mean(0), axis=1).argmax())]
    d = np.linalg.norm(coords - coords[sel[0]], axis=1)
    while len(sel) < n:
        i = int(d.argmax())
        sel.append(i)
        d = np.minimum(d, np.linalg.norm(coords - coords[i], axis=1))
    return np.array(sorted(set(sel) | {0, len(coords) - 1}))
