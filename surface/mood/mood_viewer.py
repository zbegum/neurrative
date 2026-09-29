"""
The mood surface in 3-D, with the emotions as its vertical axis and the
narrative arc lying on it.

Each paragraph's six scores become one mood value on the spectrum
sadness -> danger -> confusion -> curiosity -> wonder -> humor (mood.py), one
Gaussian surface is fitted to them over the PCA plane (surface.py), and the arc
of 40-paragraph windows is drawn on that surface: every point of the curve sits
at the surface's height where the story is. The vertical axis is labelled with
the emotions at their places on the spectrum. The text and the mood over
reading order sit beside and below.

Example:

python surface/mood/mood_viewer.py
"""

import argparse
import os
import sys

import numpy as np

# The repository root and common/ on the path, wherever this script lives.
_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_ROOT, "common")):
  _ROOT = os.path.dirname(_ROOT)
sys.path[1:1] = [_ROOT, os.path.join(_ROOT, "common")]

import mood as mood_mod
import surface as mood_surface
import viewer as V

# window arithmetic from its one definition (arc/narrative_arc/windows.py); this
# folder's own data.py would shadow common/data.py, which common/windows.py needs
sys.path.insert(1, os.path.join(_ROOT, "arc"))
from narrative_arc.windows import DEFAULT_SIZE, DEFAULT_STRIDE, window_bounds  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def book_data(book, model, blend, h):
  data, S = V.reading(book)
  X = V.pca_coords(book, model)
  extent, gx, gy = V.plane(X)
  m, order, positions, _ = mood_mod.mood(S, V.EMOTIONS, blend=blend)
  _, _, _, height_at = mood_surface.fit(X, m, h)
  Q = np.array([(a, b) for b in gy for a in gx])
  bounds = window_bounds(len(X), DEFAULT_SIZE, DEFAULT_STRIDE)
  data.update({
    "timeline": V.rounded(V.gaussian_filter1d(m, V.TIMELINE_SIGMA, mode="nearest")),
    "order": order, "positions": [float(p) for p in positions],
    "n": len(gx), "extent": extent,
    "support": V.rounded(V.footprint(X, gx, gy), 2),
    "field": V.rounded(height_at(Q)),
    "arc": [[round(float(X[a:b, 0].mean()), 4), round(float(X[a:b, 1].mean()), 4),
             (a + b - 1) / 2.0] for a, b in bounds],
  })
  return data


def main():
  ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
  ap.add_argument("--model", default="bge-m3")
  ap.add_argument("--books", nargs="+", default=V.SCORED_BOOKS)
  ap.add_argument("--blend", default="banded", choices=mood_mod.BLENDS)
  ap.add_argument("--h", default=0.2, type=float, help="surface bandwidth (standardized units)")
  ap.add_argument("--output", default=os.path.join(HERE, "output", "mood_viewer.html"))
  args = ap.parse_args()
  data = {"books": {b: book_data(b, args.model, args.blend, args.h) for b in args.books}}
  V.render(os.path.join(HERE, "mood_viewer_template.html"), data, args.output)


if __name__ == "__main__":
  main()
