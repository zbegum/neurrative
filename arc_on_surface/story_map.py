"""
The story map: the narrative arc travelling across an emotion landscape, with the
text it is passing through beside it and the emotion over reading order below.

The landscape is one of the saved surfaces (surface/kernel Gaussian, surface/
bspline, surface/poisson, or the mood surface), seen flat as contour bands or in
3-D with the arc riding on it. The arc is the 40-paragraph windows in the PCA
plane. Drag the timeline, click the map, arrows step, space plays.

Needs, for each book: projection/embedding.py, and the surface scripts whose
surfaces should appear (a method with no saved surface is left out).

Example:

python arc_on_surface/story_map.py
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

import viewer as V
from windows import DEFAULT_SIZE, DEFAULT_STRIDE, window_bounds

HERE = os.path.dirname(os.path.abspath(__file__))
# method key -> (figure family of its saved surface, name on the page)
METHODS = {"kernel": "surface/gaussian_nw", "bspline": "surface/bspline_ls",
           "poisson": "surface/poisson_open"}


def book_data(book, model):
  data, S = V.reading(book)
  X = V.pca_coords(book, model)
  extent, gx, gy = V.plane(X)
  fields = {}
  for key, figure in METHODS.items():
    per = {}
    for e in V.EMOTIONS:
      f = V.saved_field(book, model, figure, e)
      if f:
        per[e] = V.rounded(V.resample(f, gx, gy))
    if per:
      fields[key] = per
  mood, m, order, positions = V.mood_field(X, S, gx, gy)
  fields["mood"] = {"*": V.rounded(mood)}
  bounds = window_bounds(len(X), DEFAULT_SIZE, DEFAULT_STRIDE)
  data.update({
    "x": V.rounded(X[:, 0], 4), "y": V.rounded(X[:, 1], 4),
    "timeline": dict(V.timelines(S), mood=V.rounded(
      V.gaussian_filter1d(m, V.TIMELINE_SIGMA, mode="nearest"))),
    "moodTicks": [order, positions],
    "n": len(gx), "extent": extent, "support": V.rounded(V.footprint(X, gx, gy), 2),
    "fields": fields,
    "arc": [[round(float(X[a:b, 0].mean()), 4), round(float(X[a:b, 1].mean()), 4),
             (a + b - 1) / 2.0] for a, b in bounds],
  })
  return data


def main():
  ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
  ap.add_argument("--model", default="bge-m3")
  ap.add_argument("--books", nargs="+", default=V.SCORED_BOOKS)
  ap.add_argument("--output", default=os.path.join(HERE, "output", "story_map.html"))
  args = ap.parse_args()
  data = {"emotions": V.EMOTIONS,
          "books": {b: book_data(b, args.model) for b in args.books}}
  V.render(os.path.join(HERE, "story_map_template.html"), data, args.output)


if __name__ == "__main__":
  main()
