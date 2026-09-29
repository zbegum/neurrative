"""
The arc in 3-D: the windowed series in three components, the fitted B-spline
through it, and the narrative tube around it, with the text beside it and a
timeline below.

Reads what arc_3d.py and arc_tube.py saved (window 40, stride 20, the default
fit), for PCA, UMAP and t-SNE. The tube is the PCA emotion tube: one ridge per
emotion, each in that emotion's colour, pushed out where the emotion is strong.

Example (from the repository root):

python arc/curve/arc_viewer.py
"""

import argparse
import glob
import os
import sys

import numpy as np

# arc/ and the repository's common/ on the path.
_ARC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[1:1] = [_ARC, os.path.dirname(_ARC), os.path.join(os.path.dirname(_ARC), "common")]

import viewer as V
from narrative_arc import paths as arc_paths
from surface.mood.mood import EMOTION_COLOR, SPECTRUM
from narrative_arc.windows import window_bounds

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECTIONS = ["pca", "umap", "tsne"]
SIZE, STRIDE, FIT = 40, 20, "fit-d3-c12-l0.1"


def one(pattern):
  hits = sorted(glob.glob(pattern), key=os.path.getmtime)
  return hits[-1] if hits else None


def book_data(book, model):
  data, S = V.reading(book)
  base = os.path.join(arc_paths.DEFAULT_OUTPUT_DIR, book, model)
  n = len(data["text"])
  centers = [(a + b - 1) / 2.0 for a, b in window_bounds(n, SIZE, STRIDE)]
  proj = {}
  for p in PROJECTIONS:
    w = one(os.path.join(base, "arc_3d", p, f"{p}3d_w{SIZE}_s{STRIDE}*.npy"))
    f = one(os.path.join(base, "arc_3d", p, f"fit3d_{p}_w{SIZE}_s{STRIDE}*{FIT}.npz"))
    if w and f:
      W = np.load(w)
      proj[p] = {"windows": np.round(W, 4).tolist(),
                 "curve": np.round(np.load(f)["curve"], 4).tolist()}
  if not proj:
    raise SystemExit(f"no saved 3-D arcs for {book}: run arc/curve/arc_3d.py --book {book} --fit")
  data.update({"centers": centers, "proj": proj})
  if S is not None:
    data["timeline"] = V.timelines(S)
    t = one(os.path.join(base, "arc_tube", "pca",
                         f"tube_pca_w{SIZE}_s{STRIDE}_emotions-t0.35-sm1-c0.15_fit-d3-*.npz"))
    if t:
      d = np.load(t)
      data["tube"] = {"vertices": np.round(d["vertices"], 4).ravel().tolist(),
                      "faces": d["faces"].ravel().tolist(),
                      "labels": [str(x) for x in d["labels"]],
                      "colors": [EMOTION_COLOR[str(x)] for x in d["labels"]]}
  return data


def main():
  ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
  ap.add_argument("--model", default="bge-m3")
  ap.add_argument("--books", nargs="+", default=["alice_wonderland", "pride_and_prejudice", "hamlet"])
  ap.add_argument("--output", default=os.path.join(_ARC, "output", "arc_viewer.html"))
  args = ap.parse_args()
  data = {"emotions": V.EMOTIONS, "spectrum": SPECTRUM, "colors": EMOTION_COLOR,
          "books": {b: book_data(b, args.model) for b in args.books}}
  V.render(os.path.join(HERE, "arc_viewer_template.html"), data, args.output)


if __name__ == "__main__":
  main()
