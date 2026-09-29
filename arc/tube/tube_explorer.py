"""
Build the interactive tube explorer: one self-contained HTML page where book,
model, projection, window, step and section type are dropdowns, and thickness,
smoothing and rounding are live controls.

Python precomputes only what is expensive or needs the data -- per book x model
x window x step x projection:

  xyz       the 3-D window coordinates
  curve     the fitted centerline, 200 samples (smooth preset: n_control ~ n/4,
            lambda 0.1, as arc_tube.py)
  N, B      rotation-minimizing frames at every sample
  bend      radius of curvature at every sample (capped at 10x layout RMS radius)
  frac      each window's position along the curve, non-decreasing
  rms       layout RMS radius, the unit emotion radii are measured in
  spread    PCA only: each window's paragraph covariance ellipse at 12 angles,
            as a standard error, in PCA units

and per book x window x step, the pooled emotion scores. The page then does the
rest of narrative_arc/tube.py in the browser -- Gaussian smoothing across
windows, PCHIP sweep, optional rounding, fold limiting, mesh -- so sliders
respond instantly. (Rounding in the page uses a periodic Catmull-Rom spline, not
arc_tube.py's periodic cubic spline; both pass through the same vertex radii.)

Examples:

python tube/tube_explorer.py                                  # the sample book
python tube/tube_explorer.py   # every book and model
"""

import argparse
import json
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMBA_NUM_THREADS"):
  os.environ.setdefault(_var, "1")

import numpy as np

# arc/ on the path, for the narrative_arc package one level up.
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from narrative_arc import paths
from narrative_arc.explorer_data import (STRIDE_FRACTIONS, WINDOW_SIZES,
                                        book_record, rounded, stride_for)
from narrative_arc import tube as TB
from narrative_arc.curves import FitOpts, fit_curve
from narrative_arc.data import (list_books, list_models, load_book,
                                load_embeddings, load_scores)
from narrative_arc.projections import METHODS, project
from narrative_arc.windows import pool, window_bounds

SAMPLES = 200
SPREAD_SIDES = 12
TEMPLATE = os.path.join(paths.ROOT, "narrative_arc", "tube_explorer_template.html")


class _Series:
  """The two attributes spread_sections reads, without building a full Series."""
  def __init__(self, bounds):
    self.windows = bounds

  def __len__(self):
    return len(self.windows)


def compute(job):
  data_dir, book, model, size, stride_name = job
  warnings.filterwarnings("ignore")
  paragraphs, _ = load_book(data_dir, book)
  embeddings = load_embeddings(data_dir, book, model)
  bounds = window_bounds(len(paragraphs), size,
                         stride_for(size, STRIDE_FRACTIONS[stride_name]))
  pooled = pool(embeddings, bounds)
  n = len(pooled)

  out = {}
  for method in METHODS:
    try:
      proj = project(pooled, [method], n_components=3, neighbors=min(15, n - 1),
                     perplexity=float(min(30, max(5, n // 4))), seed=0)[0]
    except Exception as exc:
      out[method] = {"error": f"{type(exc).__name__}: {exc}"}
      continue

    opts = FitOpts(int(np.clip(n // 4, 6, 24)), 3, 0.1, "dn", 300)
    fit = fit_curve(proj.coords, opts, n_eval=SAMPLES)
    _, N, B = TB.rotation_minimizing_frames(fit.curve)
    frac, index = TB.window_positions(fit, SAMPLES, opts.degree)
    rms = float(np.sqrt(((proj.coords - proj.coords.mean(axis=0)) ** 2)
                        .sum(axis=1).mean()))
    bend = np.minimum(1.0 / np.clip(TB.curvature(fit.curve), 1e-12, None), 10 * rms)

    entry = {
      "labels": list(proj.labels),
      "xyz": rounded(proj.coords),
      "curve": rounded(fit.curve),
      "N": np.round(N, 4).ravel().tolist(),
      "B": np.round(B, 4).ravel().tolist(),
      "bend": rounded(bend),
      "frac": np.round(frac, 4).tolist(),
      "rms": rms,
      "n_control": opts.n_control,
    }
    if method == "pca":
      radii, _ = TB.spread_sections(proj, embeddings, _Series(bounds), N, B, index,
                                    k=SPREAD_SIDES, scale="se")
      entry["spread"] = rounded(radii)
    out[method] = entry
  return f"{book}|{model}|{size}:{stride_name}", out


def emotion_scores(data_dir, book):
  """{window key: flat pooled scores, n x emotions} -- or {} without scores."""
  paragraphs, _ = load_book(data_dir, book)
  try:
    _, matrix = load_scores(data_dir, book, paragraphs)
  except FileNotFoundError:
    return {}
  out = {}
  for size in WINDOW_SIZES:
    for name, fraction in STRIDE_FRACTIONS.items():
      bounds = window_bounds(len(paragraphs), size, stride_for(size, fraction))
      out[f"{size}:{name}"] = np.round(pool(matrix, bounds), 3).ravel().tolist()
  return out


def main():
  parser = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
  parser.add_argument("--data-dir", default=paths.DEFAULT_DATA_DIR)
  parser.add_argument("--output", default=os.path.join(paths.DEFAULT_OUTPUT_DIR,
                                                       "tube_explorer.html"))
  parser.add_argument("--books", nargs="+", help="Default: every book found.")
  parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1))
  args = parser.parse_args()

  books = args.books or list_books(args.data_dir)
  data = {"books": {}, "models": {}, "projections": {}, "samples": SAMPLES}
  jobs = []
  for book in books:
    print(f"{book}: preparing windows")
    record = book_record(args.data_dir, book)
    record["scores"] = emotion_scores(args.data_dir, book)
    data["books"][book] = record
    data["models"][book] = list_models(args.data_dir, book)
    for model in data["models"][book]:
      for size in WINDOW_SIZES:
        for stride_name in STRIDE_FRACTIONS:
          jobs.append((args.data_dir, book, model, size, stride_name))

  print(f"computing {len(jobs)} settings on {args.workers} workers ...")
  with ProcessPoolExecutor(max_workers=args.workers) as pool_:
    futures = [pool_.submit(compute, job) for job in jobs]
    for done, future in enumerate(as_completed(futures), 1):
      key, value = future.result()
      data["projections"][key] = value
      errors = [m for m, v in value.items() if "error" in v]
      if errors or done % 27 == 0 or done == len(jobs):
        print(f"  [{done}/{len(jobs)}] {key}"
              + (f"  (unavailable: {', '.join(errors)})" if errors else ""))

  payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
  with open(TEMPLATE, "r") as f:
    page = f.read().replace("/*__TUBE_DATA__*/null", payload)
  os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
  with open(args.output, "w") as f:
    f.write(page)
  print(f"\nwrote {args.output} ({os.path.getsize(args.output) / 1e6:.1f} MB)")


if __name__ == "__main__":
  main()
