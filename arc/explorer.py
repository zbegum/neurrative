"""
Build the interactive explorer: one self-contained HTML page with every
combination of book, model, projection method, window size and stride
precomputed, so the dropdowns switch instantly.

The page shows the 3-D arc, and below it the same points seen along each axis
(1-2, 1-3, 2-3). Hovering a window, or scrubbing the reading-position slider,
marks that window in all four views and shows its paragraph range, chapter and
opening text.

What is precomputed, per book x model x window x stride:

  pca, umap, tsne   3-component coordinates, axis labels, trustworthiness
  smooth, tight     two B-spline fits (bspline-regression) per projection:
                      smooth  n_control ~ n/4, lambda 0.1
                      tight   n_control ~ n/2, lambda 0.01
                    control points included

Window sizes are 10, 20, 40, 80 paragraphs; strides are a quarter, half, or all
of the window. t-SNE perplexity is min(30, n/4) and UMAP n_neighbors min(15, n-1),
as in sweep.py. Everything is computed in parallel, one process per book x model
x window x stride.

Examples:

python explorer.py                                   # the sample book
python explorer.py --data-dir ../neurrative/books    # every book and model there
"""

import argparse
import json
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

# One thread per worker process. Left alone, BLAS and numba (UMAP) each start a
# thread per core inside every worker, and the pool spends its time contending.
# Set before numpy is imported so the worker processes inherit it.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMBA_NUM_THREADS"):
  os.environ.setdefault(_var, "1")

import numpy as np
from sklearn.manifold import trustworthiness

from narrative_arc import paths
from narrative_arc.curves import FitOpts, fit_curve
from narrative_arc.data import (EMOTION_STYLE, UNCLEAR_STYLE, dominant, list_books,
                                list_models, load_book, load_embeddings,
                                load_scores)
from narrative_arc.projections import METHODS, project
from narrative_arc.windows import pool, window_bounds, window_centers

WINDOW_SIZES = (10, 20, 40, 80)
STRIDE_FRACTIONS = {"quarter": 0.25, "half": 0.5, "full": 1.0}
SNIPPET_CHARS = 240
TEMPLATE = os.path.join(paths.ROOT, "narrative_arc", "explorer_template.html")

BOOK_TITLES = {
  "alice_wonderland": "Alice's Adventures in Wonderland",
  "hamlet": "Hamlet",
  "pride_and_prejudice": "Pride and Prejudice",
}


def fit_presets(n):
  return {
    "smooth": FitOpts(int(np.clip(n // 4, 6, 24)), 3, 0.1, "dn", 300),
    "tight": FitOpts(int(np.clip(n // 2, 8, 40)), 3, 0.01, "dn", 300),
  }


def rounded(a):
  """Flat list at 4 significant figures relative to the array's scale -- the
  page only draws these, so more digits are just bytes."""
  a = np.asarray(a, dtype=float)
  scale = np.abs(a).max() or 1.0
  decimals = int(max(0, 3 - np.floor(np.log10(scale))))
  return np.round(a, decimals).ravel().tolist()


def chapter_label(chapter):
  title = (chapter.get("title") or "").strip()
  number = chapter.get("chapter_number")
  if title and not title.startswith("["):
    if number and not title.upper().startswith(("ACT", "CHAPTER", "SCENE")):
      return f"{number}. {title}"
    return title
  return f"Chapter {number}" if number else f"Chapter {chapter['chapter_id'] + 1}"


def stride_for(size, fraction):
  return max(1, int(round(size * fraction)))


# --------------------------------------------------------------------------
# per book: everything that does not depend on the model
# --------------------------------------------------------------------------

def book_record(data_dir, book):
  paragraphs, chapters = load_book(data_dir, book)
  n = len(paragraphs)
  try:
    emotions, matrix = load_scores(data_dir, book, paragraphs)
  except FileNotFoundError:
    emotions, matrix = [], None

  chapter_titles = {c["chapter_id"]: chapter_label(c) for c in chapters}
  windows, used = {}, set()
  for size in WINDOW_SIZES:
    for name, fraction in STRIDE_FRACTIONS.items():
      stride = stride_for(size, fraction)
      bounds = window_bounds(n, size, stride)
      mid = window_centers(bounds).round().astype(int)
      used.update(mid.tolist())
      record = {
        "starts": [s for s, _ in bounds],
        "stops": [e for _, e in bounds],
        "mid": mid.tolist(),
        "chapter": [paragraphs[i]["chapter_id"] for i in mid],
      }
      if matrix is not None:
        codes, _, _ = dominant(emotions, pool(matrix, bounds), min_score=0.2)
        record["dominant"] = codes.tolist()
      windows[f"{size}:{name}"] = record

  def snippet(text):
    text = " ".join(text.split())
    if len(text) > SNIPPET_CHARS:
      text = text[:SNIPPET_CHARS].rsplit(" ", 1)[0] + " …"
    return text

  return {
    "title": BOOK_TITLES.get(book, book.replace("_", " ").title()),
    "paragraphs": n,
    "chapters": {str(k): v for k, v in chapter_titles.items()},
    "emotions": [
      {"name": e, "color": EMOTION_STYLE[e][0]} for e in emotions
    ] + ([{"name": "unclear", "color": UNCLEAR_STYLE[0]}] if emotions else []),
    "windows": windows,
    "text": {str(i): snippet(paragraphs[i]["text"]) for i in sorted(used)},
  }


# --------------------------------------------------------------------------
# per book x model x window x stride: projections and fits (worker process)
# --------------------------------------------------------------------------

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
      proj = project(pooled, [method], n_components=3,
                     neighbors=min(15, n - 1),
                     perplexity=float(min(30, max(5, n // 4))), seed=0)[0]
    except Exception as exc:  # e.g. UMAP's spectral init on a handful of windows
      out[method] = {"error": f"{type(exc).__name__}: {exc}"}
      continue
    k = min(5, n // 2 - 1)
    entry = {
      "labels": list(proj.labels),
      "xyz": rounded(proj.coords),
      "trust": round(float(trustworthiness(pooled, proj.coords, n_neighbors=k)), 3),
      "fits": {},
    }
    for preset, opts in fit_presets(n).items():
      fit = fit_curve(proj.coords, opts, n_eval=240)
      entry["fits"][preset] = {
        "curve": rounded(fit.curve),
        "control": rounded(fit.control_points),
        "n_control": opts.n_control,
        "lambda": opts.lambda_,
      }
    out[method] = entry
  return f"{book}|{model}|{size}:{stride_name}", out


def main():
  parser = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
  parser.add_argument("--data-dir", default=paths.DEFAULT_DATA_DIR)
  parser.add_argument("--output", default=os.path.join(paths.DEFAULT_OUTPUT_DIR,
                                                       "explorer.html"))
  parser.add_argument("--books", nargs="+", help="Default: every book found.")
  parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1))
  args = parser.parse_args()

  books = args.books or list_books(args.data_dir)
  data = {"books": {}, "models": {}, "projections": {}}
  jobs = []
  for book in books:
    print(f"{book}: preparing windows")
    data["books"][book] = book_record(args.data_dir, book)
    data["models"][book] = list_models(args.data_dir, book)
    for model in data["models"][book]:
      for size in WINDOW_SIZES:
        for stride_name in STRIDE_FRACTIONS:
          jobs.append((args.data_dir, book, model, size, stride_name))

  print(f"computing {len(jobs)} book x model x window x stride settings "
        f"on {args.workers} workers ...")
  with ProcessPoolExecutor(max_workers=args.workers) as pool_:
    futures = [pool_.submit(compute, job) for job in jobs]
    for done, future in enumerate(as_completed(futures), 1):
      key, value = future.result()
      data["projections"][key] = value
      errors = [m for m, v in value.items() if "error" in v]
      print(f"  [{done}/{len(jobs)}] {key}"
            + (f"  (unavailable: {', '.join(errors)})" if errors else ""))

  payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
  with open(TEMPLATE, "r") as f:
    page = f.read().replace("/*__ARC_DATA__*/null", payload)
  os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
  with open(args.output, "w") as f:
    f.write(page)
  print(f"\nwrote {args.output} ({os.path.getsize(args.output) / 1e6:.1f} MB)")


if __name__ == "__main__":
  main()
