"""The windowed series: one definition of the timeseries every arc script draws.

Row i of embeddings.npy is paragraph i in reading order, so sliding a window over
the rows and mean-pooling each one turns a book into a short, ordered sequence of
vectors. That sequence -- not any particular 2-D picture of it -- is what the arc
scripts are actually about. PCA, UMAP and t-SNE are three drawings of the same
series, which is the whole reason they are compared: agreement across them is
evidence about the series, not about a projection.

Two problems this module fixes.

**Four definitions of one window.** `window_bounds` was copy-pasted into
`narrative_arc.py` (since replaced by `narrative-arc/`), `arc_on_surface.py`,
`arc_comparison.py` and `arc_comparison_projections.py`, and nine more scripts
imported one of those copies. They had already diverged: only the `narrative_arc` copy checked
`size > n`, and the other three raised `IndexError` on `starts[-1]` instead of
saying what was wrong. The bounds live here now and every caller imports them.

**The series was never saved.** `narrative_arc.py` wrote its *projected*
coordinates and nothing else, so the pooled vectors, the window ranges and the
pooled emotion scores were re-derived from scratch by every script that wanted
them, and could not be cited on their own. `build()` computes the series once and
`save()` writes it to

    output/<book>/<model>/windows/w40_s20[_l2on]/series.npz

alongside the usual `params.json`. `load()` reads it back; `series()` reads it if
it is there and builds it if it is not, so a caller never has to know which.

The canonical setting is w40 s20 -- windows of 40 paragraphs stepping by 20, i.e.
half-overlapping. That is what `--size`/`--stride` default to here.

Everything is stored at window resolution. Paragraph-level data is already on
disk in `books/`, so duplicating it here would just be a second copy that can go
stale; `starts`/`stops` are the join back to it.

Build the canonical series for every book and model:

    python visualization/windows.py --all-books --all-models
"""

import argparse
import os

import numpy as np

import paths
from data import load_paragraphs, load_scores

# Windows of 40 paragraphs stepping by 20. Wide enough that a window is a scene
# rather than a remark, and half-overlapping so consecutive windows share half
# their text and the series moves smoothly instead of jumping.
DEFAULT_SIZE = 40
DEFAULT_STRIDE = 20

SERIES_FILE = "series.npz"


def window_bounds(n, size, stride):
  """Start/stop indices of each window over n paragraphs, in reading order.

  Windows step by `stride` until one would run past the end; a final window is
  pinned to [n - size, n) so the closing paragraphs are never dropped. Returns a
  list of half-open (start, stop) pairs.
  """
  if size < 1 or stride < 1:
    raise ValueError(f"size and stride must be >= 1 (got {size}, {stride}).")
  if size > n:
    raise ValueError(f"window size {size} > {n} paragraphs.")

  starts = list(range(0, n - size + 1, stride))
  if starts[-1] != n - size:
    starts.append(n - size)
  return [(s, s + size) for s in starts]


def window_centers(windows):
  """The paragraph index at the middle of each window -- what the window 'is at'.

  Half-integral when the window is even-length, which is correct: the centre of
  [0, 40) is 19.5, and rounding it here would quietly bias every window the same
  direction. Callers that need an actual paragraph round at the point of use.
  """
  return np.array([(s + e - 1) / 2.0 for s, e in windows])


def pool(rows, windows, l2=False):
  """Mean-pool each window's rows into one vector.

  `l2` normalizes each row first, so the pooling is by direction rather than
  magnitude -- which matches how these embeddings are compared (cosine). Off by
  default so the raw manifold is what gets projected unless asked otherwise, and
  meaningless for score rows, which are already on a fixed scale.
  """
  if l2:
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    rows = rows / np.clip(norms, 1e-12, None)
  return np.array([rows[s:e].mean(axis=0) for s, e in windows])


class Series:
  """One book × model as an ordered sequence of windows.

  Attributes, all row-aligned and in reading order:

    starts, stops   the paragraph range [start, stop) each window covers -- the
                    join back to `books/<book>/processed.json`
    centers         the middle paragraph index of each window, possibly .5
    pooled          (n_windows, d) mean-pooled embeddings, the series itself
    emotions        the emotion names, in the order `scores` columns are in
    scores          (n_windows, n_emotions) mean-pooled annotation scores, or
                    None when the book has no paragraph_scores.json
    chapters        the chapter (or act) of each window's middle paragraph
    center_ids      the paragraph id at each centre, so a window on a plot can be
                    traced back to text without recomputing the bounds
  """

  def __init__(self, book, model, size, stride, l2, starts, stops, centers,
               pooled, emotions, scores, chapters, center_ids):
    self.book, self.model = book, model
    self.size, self.stride, self.l2 = size, stride, l2
    self.starts, self.stops, self.centers = starts, stops, centers
    self.pooled = pooled
    self.emotions, self.scores = emotions, scores
    self.chapters, self.center_ids = chapters, center_ids

  def __len__(self):
    return len(self.starts)

  @property
  def windows(self):
    """The (start, stop) pairs, for the callers that still want the list form."""
    return list(zip(self.starts.tolist(), self.stops.tolist()))

  def emotion(self, name):
    """The pooled series for one emotion, as a 1-D array over windows."""
    if self.scores is None:
      raise ValueError(f"{self.book} has no scores; cannot read {name!r}.")
    if name not in self.emotions:
      raise ValueError(f"Unknown emotion {name!r}. Have: {', '.join(self.emotions)}")
    return self.scores[:, self.emotions.index(name)]


def build(book, model, size=DEFAULT_SIZE, stride=DEFAULT_STRIDE, l2=False):
  """Compute the series from `books/`, without touching `output/`."""
  paragraphs = load_paragraphs(book)
  embeddings = np.load(
    os.path.join(paths.ROOT, "books", book, "embeddings", model, "embeddings.npy")
  )
  if len(embeddings) != len(paragraphs):
    raise ValueError(
      f"embeddings ({len(embeddings)}) and paragraphs ({len(paragraphs)}) mismatch."
    )

  windows = window_bounds(len(paragraphs), size, stride)
  centers = window_centers(windows)
  mid = centers.round().astype(int)

  # Scores are optional: the geometry runs on embeddings alone, and a book can be
  # windowed before it has been annotated.
  try:
    emotions, matrix = load_scores(book, paragraphs)
    scores = pool(matrix, windows)
  except FileNotFoundError:
    emotions, scores = [], None

  return Series(
    book, model, size, stride, l2,
    starts=np.array([s for s, _ in windows]),
    stops=np.array([e for _, e in windows]),
    centers=centers,
    pooled=pool(embeddings, windows, l2),
    emotions=emotions,
    scores=scores,
    chapters=np.array([paragraphs[i]["chapter_id"] for i in mid]),
    center_ids=np.array([paragraphs[i]["id"] for i in mid]),
  )


def series_dir(book, model, size=DEFAULT_SIZE, stride=DEFAULT_STRIDE, l2=False,
               create=True):
  """`output/<book>/<model>/windows/w40_s20[_l2on]/`."""
  return paths.out_dir(book, model, paths.WINDOWS,
                       {"w": size, "s": stride, "l2": "on" if l2 else None},
                       create=create)


def save(series, args=None):
  """Write `series.npz` (+ `params.json`) and return the directory."""
  out = series_dir(series.book, series.model, series.size, series.stride,
                   series.l2)
  arrays = dict(
    starts=series.starts, stops=series.stops, centers=series.centers,
    pooled=series.pooled, chapters=series.chapters,
    center_ids=series.center_ids, emotions=np.array(series.emotions),
  )
  if series.scores is not None:
    arrays["scores"] = series.scores
  np.savez(os.path.join(out, SERIES_FILE), **arrays)

  paths.stamp(out, __file__, args, book=series.book, model=series.model,
              size=series.size, stride=series.stride, l2=series.l2,
              n_windows=len(series), n_dims=int(series.pooled.shape[1]),
              emotions=list(series.emotions),
              pooling="mean over the window's rows"
                      + (", L2-normalized first" if series.l2 else ""))
  return out


def load(book, model, size=DEFAULT_SIZE, stride=DEFAULT_STRIDE, l2=False):
  """Read a saved series, or None if it has not been built."""
  path = os.path.join(series_dir(book, model, size, stride, l2, create=False),
                      SERIES_FILE)
  if not os.path.exists(path):
    return None

  z = np.load(path, allow_pickle=False)
  return Series(
    book, model, size, stride, l2,
    starts=z["starts"], stops=z["stops"], centers=z["centers"],
    pooled=z["pooled"], emotions=[str(e) for e in z["emotions"]],
    scores=z["scores"] if "scores" in z.files else None,
    chapters=z["chapters"], center_ids=z["center_ids"],
  )


def series(book, model, size=DEFAULT_SIZE, stride=DEFAULT_STRIDE, l2=False,
           cache=True):
  """The series, read from disk if it is there and built if it is not.

  `cache=False` always rebuilds, for a caller that has just changed how the
  underlying scores or embeddings were produced.
  """
  if cache:
    found = load(book, model, size, stride, l2)
    if found is not None:
      return found
  return build(book, model, size, stride, l2)


def main():
  parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  parser.add_argument("--book", default="alice_wonderland", type=str)
  parser.add_argument("--all-books", action="store_true")
  parser.add_argument("--model", type=str)
  parser.add_argument("--all-models", action="store_true")
  parser.add_argument("--size", default=DEFAULT_SIZE, type=int,
                      help="Window length in paragraphs.")
  parser.add_argument("--stride", default=DEFAULT_STRIDE, type=int,
                      help="Paragraphs between consecutive windows.")
  parser.add_argument("--l2", action="store_true",
                      help="L2-normalize embeddings before pooling.")
  args = parser.parse_args()

  books_dir = os.path.join(paths.ROOT, "books")
  books = (sorted(b for b in os.listdir(books_dir)
                  if os.path.isdir(os.path.join(books_dir, b, "embeddings")))
           if args.all_books else [args.book])

  for book in books:
    if args.all_models:
      models = sorted(os.listdir(os.path.join(books_dir, book, "embeddings")))
    elif args.model:
      models = [args.model]
    else:
      parser.error("pass --model or --all-models")

    for model in models:
      s = build(book, model, args.size, args.stride, args.l2)
      out = save(s, args)
      print(f"{book} / {model}: {len(s)} windows x {s.pooled.shape[1]} dims "
            f"(size {s.size}, stride {s.stride}) -> {os.path.relpath(out, paths.ROOT)}")

  print("\nDone.")


if __name__ == "__main__":
  main()
