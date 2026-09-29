"""The windowed series: one definition of the timeseries every arc is drawn from.

Row i of embeddings.npy is paragraph i in reading order, so sliding a window over
the rows and mean-pooling each one turns a book into a short, ordered sequence of
vectors. That sequence -- not any particular picture of it -- is what the arc is
actually about. PCA, UMAP and t-SNE, in 2-D or 3-D, are drawings of the same
series, which is the whole reason they are compared: agreement across them is
evidence about the series, not about a projection.

`build()` computes the series and `save()` writes it to

    <output-dir>/<book>/<model>/windows/w40_s20[_l2]/series.npz

so the thing being drawn can be cited independently of any one projection.

The canonical setting is w40 s20 -- windows of 40 paragraphs stepping by 20, i.e.
half-overlapping.
"""

import os

import numpy as np

from . import paths
from .data import load_book, load_embeddings, load_scores

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
  """One book x model as an ordered sequence of windows.

  Attributes, all row-aligned and in reading order:

    starts, stops   the paragraph range [start, stop) each window covers -- the
                    join back to `<data-dir>/<book>/processed.json`
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
    """The (start, stop) pairs, for callers that want the list form."""
    return list(zip(self.starts.tolist(), self.stops.tolist()))

  def emotion(self, name):
    """The pooled series for one emotion, as a 1-D array over windows."""
    if self.scores is None:
      raise ValueError(f"{self.book} has no scores; cannot read {name!r}.")
    if name not in self.emotions:
      raise ValueError(f"Unknown emotion {name!r}. Have: {', '.join(self.emotions)}")
    return self.scores[:, self.emotions.index(name)]


def build(data_dir, book, model, size=DEFAULT_SIZE, stride=DEFAULT_STRIDE,
          l2=False):
  """Compute the series from the data directory, without writing anything."""
  paragraphs, _ = load_book(data_dir, book)
  embeddings = load_embeddings(data_dir, book, model)
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
    emotions, matrix = load_scores(data_dir, book, paragraphs)
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


def series_dir(output_root, book, model, size=DEFAULT_SIZE,
               stride=DEFAULT_STRIDE, l2=False, create=True):
  """`<output-dir>/<book>/<model>/windows/w40_s20[_l2]/`."""
  return paths.out_dir(output_root, book, model, paths.WINDOWS,
                       paths.window_tag(size, stride, l2), create=create)


def save(series, output_root, script=None, args=None):
  """Write `series.npz` (+ `params.json`) and return the directory."""
  out = series_dir(output_root, series.book, series.model, series.size,
                   series.stride, series.l2)
  arrays = dict(
    starts=series.starts, stops=series.stops, centers=series.centers,
    pooled=series.pooled, chapters=series.chapters,
    center_ids=series.center_ids, emotions=np.array(series.emotions),
  )
  if series.scores is not None:
    arrays["scores"] = series.scores
  np.savez(os.path.join(out, SERIES_FILE), **arrays)

  paths.stamp(out, script or __file__, args, book=series.book,
              model=series.model, size=series.size, stride=series.stride,
              l2=series.l2, n_windows=len(series),
              n_dims=int(series.pooled.shape[1]),
              emotions=list(series.emotions),
              pooling="mean over the window's rows"
                      + (", L2-normalized first" if series.l2 else ""))
  return out


def load(output_root, book, model, size=DEFAULT_SIZE, stride=DEFAULT_STRIDE,
         l2=False):
  """Read a saved series, or None if it has not been built."""
  path = os.path.join(
    series_dir(output_root, book, model, size, stride, l2, create=False),
    SERIES_FILE,
  )
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
