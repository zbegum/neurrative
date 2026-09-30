"""The windowed series: one definition of the timeseries every arc script draws.

Row i of embeddings.npy is paragraph i in reading order, so sliding a window over
the rows and mean-pooling each one turns a book into a short, ordered sequence of
vectors. That sequence -- not any particular 2-D picture of it -- is what the arc
scripts are actually about. PCA, UMAP and t-SNE are three drawings of the same
series, which is the whole reason they are compared: agreement across them is
evidence about the series, not about a projection.

Two problems this module fixes.

**Four definitions of one window.** `window_bounds` was copy-pasted into
`narrative_arc.py` (since replaced by `arc/`), `arc_on_surface.py`,
`arc_comparison.py` and `arc_comparison_projections.py`, and nine more scripts
imported one of those copies. They had already diverged: only the `narrative_arc` copy checked
`size > n`, and the other three raised `IndexError` on `starts[-1]` instead of
saying what was wrong. The bounds live here now and every caller imports them.

**The series was never saved.** `narrative_arc.py` wrote its *projected*
coordinates and nothing else, so the pooled vectors, the window ranges and the
pooled emotion scores were re-derived from scratch by every script that wanted
them, and could not be cited on their own. `build()` computes the series once and
`save()` writes it to

    arc/output/<book>/<model>/windows/w40_s20[_l2]/series.npz

alongside the usual `params.json`. `load()` reads it back; `series()` reads it if
it is there and builds it if it is not, so a caller never has to know which.

The implementation lives in the arc package (arc/narrative_arc/windows.py);
this module fixes its data and output folders for the rest of the repository.

The canonical setting is w40 s20 -- windows of 40 paragraphs stepping by 20, i.e.
half-overlapping. That is what `--size`/`--stride` default to here.

Everything is stored at window resolution. Paragraph-level data is already on
disk in `books/`, so duplicating it here would just be a second copy that can go
stale; `starts`/`stops` are the join back to it.

Build the canonical series for every book and model:

    python common/windows.py --all-books --all-models
"""

import argparse
import os
import sys

import paths

# One definition of a window, shared with the arc package in arc/: the bounds,
# centres and pooling are imported from there rather than kept as a copy.
sys.path.insert(1, os.path.join(paths.ROOT, "arc"))
# Windows of 40 paragraphs stepping by 20 (DEFAULT_SIZE / DEFAULT_STRIDE): wide
# enough that a window is a scene rather than a remark, and half-overlapping so
# the series moves smoothly instead of jumping.
from narrative_arc.windows import (DEFAULT_SIZE, DEFAULT_STRIDE, pool,  # noqa: E402
                                   window_bounds, window_centers)

# One implementation of the saved series, in the arc package: these wrappers
# only fix the data directory (books/) and the output root (arc/output/).
from narrative_arc import windows as _arc  # noqa: E402
from narrative_arc import paths as _arc_paths  # noqa: E402

Series = _arc.Series
SERIES_FILE = _arc.SERIES_FILE
DATA_DIR = os.path.join(paths.ROOT, "books")
OUTPUT_DIR = _arc_paths.DEFAULT_OUTPUT_DIR


def build(book, model, size=DEFAULT_SIZE, stride=DEFAULT_STRIDE, l2=False):
  """Compute the series from `books/`, without touching `output/`."""
  return _arc.build(DATA_DIR, book, model, size, stride, l2)


def series_dir(book, model, size=DEFAULT_SIZE, stride=DEFAULT_STRIDE, l2=False,
               create=True):
  """`arc/output/<book>/<model>/windows/w40_s20[_l2]/`."""
  return _arc.series_dir(OUTPUT_DIR, book, model, size, stride, l2, create)


def save(series, args=None):
  """Write `series.npz` (+ `params.json`) and return the directory."""
  return _arc.save(series, OUTPUT_DIR, __file__, args)


def load(book, model, size=DEFAULT_SIZE, stride=DEFAULT_STRIDE, l2=False):
  """Read a saved series, or None if it has not been built."""
  return _arc.load(OUTPUT_DIR, book, model, size, stride, l2)


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
