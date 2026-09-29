"""What both interactive explorers share: the window presets, the per-book
record (chapters, windows, text snippets) and small formatting helpers.

Used by curve/explorer.py and tube/tube_explorer.py.
"""

import numpy as np

from .data import EMOTION_STYLE, UNCLEAR_STYLE, dominant, load_book, load_scores
from .windows import pool, window_bounds, window_centers

WINDOW_SIZES = (10, 20, 40, 80)
STRIDE_FRACTIONS = {"quarter": 0.25, "half": 0.5, "full": 1.0}
SNIPPET_CHARS = 240

BOOK_TITLES = {
  "alice_wonderland": "Alice's Adventures in Wonderland",
  "hamlet": "Hamlet",
  "pride_and_prejudice": "Pride and Prejudice",
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
