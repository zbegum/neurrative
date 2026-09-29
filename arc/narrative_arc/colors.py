"""What the windows are colored by.

Emotions and chapters are pooled per window the same way the embeddings are, so a
window's color describes the same span of text as its point.
"""

from collections import namedtuple

import numpy as np

ColorSpec = namedtuple("ColorSpec", "name values cmap label vmin vmax")

COLOR_HELP = ("progression, chapter, emotions (one plot per emotion), "
              "or individual emotion names.")


def resolve_colors(names, series):
  """ColorSpecs for the windowed path, one per requested coloring, deduplicated."""
  specs = []

  for name in names:
    if name == "progression":
      specs.append(ColorSpec(
        "progression", np.arange(len(series)), "plasma",
        "reading order (window)", None, None,
      ))
      continue

    if name == "chapter":
      # A window spans paragraphs; it is the chapter of its middle paragraph.
      specs.append(ColorSpec(
        "chapter", series.chapters, "viridis", "chapter", None, None,
      ))
      continue

    if series.scores is None:
      raise ValueError(
        f"--color {name!r} needs emotion scores, but {series.book} has no "
        f"paragraph_scores.json. Use progression or chapter."
      )

    wanted = series.emotions if name == "emotions" else [name]
    for emotion in wanted:
      if emotion not in series.emotions:
        raise ValueError(
          f"Unknown --color {emotion!r}. Available: progression, chapter, "
          f"emotions, {', '.join(series.emotions)}"
        )
      specs.append(ColorSpec(emotion, series.emotion(emotion), "viridis",
                             emotion, 0.0, 1.0))

  seen, unique = set(), []
  for spec in specs:
    if spec.name not in seen:
      seen.add(spec.name)
      unique.append(spec)
  return unique
