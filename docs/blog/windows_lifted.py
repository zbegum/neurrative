"""Figure 5b for the blog post: the windows placed on the mood surface.

The companion of 5_paragraphs_lifted.png, drawn the same way (grey relief, so
colour is reading time only). Kept apart from figures.py so the other figures
are not redrawn.

python docs/blog/windows_lifted.py
"""

import matplotlib.pyplot as plt
import numpy as np

from figures import (BOOK, INK, MODEL, SIZE, STRIDE, TIME, draped,
                     grey_relief, mood, mood_data, mood_surface, save, surface_axes, windows)


def main():
  X = mood_data.load_coords(BOOK, MODEL, "pca")
  _, emotions, scores, _, _ = mood_data.load_book(BOOK)
  m, order, positions, _ = mood(scores, emotions)
  GX, GY, Z, height_at = mood_surface.fit(X, m, h=0.2, resolution=260)

  P = windows(X, SIZE, STRIDE)
  W3 = np.column_stack([P, height_at(P) + 0.006])

  grey = grey_relief(GX, GY, Z)

  fig = plt.figure(figsize=(13, 5.6))
  for k, title in enumerate(["windows, joined in reading order", "windows as dots"]):
    ax = fig.add_subplot(1, 2, k + 1, projection="3d", computed_zorder=False)
    ax.plot_surface(GX, GY, Z, facecolors=grey, rstride=2, cstride=2, lw=0,
                    antialiased=True, shade=False)
    if k == 0:
      draped(ax, P, 1.4, height_at, samples=16)
    ax.scatter(*W3.T, c=np.linspace(0, 1, len(P)), cmap=TIME, s=12 if k == 0 else 30,
               edgecolor="white", lw=0.5, depthshade=False)
    surface_axes(ax, positions, order)
    ax.set_title(title, fontsize=10, color=INK)
  save(fig, "5b_windows_lifted.png")


if __name__ == "__main__":
  main()
