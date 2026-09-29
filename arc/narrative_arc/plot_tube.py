"""Drawing a Tube: a shaded triangle mesh around the arc, with the polygon at each
window outlined, as a static matplotlib PNG.

Surface color is either `emotion` (each polygon vertex carries its emotion's hue,
blended between neighbouring vertices, so a ridge of one color swells where that
emotion is strong) or `progression` (plasma along reading order).

The render uses true data aspect: the sections are built in data units, and
a cube aspect would squash a round section into an ellipse.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
import numpy as np

# The shared emotion palette.
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from surface.mood.mood import EMOTION_COLOR

LIGHT = np.array([0.35, -0.55, 0.75])
CAP_COLOR = (0.62, 0.61, 0.66)


def vertex_colors(tube, mode):
  """(V, 3) RGB in [0, 1] for every mesh vertex, caps included."""
  S, m, _ = tube.rings.shape
  if mode == "emotion" and tube.labels:
    k = len(tube.labels)
    hues = np.array([to_rgb(EMOTION_COLOR[name]) for name in tube.labels])
    x = tube.angles / (2 * np.pi) * k
    j = np.floor(x).astype(int) % k
    f = (x - np.floor(x))[:, None]
    ring = (1 - f) * hues[j] + f * hues[(j + 1) % k]  # (m, 3)
    colors = np.tile(ring, (S, 1))
  else:
    along = colormaps["plasma"](np.linspace(0, 1, S))[:, :3]
    colors = np.repeat(along, m, axis=0)
  return np.vstack([colors, [CAP_COLOR, CAP_COLOR]])


def shaded_face_colors(tube, colors):
  """Per-face RGB: the quad's mean color times a two-sided Lambert term.

  The side faces come from mesh() as two triangle blocks of (S-1) x m quads.
  Both triangles of a quad take the quad's color and normal -- coloring each
  triangle from its own three vertices alternates between them and draws a
  sawtooth along the tube.
  """
  S, m, _ = tube.rings.shape
  q = (S - 1) * m
  tri = tube.vertices[tube.faces]
  normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
  face_colors = colors[tube.faces].mean(axis=1)

  quad_normals = normals[:q] + normals[q:2 * q]
  quad_colors = (face_colors[:q] + face_colors[q:2 * q]) / 2
  normals[:q] = normals[q:2 * q] = quad_normals
  face_colors[:q] = face_colors[q:2 * q] = quad_colors

  normals /= np.clip(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12, None)
  light = LIGHT / np.linalg.norm(LIGHT)
  shade = 0.5 + 0.5 * np.abs(normals @ light)
  return np.clip(face_colors * shade[:, None], 0, 1)


def _legend(tube, mode):
  if mode == "emotion" and tube.labels:
    return [Patch(color=EMOTION_COLOR[name], label=name) for name in tube.labels]
  return None


def tube_plot(tube, proj, curve, title, output_path, color="emotion",
              show_rings=True, elev=22, azim=-60):
  fig = plt.figure(figsize=(10, 8.5))
  ax = fig.add_subplot(projection="3d")

  colors = vertex_colors(tube, color)
  surface = Poly3DCollection(tube.vertices[tube.faces],
                             facecolors=shaded_face_colors(tube, colors),
                             edgecolors="none", alpha=0.92)
  ax.add_collection3d(surface)

  if show_rings:
    # Pushed 4% outward: matplotlib sorts collections, not pixels, so outlines
    # lying exactly on the surface vanish behind it.
    centers = curve[tube.window_index][:, None, :]
    lifted = centers + 1.04 * (tube.window_rings - centers)
    closed = np.concatenate([lifted, lifted[:, :1]], axis=1)
    segments = np.concatenate([closed[:, :-1, None], closed[:, 1:, None]], axis=2)
    ax.add_collection3d(Line3DCollection(segments.reshape(-1, 2, 3),
                                         colors="#1a1823", linewidths=0.6,
                                         alpha=0.55))

  ax.plot(*proj.coords.T, color="0.45", linewidth=0.6, alpha=0.6)
  ax.scatter(*proj.coords[0], s=120, facecolor="none", edgecolor="black",
             linewidths=1.5, depthshade=False)
  ax.text(*proj.coords[0], "  O", fontsize=10, fontweight="bold")
  ax.scatter(*proj.coords[-1], s=130, color="black", marker="X", depthshade=False)

  pts = np.vstack([tube.vertices, proj.coords])
  lo, hi = pts.min(axis=0), pts.max(axis=0)
  ax.set_xlim(lo[0], hi[0])
  ax.set_ylim(lo[1], hi[1])
  ax.set_zlim(lo[2], hi[2])
  ax.set_box_aspect(np.maximum(hi - lo, 1e-9))  # true proportions

  ax.set_xlabel(proj.labels[0], fontsize=8)
  ax.set_ylabel(proj.labels[1], fontsize=8)
  ax.set_zlabel(proj.labels[2], fontsize=8)
  ax.tick_params(labelsize=7)
  ax.view_init(elev=elev, azim=azim)
  ax.set_title(title)
  handles = _legend(tube, color)
  if handles:
    ax.legend(handles=handles, title="section vertex", loc="upper left",
              fontsize=8, title_fontsize=8, frameon=True, framealpha=0.9)
  fig.tight_layout()
  fig.savefig(output_path, dpi=220)
  plt.close(fig)
  print(f"  wrote {output_path}")
