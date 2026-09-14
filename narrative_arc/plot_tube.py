"""Drawing a Tube: a shaded triangle mesh around the arc, with the polygon at each
window outlined, as a static matplotlib PNG and a rotatable plotly HTML.

Surface color is either `emotion` (each polygon vertex carries its emotion's hue,
blended between neighbouring vertices, so a ridge of one color swells where that
emotion is strong) or `progression` (plasma along reading order).

Both renderings use true data aspect: the sections are built in data units, and
a cube aspect would squash a round section into an ellipse.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
import numpy as np

from .data import EMOTION_STYLE

LIGHT = np.array([0.35, -0.55, 0.75])
CAP_COLOR = (0.62, 0.61, 0.66)


def vertex_colors(tube, mode):
  """(V, 3) RGB in [0, 1] for every mesh vertex, caps included."""
  S, m, _ = tube.rings.shape
  if mode == "emotion" and tube.labels:
    k = len(tube.labels)
    hues = np.array([to_rgb(EMOTION_STYLE[name][0]) for name in tube.labels])
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
    return [Patch(color=EMOTION_STYLE[name][0], label=name) for name in tube.labels]
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


def tube_html(tube, proj, curve, hover, title, output_path, color="emotion",
              show_rings=True):
  import plotly.graph_objects as go

  colors = vertex_colors(tube, color)
  rgb = [f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})" for r, g, b in colors]
  V, F = tube.vertices, tube.faces
  traces = [go.Mesh3d(
    x=V[:, 0], y=V[:, 1], z=V[:, 2], i=F[:, 0], j=F[:, 1], k=F[:, 2],
    vertexcolor=rgb, opacity=0.95, flatshading=False, hoverinfo="skip",
    lighting=dict(ambient=0.55, diffuse=0.7, specular=0.15, roughness=0.7),
    lightposition=dict(x=100, y=-200, z=300), name="tube",
  )]

  if show_rings:
    xs, ys, zs = [], [], []
    for ring in tube.window_rings:
      loop = np.vstack([ring, ring[:1]])
      xs += loop[:, 0].tolist() + [None]
      ys += loop[:, 1].tolist() + [None]
      zs += loop[:, 2].tolist() + [None]
    traces.append(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines",
                               line=dict(color="rgba(26,24,35,0.55)", width=2),
                               hoverinfo="skip", name="window sections"))

  c = proj.coords
  traces.append(go.Scatter3d(
    x=c[:, 0], y=c[:, 1], z=c[:, 2], mode="markers+lines",
    line=dict(color="rgba(90,88,100,0.5)", width=2),
    marker=dict(size=2.5, color="rgba(26,24,35,0.8)"),
    hovertext=hover, hoverinfo="text", name="windows"))
  for i, label, symbol in ((0, "O", "circle-open"), (len(c) - 1, "X", "x")):
    traces.append(go.Scatter3d(
      x=[c[i, 0]], y=[c[i, 1]], z=[c[i, 2]], mode="markers+text", text=[label],
      textposition="top center", marker=dict(size=7, color="black", symbol=symbol),
      hoverinfo="skip", showlegend=False))

  if color == "emotion" and tube.labels:
    for name in tube.labels:
      traces.append(go.Scatter3d(x=[None], y=[None], z=[None], mode="markers",
                                 marker=dict(size=8, color=EMOTION_STYLE[name][0]),
                                 name=name))

  fig = go.Figure(traces)
  fig.update_layout(
    title=dict(text=title.replace("\n", "<br>"), x=0.5),
    template="plotly_white", height=820, margin=dict(l=0, r=0, t=70, b=0),
    scene=dict(xaxis_title=proj.labels[0], yaxis_title=proj.labels[1],
               zaxis_title=proj.labels[2], aspectmode="data"),
    legend=dict(itemsizing="constant"),
  )
  fig.write_html(output_path, include_plotlyjs="cdn")
  print(f"  wrote {output_path}")
