"""The narrative arc in 3-D: windows projected to three components and joined in
reading order.

The same drawing as plot_2d -- gray path, direction cues, O at the opening, X at
the ending, windows colored by a ColorSpec, optional fitted curve -- with a third
projected axis instead of a flat plane. Two renderings:

  static       matplotlib PNG from one fixed viewpoint (--elev / --azim)
  interactive  plotly HTML you can rotate, with hover showing each window's
               paragraph range, chapter, color value and the opening of its
               middle paragraph -- so a point can be read back as text

How to read it: for PCA the third axis is the next direction of variance and the
shape is real geometry (within the variance captured). For UMAP and t-SNE all
three axes are non-metric; only which windows sit near which is signal, not
distances, angles or the overall shape.
"""

import html
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np

from .data import UNCLEAR

PATH_GRAY = "#8c8c8c"


def direction_cues(coords, n_cues=12, scale=0.04):
  """Tails and vectors for short arrows along the path, pointing forward.

  In 2-D a chevron is a fixed fraction of its segment. In 3-D that makes
  foreshortened segments' arrows vanish, so each cue is instead a fixed fraction
  of the layout's extent -- capped at half the segment, so it never overshoots
  the windows on either side.
  """
  n = len(coords)
  idx = np.linspace(0, n - 2, min(n_cues, n - 1)).round().astype(int)
  seg = coords[idx + 1] - coords[idx]
  seg_len = np.linalg.norm(seg, axis=1, keepdims=True)
  unit = seg / np.clip(seg_len, 1e-12, None)
  extent = np.ptp(coords, axis=0).max()
  length = np.minimum(scale * extent, 0.5 * seg_len)
  mids = (coords[idx] + coords[idx + 1]) / 2.0
  return mids, unit * length


def window_hover(series, paragraphs, chapters, snippet_chars=160):
  """One hover block per window: range, chapter, and the middle paragraph's start."""
  titles = {c["chapter_id"]: c.get("title", "") for c in chapters}
  by_id = {p["id"]: p for p in paragraphs}
  out = []
  for k in range(len(series)):
    chapter = int(series.chapters[k])
    title = titles.get(chapter, "")
    text = by_id[str(series.center_ids[k])]["text"]
    if len(text) > snippet_chars:
      text = text[:snippet_chars].rsplit(" ", 1)[0] + " ..."
    snippet = "<br>".join(textwrap.wrap(html.escape(text), 60))
    out.append(
      f"<b>window {k}</b> - paragraphs {series.starts[k]}-{series.stops[k] - 1}"
      f"<br>chapter {chapter}" + (f": {html.escape(title)}" if title else "")
      + f"<br><i>{snippet}</i>"
    )
  return out


# --------------------------------------------------------------------------
# static (matplotlib)
# --------------------------------------------------------------------------

def draw_path_3d(ax, coords, faint=False):
  ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], color="0.6",
          linewidth=0.7, alpha=0.35 if faint else 0.75)

  tails, vecs = direction_cues(coords)
  tails = tails - vecs / 2.0  # centre each arrow on its segment midpoint
  ax.quiver(tails[:, 0], tails[:, 1], tails[:, 2],
            vecs[:, 0], vecs[:, 1], vecs[:, 2],
            color="0.45", linewidth=0.9, arrow_length_ratio=0.6,
            normalize=False)

  ax.scatter(*coords[0], s=140, facecolor="none", edgecolor="black",
             linewidths=1.6, marker="o", depthshade=False)
  ax.text(*coords[0], "O", fontsize=9, fontweight="bold",
          ha="center", va="center")
  ax.scatter(*coords[-1], s=160, color="black", marker="X", depthshade=False)


def draw_smooth_curve_3d(ax, fit, show_control=False):
  segments = np.stack([fit.curve[:-1], fit.curve[1:]], axis=1)
  lc = Line3DCollection(segments, cmap="plasma", linewidth=2.6, alpha=0.9)
  lc.set_array(fit.t_norm[:-1])
  ax.add_collection3d(lc)
  if show_control:
    X = fit.control_points
    ax.plot(X[:, 0], X[:, 1], X[:, 2], color="black", linestyle="--",
            linewidth=0.7, marker="s", markersize=3, alpha=0.6)


def draw_points_3d(ax, coords, spec, size=40):
  if spec.categories is None:
    sc = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=spec.values,
                    cmap=spec.cmap, vmin=spec.vmin, vmax=spec.vmax, s=size,
                    alpha=0.9, linewidths=0.3, edgecolors="white",
                    depthshade=False)
    return sc, None

  handles = []
  for code, (label, hue, marker, _symbol) in enumerate(spec.categories):
    mask = spec.values == code
    if not mask.any():
      continue
    recessive = label == UNCLEAR
    handle = ax.scatter(
      coords[mask, 0], coords[mask, 1], coords[mask, 2], c=hue, marker=marker,
      s=size * (0.7 if recessive else 1.0), alpha=0.4 if recessive else 0.9,
      linewidths=0 if recessive else 0.3,
      edgecolors="none" if recessive else "#33322e",
      depthshade=False, label=f"{label} ({int(mask.sum())})",
    )
    handles.append(handle)
  return None, handles


def draw_arc_3d(fig, ax, proj, spec, title, fit=None, elev=22, azim=-60,
                show_control=False, key=True):
  coords = proj.coords
  draw_path_3d(ax, coords, faint=fit is not None)
  if fit is not None:
    draw_smooth_curve_3d(ax, fit, show_control)
  sc, handles = draw_points_3d(ax, coords, spec)

  ax.set_xlabel(proj.labels[0], fontsize=8)
  ax.set_ylabel(proj.labels[1], fontsize=8)
  ax.set_zlabel(proj.labels[2], fontsize=8)
  ax.tick_params(labelsize=7)
  ax.set_title(title)
  ax.view_init(elev=elev, azim=azim)
  if not key:
    return
  if handles:
    ax.legend(handles=handles, title=spec.label, loc="upper left",
              frameon=True, framealpha=0.9, fontsize=8, markerscale=1.2)
  else:
    fig.colorbar(sc, ax=ax, label=spec.label, shrink=0.6, pad=0.1)


def arc_plot_3d(proj, spec, title, output_path, fit=None, elev=22, azim=-60,
                show_control=False):
  fig = plt.figure(figsize=(9, 8))
  ax = fig.add_subplot(projection="3d")
  draw_arc_3d(fig, ax, proj, spec, title, fit, elev, azim, show_control)
  fig.tight_layout()
  fig.savefig(output_path, dpi=250)
  plt.close(fig)
  print(f"  wrote {output_path}")


def arc_grid_3d(projections, spec, title, output_path, fits=None, elev=22,
                azim=-60, show_control=False):
  n = len(projections)
  fits = fits or [None] * n
  fig = plt.figure(figsize=(7.5 * n, 7))
  for i, (proj, fit) in enumerate(zip(projections, fits)):
    ax = fig.add_subplot(1, n, i + 1, projection="3d")
    draw_arc_3d(fig, ax, proj, spec, proj.name, fit, elev, azim, show_control)
  fig.suptitle(title)
  fig.tight_layout()
  fig.savefig(output_path, dpi=200)
  plt.close(fig)
  print(f"  wrote {output_path}")


# --------------------------------------------------------------------------
# interactive (plotly)
# --------------------------------------------------------------------------

def _plotly_value(spec, k):
  if spec.categories is not None:
    return spec.categories[spec.values[k]][0]
  value = spec.values[k]
  return f"{value:.2f}" if spec.vmax is not None else f"{value:g}"


def arc_traces_3d(proj, spec, hover, fit=None, show_legend=True,
                  show_colorbar=True, scene="scene", show_control=False):
  """The plotly traces for one arc, all bound to `scene`."""
  import plotly.graph_objects as go

  coords = proj.coords
  x, y, z = coords.T
  traces = []

  traces.append(go.Scatter3d(
    x=x, y=y, z=z, mode="lines", scene=scene,
    line=dict(color=PATH_GRAY, width=2.5 if fit is None else 1.5),
    opacity=0.8 if fit is None else 0.4,
    hoverinfo="skip", showlegend=False,
  ))

  if fit is not None:
    curve = fit.curve
    traces.append(go.Scatter3d(
      x=curve[:, 0], y=curve[:, 1], z=curve[:, 2], mode="lines", scene=scene,
      line=dict(color=fit.t_norm, colorscale="Plasma", width=6),
      hoverinfo="skip", showlegend=False,
    ))
    if show_control:
      X = fit.control_points
      traces.append(go.Scatter3d(
        x=X[:, 0], y=X[:, 1], z=X[:, 2], mode="lines+markers", scene=scene,
        line=dict(color="black", width=2, dash="dash"),
        marker=dict(size=3, color="black", symbol="square"), opacity=0.6,
        name="control points", hoverinfo="skip", showlegend=show_legend,
      ))

  # Cones sized in data units ("raw"), so an arrow is exactly as long as the cue
  # vector direction_cues computed.
  mids, vecs = direction_cues(coords)
  traces.append(go.Cone(
    x=mids[:, 0], y=mids[:, 1], z=mids[:, 2],
    u=vecs[:, 0], v=vecs[:, 1], w=vecs[:, 2],
    sizemode="raw", sizeref=1.0, anchor="center", scene=scene,
    colorscale=[[0, "#6e6e6e"], [1, "#6e6e6e"]], showscale=False,
    hoverinfo="skip", showlegend=False,
  ))

  labels = [f"{h}<br>{spec.label}: {_plotly_value(spec, k)}"
            for k, h in enumerate(hover)]

  if spec.categories is None:
    traces.append(go.Scatter3d(
      x=x, y=y, z=z, mode="markers", scene=scene,
      marker=dict(
        size=5, color=spec.values, colorscale=spec.cmap.capitalize(),
        cmin=spec.vmin, cmax=spec.vmax, opacity=0.95,
        line=dict(color="white", width=0.5),
        showscale=show_colorbar,
        colorbar=dict(title=spec.label, len=0.6) if show_colorbar else None,
      ),
      hovertext=labels, hoverinfo="text", showlegend=False,
    ))
  else:
    for code, (label, hue, _marker, symbol) in enumerate(spec.categories):
      mask = spec.values == code
      if not mask.any():
        continue
      recessive = label == UNCLEAR
      traces.append(go.Scatter3d(
        x=x[mask], y=y[mask], z=z[mask], mode="markers", scene=scene,
        marker=dict(size=4 if recessive else 5.5, color=hue, symbol=symbol,
                    opacity=0.5 if recessive else 0.95),
        name=f"{label} ({int(mask.sum())})", legendgroup=label,
        showlegend=show_legend,
        hovertext=[labels[k] for k in np.flatnonzero(mask)], hoverinfo="text",
      ))

  traces.append(go.Scatter3d(
    x=[x[0]], y=[y[0]], z=[z[0]], mode="markers+text", scene=scene,
    marker=dict(size=11, color="black", symbol="circle-open",
                line=dict(width=3)),
    text=["O"], textposition="top center", textfont=dict(size=14),
    hovertext=[f"<b>opening</b><br>{hover[0]}"], hoverinfo="text",
    showlegend=False,
  ))
  traces.append(go.Scatter3d(
    x=[x[-1]], y=[y[-1]], z=[z[-1]], mode="markers+text", scene=scene,
    marker=dict(size=7, color="black", symbol="x"),
    text=["X"], textposition="top center", textfont=dict(size=14),
    hovertext=[f"<b>ending</b><br>{hover[-1]}"], hoverinfo="text",
    showlegend=False,
  ))
  return traces


def _scene(proj):
  return dict(
    xaxis_title=proj.labels[0], yaxis_title=proj.labels[1],
    zaxis_title=proj.labels[2], aspectmode="cube",
  )


def interactive_plot_3d(proj, spec, hover, title, output_path, fit=None,
                        show_control=False):
  import plotly.graph_objects as go

  fig = go.Figure(arc_traces_3d(proj, spec, hover, fit,
                                show_control=show_control))
  fig.update_layout(
    title=dict(text=title.replace("\n", "<br>"), x=0.5),
    scene=_scene(proj), template="plotly_white",
    legend=dict(title=spec.label, itemsizing="constant"),
    margin=dict(l=0, r=0, t=70, b=0), height=800,
  )
  fig.write_html(output_path, include_plotlyjs="cdn")
  print(f"  wrote {output_path}")


def interactive_grid_3d(projections, spec, hover, title, output_path, fits=None,
                        show_control=False):
  """All projections as side-by-side rotatable scenes, one shared key."""
  import plotly.graph_objects as go
  from plotly.subplots import make_subplots

  n = len(projections)
  fits = fits or [None] * n
  fig = make_subplots(rows=1, cols=n, specs=[[{"type": "scene"}] * n],
                      subplot_titles=[p.name for p in projections],
                      horizontal_spacing=0.02)
  layout = {}
  for i, (proj, fit) in enumerate(zip(projections, fits)):
    scene = "scene" if i == 0 else f"scene{i + 1}"
    last = i == n - 1
    for trace in arc_traces_3d(proj, spec, hover, fit, show_legend=i == 0,
                               show_colorbar=last, scene=scene,
                               show_control=show_control):
      fig.add_trace(trace)
    layout[scene] = _scene(proj)

  fig.update_layout(
    title=dict(text=title.replace("\n", "<br>"), x=0.5),
    template="plotly_white",
    legend=dict(title=spec.label, itemsizing="constant"),
    margin=dict(l=0, r=0, t=90, b=0), height=700, **layout,
  )
  fig.write_html(output_path, include_plotlyjs="cdn")
  print(f"  wrote {output_path}")
