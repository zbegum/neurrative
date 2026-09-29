"""The mood surface as a rotatable HTML page, with every paragraph inspectable.

The static PNG shows the terrain or the points but never which paragraph a bump
is. Here each point carries its reading position, mood, dominant emotion, all
six scores and the annotator's one-line summary, so a feature of the surface
can be traced back to the text that made it.

Two colourings of the same points, one visible at a time: by reading order
(where the beginning, middle and end sit on the landscape), and by dominant
emotion (click a legend entry).
"""

import numpy as np
import plotly.graph_objects as go


def page(GX, GY, Z, coords, m, winner, order, positions, chapters, summaries,
         emotions, scores, title, plane):
    """Build the figure. `winner` indexes `order`, as returned by mood.mood()."""
    fig = go.Figure()
    # Grey terrain: the height already carries the mood, and a coloured surface
    # would swallow the points, which are the thing to inspect here.
    fig.add_surface(x=GX, y=GY, z=Z, colorscale="Greys", cmin=-0.15, cmax=1.15,
                    opacity=0.55, showscale=False, name="mood surface",
                    showlegend=True,
                    hovertemplate="x %{x:.3f}<br>y %{y:.3f}<br>"
                                  "mood %{z:.3f}<extra>surface</extra>")

    text = []
    for i in range(len(coords)):
        row = "  ".join(f"{e} {scores[i, emotions.index(e)]:.2f}" for e in order)
        text.append(f"paragraph {i} &middot; chapter {chapters[i]}"
                    f"<br>mood {m[i]:.3f} &middot; dominant {order[winner[i]]}"
                    f"<br>{row}<br><br>{summaries[i]}")

    fig.add_scatter3d(
        x=coords[:, 0], y=coords[:, 1], z=m, mode="markers",
        marker=dict(size=3.4, color=np.arange(len(coords)), colorscale="Plasma",
                    showscale=True, line=dict(width=0),
                    colorbar=dict(title="reading<br>position", len=0.55,
                                  thickness=14)),
        name="paragraphs, by reading order", text=text, hoverinfo="text")

    for k, e in enumerate(order):
        sel = winner == k
        if not sel.any():
            continue
        fig.add_scatter3d(
            x=coords[sel, 0], y=coords[sel, 1], z=m[sel], mode="markers",
            marker=dict(size=3.4, line=dict(width=0)),
            name=f"{e} ({int(sel.sum())})", visible="legendonly",
            text=[t for t, s in zip(text, sel) if s], hoverinfo="text")

    axes = ("PC1", "PC2") if plane == "pca" else ("UMAP-1", "UMAP-2")
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title=axes[0], yaxis_title=axes[1], zaxis_title="mood",
                   zaxis=dict(range=[0, 1], tickvals=list(positions),
                              ticktext=list(order)),
                   aspectmode="cube"),
        legend=dict(orientation="h", y=-0.04),
        margin=dict(l=0, r=0, t=80, b=0), height=820)
    return fig
