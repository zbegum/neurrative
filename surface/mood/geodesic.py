"""Join waypoints with geodesics that lie on the surface, not chords through it.

Needs the geometry/ solver at the repo root (mesh + exact geodesic).
"""

import os
import sys

import numpy as np

# The repository root and common/ on the path, wherever this script lives.
_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_ROOT, "common")):
    _ROOT = os.path.dirname(_ROOT)
sys.path[1:1] = [_ROOT, os.path.join(_ROOT, "common")]

from geometry import mesh as gmesh
from geometry.geodesic import edge_flip_paths


def route(GX, GY, Z, waypoints, reading_pos, alpha=6.0, snap_tol=1.0):
    """Geodesic legs between consecutive waypoints.

    `alpha` is the height-to-plane exchange rate that defines "shortest".
    Returns (runs, solved, total). Each run is (points_xyz, reading_pos_along_it);
    a run breaks where a leg could not be solved.
    """
    cell = abs(GX[0, 1] - GX[0, 0])
    verts, faces, _ = gmesh.height_mesh(GX, GY, Z, np.isfinite(Z), alpha)
    verts, faces, _ = gmesh.largest_component(verts, faces)
    graph = gmesh.edge_graph(verts, faces)

    feet, dist = gmesh.snap(waypoints, verts)
    ok = dist <= snap_tol * cell
    pairs = [i for i in range(len(feet) - 1) if ok[i] and ok[i + 1]]
    solved = dict(zip(pairs, edge_flip_paths(
        verts, faces, [(int(feet[i]), int(feet[i + 1])) for i in pairs],
        graph=graph)))

    runs, cur_p, cur_t = [], [], []

    def flush():
        if cur_p:
            P, t = np.vstack(cur_p), np.concatenate(cur_t)
            if len(P) > 1:
                runs.append((P, t))

    for i in range(len(feet) - 1):
        leg = solved.get(i)
        if leg is None:
            flush()
            cur_p.clear(); cur_t.clear()
            continue
        P = np.column_stack([leg[:, 0], leg[:, 1], leg[:, 2] / alpha])
        seg = np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        frac = s / s[-1] if s[-1] > 0 else np.zeros(len(P))
        t = reading_pos[i] + frac * (reading_pos[i + 1] - reading_pos[i])
        if cur_p:
            P, t = P[1:], t[1:]
        cur_p.append(P); cur_t.append(t)
    flush()

    return runs, sum(v is not None for v in solved.values()), len(feet) - 1
