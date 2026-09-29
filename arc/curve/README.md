# arc/curve — the arc as a curve

The windowed series projected with PCA, UMAP and t-SNE and joined in reading
order, in 2-D (`arc_2d.py`) or 3-D (`arc_3d.py`). `--fit` adds a uniform cubic
B-spline through the windows (`narrative_arc/curves.py`, vendored
bspline-regression); `--show-control` draws its control polygon.

    python curve/arc_2d.py --fit --show-control      # run from arc/
    python curve/arc_3d.py --fit
    python curve/arc_viewer.py     # the arc in 3-D with its fit, the tube and the text

Options and outputs: [../README.md](../README.md#usage).
