"""Trace a book's narrative arc through embedding space.

Slide a window over the paragraphs, mean-pool each window's embeddings, project
the pooled series to 2-D or 3-D (PCA, UMAP, t-SNE), and connect the points in
reading order. The line is the story's path through semantic space.

  data         loading books, embeddings and emotion scores
  windows      the windowed series itself -- bounds, pooling, save/load
  projections  PCA / UMAP / t-SNE in 2 or 3 components
  colors       what the points are colored by
  curves       uniform B-spline fit through the windows (bspline-regression)
  plot_2d      the flat arc (matplotlib)
  plot_3d      the arc in 3-D (matplotlib PNG)
"""
