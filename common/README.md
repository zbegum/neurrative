# common

Shared plumbing, imported by the scripts in every step folder. Nothing here
draws a figure.

| module | what |
|---|---|
| `paths.py` | Where every output goes. `out_dir(book, model, figure)` returns `<step folder>/output/<book>/<model>/<figure>/`; `FAMILY_HOME` and `SURFACE_HOME` say which folder owns each figure family. `stamp()` writes the `params.json` that records how a result was made. |
| `data.py` | Loading paragraphs, chapters and emotion scores from `books/`, and the shared emotion conventions. |
| `windows.py` | The windowed series: the book as an ordered sequence of mean-pooled windows, saved to `arc/output/<book>/<model>/windows/`. The window arithmetic itself (`window_bounds`, `window_centers`, `pool`) is imported from `arc/narrative_arc/windows.py`, so there is one definition. Runnable: `python common/windows.py --book <book> --model <model>`. |
| `smooth_common.py` | The protocol every surface fit is held to: standardized coordinates, k-fold CV on a training split, one held-out score, and the shared figures. Used by `surface/` and `arc_on_surface/`. |
| `embedding_common.py` | What the projection scripts share: colors, drawing, and where their output goes. |

## How scripts find these modules

Each script starts with the same short block, which walks up from the script
to the folder that contains `common/` and puts the repository root and `common/`
on `sys.path`:

```python
_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_ROOT, "common")):
  _ROOT = os.path.dirname(_ROOT)
sys.path[1:1] = [_ROOT, os.path.join(_ROOT, "common")]
```

So `import paths` and `from geometry.smoothers import ...` work from any
working directory, and a script's own folder still comes first on the path.
