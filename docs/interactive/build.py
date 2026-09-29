"""
Build docs/interactive/index.html: the three interactive views in one page, one
tab each -- the story map (arc_on_surface/), the mood surface (surface/mood/) and
the arc in 3-D (arc/curve/).

Each view is rebuilt by its own script, then embedded whole; a tab starts its
view the first time it is opened. The book and the reading position follow you
from tab to tab; everything else a tab keeps for itself.

Example (from anywhere):

python docs/interactive/build.py
"""

import html
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
# (tab id, name on the tab, builder, the page it writes, where book / paragraph sit
#  in its state "a/b/c", the state to open with before it has one of its own)
TABS = [
  ("story", "story map", "arc_on_surface/story_map.py", "arc_on_surface/output/story_map.html",
   (0, 2), "{book}/wonder/{i}"),
  ("mood", "mood surface", "surface/mood/mood_viewer.py", "surface/mood/output/mood_viewer.html",
   (0, 1), "{book}/{i}"),
  ("arc", "arc 3-D", "arc/curve/arc_viewer.py", "arc/output/arc_viewer.html",
   (0, 3), "{book}/pca/wonder/{i}"),
]

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>neurrative</title>
<style>
:root { --surface: #fcfcfb; --ink: #0b0b0b; --ink-3: #8a8983; }
@media (prefers-color-scheme: dark) { :root { --surface: #1a1a19; --ink: #ffffff; --ink-3: #85847d; } }
html, body { margin: 0; height: 100%; background: var(--surface); }
body { display: grid; grid-template-rows: auto 1fr; font: 14px/1.4 system-ui, -apple-system, sans-serif; }
nav { display: flex; gap: 20px; padding: 10px 16px 0; }
nav button { font: inherit; background: none; border: 0; padding: 2px 0; cursor: pointer;
             color: var(--ink-3); border-bottom: 1.5px solid transparent; }
nav button[aria-selected="true"] { color: var(--ink); border-bottom-color: var(--ink); }
iframe { width: 100%; height: 100%; border: 0; display: block; }
iframe[hidden] { display: none; }
textarea { display: none; }
</style>
</head>
<body>
<nav role="tablist">__BUTTONS__</nav>
<main style="min-height:0">__FRAMES__</main>
__SOURCES__
<script>
const TABS = __TABS__;
const shared = { book: null, i: null };
const last = {};             // each tab's own latest state
const frames = {};
const stateFor = (id) => {
  const t = TABS[id], base = (last[id] || t.start.replace("{book}", "").replace("{i}", "")).split("/");
  if (shared.book) base[t.at[0]] = shared.book;
  if (shared.i !== null) base[t.at[1]] = String(shared.i);
  return base.join("/");
};
function show(id) {
  for (const k in TABS) {
    document.getElementById("tab-" + k).setAttribute("aria-selected", k === id);
    if (frames[k]) frames[k].hidden = k !== id;
  }
  let f = frames[id];
  const state = stateFor(id);
  if (!f) {
    f = frames[id] = document.createElement("iframe");
    f.title = TABS[id].name;
    f.addEventListener("load", () => f.contentWindow.postMessage({ viewerState: state }, "*"), { once: true });
    f.srcdoc = document.getElementById("src-" + id).value;
    document.querySelector("main").appendChild(f);
  } else {
    f.contentWindow.postMessage({ viewerState: state }, "*");
  }
  try { history.replaceState(null, "", "#" + id); } catch (e) {}
}
window.addEventListener("message", (e) => {
  const id = Object.keys(frames).find((k) => frames[k].contentWindow === e.source);
  if (!id || typeof e.data?.viewerState !== "string") return;
  last[id] = e.data.viewerState;
  const parts = e.data.viewerState.split("/"), at = TABS[id].at;
  if (parts[at[0]]) shared.book = parts[at[0]];
  if (parts[at[1]] && !isNaN(+parts[at[1]])) shared.i = +parts[at[1]];
});
for (const id in TABS) document.getElementById("tab-" + id).onclick = () => show(id);
show(TABS[location.hash.slice(1)] ? location.hash.slice(1) : Object.keys(TABS)[0]);
</script>
</body>
</html>
"""


def main():
  import json
  buttons, sources, tabs = [], [], {}
  for tid, name, builder, page, at, start in TABS:
    subprocess.run([sys.executable, os.path.join(ROOT, builder)], check=True)
    body = open(os.path.join(ROOT, page)).read()
    buttons.append(f'<button role="tab" id="tab-{tid}">{html.escape(name)}</button>')
    # a <textarea> holds the page verbatim; only "</textarea" and "&" need escaping
    sources.append(f'<textarea id="src-{tid}">{html.escape(body, quote=False)}</textarea>')
    tabs[tid] = {"name": name, "at": list(at), "start": start}
  out = (PAGE.replace("__BUTTONS__", "".join(buttons)).replace("__FRAMES__", "")
             .replace("__SOURCES__", "\n".join(sources))
             .replace("__TABS__", json.dumps(tabs)))
  path = os.path.join(HERE, "index.html")
  with open(path, "w") as f:
    f.write(out)
  print(f"wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB)")


if __name__ == "__main__":
  main()
