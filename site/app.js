/* Renders the pipeline reference from window.DATA (site/build.py) and
 * window.REF (site/export_reference.py).
 *
 * Nothing here restates a number: defaults come from the AST walk, chosen
 * parameters from the metrics files, figures from the filesystem. The only
 * authored text is the provenance table and the conflict list. */

const D = window.DATA, R = window.REF;
const REPO = "../";
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

/* argparse defaults arrive as real JSON values; render them the way you would
 * type them on a command line. */
function val(v) {
  if (v === null || v === undefined) return "—";
  if (v === true) return "true";
  if (v === false) return "false";
  if (Array.isArray(v)) return v.join(" ");
  return String(v);
}

const groups = D.groups.filter((g) => g.book === R.book && g.model === R.model);
const group = (fig) => groups.find((g) => g.figure === fig);

function table(headers, rows) {
  return `<table><thead><tr>${headers.map((h) =>
    `<th>${esc(h)}</th>`).join("")}</tr></thead><tbody>${rows.map((r) =>
    `<tr>${r.join("")}</tr>`).join("")}</tbody></table>`;
}

/* ---------------- tabs ---------------- */

const VIEWS = [
  ["flow", "Pipeline"], ["algos", "Algorithms"], ["params", "Parameters"],
  ["tuning", "Tuning"], ["figs", "Figures"], ["flags", "Conflicts"],
];

function tabs() {
  $("tabs").innerHTML = VIEWS.map(([k, t], i) =>
    `<button data-v="${k}" class="${i ? "" : "on"}">${esc(t)}</button>`).join("");
  $("tabs").onclick = (e) => {
    const b = e.target.closest("button[data-v]");
    if (!b) return;
    show(b.dataset.v);
  };
  show(location.hash.slice(1) || "flow");
}

function show(key) {
  if (!VIEWS.some(([k]) => k === key)) key = "flow";
  VIEWS.forEach(([k]) => $("v-" + k).classList.toggle("on", k === key));
  [...$("tabs").children].forEach((b) => b.classList.toggle("on", b.dataset.v === key));
  history.replaceState(null, "", "#" + key);
}

/* ---------------- pipeline ---------------- */

function renderFlow() {
  $("flow").innerHTML = R.flow.map((s, i) => `
    <button class="stage${i === 0 ? " on" : ""}" data-k="${esc(s.key)}">
      <span class="n">${String(i + 1).padStart(2, "0")}</span>
      <span class="t">${esc(s.title)}</span>
      <span class="c">${s.scripts.length} script${s.scripts.length === 1 ? "" : "s"}</span>
    </button>
    ${i < R.flow.length - 1 ? '<span class="arrowcol">→</span>' : ""}`).join("");

  $("flow").onclick = (e) => {
    const b = e.target.closest(".stage");
    if (!b) return;
    [...$("flow").querySelectorAll(".stage")].forEach((s) =>
      s.classList.toggle("on", s === b));
    stageDetail(b.dataset.k);
  };
  stageDetail(R.flow[0].key);
}

function stageDetail(key) {
  const stage = R.flow.find((s) => s.key === key);
  const blocks = stage.scripts.map((path) => {
    const s = R.scripts[path];
    if (!s) return `<div class="scriptblk"><div class="scriptname">${esc(path)}</div>
      <div class="purpose">not found in the source tree</div></div>`;

    const ins = (s.reads || []).map((w) =>
      `<span class="chip">← ${esc(R.paths[w] || w)}/</span>`).join("");
    const outs = s.writes.map((w) =>
      `<span class="chip out">→ ${esc(R.paths[w] || w)}/</span>`).join("");
    const deps = s.imports.filter((m) => m !== "paths")
      .map((m) => `<span class="chip dep">${esc(m)}</span>`).join("");
    const size = `<span class="chip">${s.lines} lines</span>`;

    /* --book / --model are the same on every script and add noise -- unless
     * they are the whole interface, in which case hiding them shows nothing. */
    const trimmed = s.args.filter((a) => !["--book", "--model"].includes(a.flag));
    const args = trimmed.length ? trimmed : s.args;

    const row = (a) => [
      `<td class="k">${esc(a.flag)}${a.positional
        ? ' <span style="color:var(--faint)">(positional)</span>' : ""}</td>`,
      `<td class="v">${a.required && a.default === null
        ? '<span style="color:var(--warn)">required</span>' : esc(val(a.default))}</td>`,
      `<td class="h">${a.choices ? `<code>${esc(a.choices.join(" | "))}</code> ` : ""}${
        esc(a.help || "")}</td>`,
    ];

    /* The shared protocol flags are identical across the four smoother wrappers;
     * repeating all seven in each block buries the one flag that differs. */
    const own = args.filter((a) => !a.from);
    const inherited = args.filter((a) => a.from);
    const donor = inherited.length ? inherited[0].from.split("/").pop() : "";

    const ladder = Object.values(s.grids || {}).map((g) =>
      Object.entries(g).map(([p, vs]) =>
        `<span class="chip">${esc(p)}: ${esc(vs.join(" "))}</span>`).join("")).join("");

    return `<div class="scriptblk">
      <div class="scriptname">${esc(path)}</div>
      <div class="purpose">${s.purpose ? esc(s.purpose)
        : '<span style="color:var(--warn)">no module docstring</span>'}</div>
      <div class="chips">${size}${ins}${outs}${deps}</div>
      ${ladder ? `<div class="chips" style="margin-bottom:10px">
        <span class="chip" style="border:none;padding-left:0">searched:</span>${ladder}</div>` : ""}
      ${own.length ? `<div class="scroller">${
        table(["Argument", "Default", "Notes"], own.map(row))}</div>` : ""}
      ${inherited.length ? `<details style="margin-top:8px">
        <summary style="cursor:pointer;color:var(--faint);font-size:12px">
          + ${inherited.length} shared flags from ${esc(donor)}</summary>
        <div class="scroller">${
          table(["Argument", "Default", "Notes"], inherited.map(row))}</div>
      </details>` : ""}
      ${!own.length && !inherited.length
        ? `<div class="purpose">no command-line arguments</div>` : ""}
    </div>`;
  }).join("");

  $("stage-detail").innerHTML =
    `<div class="io">${esc(stage.io)}</div>${blocks}`;
}

/* ---------------- algorithms ---------------- */

function renderAlgos() {
  $("algos").innerHTML = R.provenance.map((a) => `
    <div class="algo">
      <h3>${esc(a.name)}</h3>
      <dl>
        <dt>What</dt><dd>${esc(a.what)}</dd>
        <dt>Implementation</dt><dd class="impl">${esc(a.impl)}</dd>
        <dt>Origin</dt><dd>${esc(a.origin)}</dd>
        ${a.links.length ? `<dt>Links</dt><dd>${a.links.map(([t, u]) =>
          `<a href="${esc(u)}" target="_blank" rel="noopener">${esc(t)}</a>`
        ).join("<br>")}</dd>` : ""}
        ${a.note ? `<dt>Note</dt><dd class="warn">${esc(a.note)}</dd>` : ""}
      </dl>
    </div>`).join("");
}

/* ---------------- parameters ---------------- */

function renderParams() {
  const rows = [];
  Object.entries(R.scripts).forEach(([path, s]) =>
    s.args.forEach((a) => rows.push({ path, ...a })));

  const scripts = [...new Set(rows.map((r) => r.path))].sort();
  $("p-script").innerHTML = `<option value="">all scripts</option>` +
    scripts.map((p) => `<option value="${esc(p)}">${esc(p)}</option>`).join("");

  function draw() {
    const sc = $("p-script").value;
    const q = $("p-text").value.trim().toLowerCase();
    const hit = rows.filter((r) => (!sc || r.path === sc) &&
      (!q || r.flag.toLowerCase().includes(q) ||
        (r.help || "").toLowerCase().includes(q)));

    $("p-table").innerHTML = table(["Argument", "Default", "Script", "Notes"],
      hit.map((r) => [
        `<td class="k">${esc(r.flag)}</td>`,
        `<td class="v">${r.required && r.default === null
          ? '<span style="color:var(--warn)">required</span>' : esc(val(r.default))}</td>`,
        `<td class="k" style="color:var(--faint)">${esc(r.path.split("/").pop())}</td>`,
        `<td class="h">${r.choices ? `<code>${esc(r.choices.join(" | "))}</code> ` : ""}${
          esc(r.help || "")}</td>`,
      ]));
    $("p-count").textContent = `${hit.length} / ${rows.length} flags`;
  }
  $("p-script").onchange = draw;
  $("p-text").oninput = draw;
  draw();
}

/* ---------------- tuning ---------------- */

function renderTuning() {
  /* The four search ladders, as declared in the smooth_*.py modules. */
  const ladders = [];
  Object.entries(R.scripts).forEach(([path, s]) => {
    Object.entries(s.grids || {}).forEach(([name, grid]) => {
      ladders.push({ method: path.split("/").pop().replace(/^smooth_|\.py$/g, ""),
                     grid });
    });
  });

  $("grids").innerHTML = ladders.length ? table(
    ["Method", "Parameter", "Values searched"],
    ladders.flatMap((l) => Object.entries(l.grid).map(([p, vs], i) => [
      `<td class="k">${i === 0 ? esc(l.method) : ""}</td>`,
      `<td class="k">${esc(p)}</td>`,
      `<td class="v">${esc(vs.join("  "))}</td>`,
    ]))) : "";

  const emotions = [...new Set(Object.values(R.chosen)
    .flatMap((m) => Object.keys(m)))].sort();
  $("t-emotion").innerHTML = emotions.map((e) =>
    `<option value="${esc(e)}">${esc(e)}</option>`).join("");

  function draw() {
    const e = $("t-emotion").value;
    const methods = Object.keys(R.chosen).sort();
    const perf = D.smoothing[e] || {};
    const best = Math.min(...Object.values(perf).map((v) => v.val_rmse));

    $("tuning").innerHTML = table(
      ["Smoother", "Chosen", "Held-out RMSE", "Baseline", "vs baseline"],
      methods.map((m) => {
        const p = R.chosen[m][e], v = perf[m];
        return [
          `<td class="k">${esc(m)}</td>`,
          `<td class="v">${esc(Object.entries(p || {})
            .map(([k, x]) => `${k}=${x}`).join(" "))}</td>`,
          `<td class="v">${v ? (v.val_rmse === best ? "<b>" : "") +
            v.val_rmse.toFixed(4) + (v.val_rmse === best ? "</b>" : "") : "—"}</td>`,
          `<td class="v" style="color:var(--faint)">${
            v ? v.baseline_rmse.toFixed(4) : "—"}</td>`,
          `<td class="v" style="color:var(--good)">${
            v ? "+" + v.improvement.toFixed(1) + "%" : "—"}</td>`,
        ];
      }));

    const vs = Object.values(perf).map((x) => x.val_rmse);
    $("t-count").textContent = vs.length
      ? `best-to-worst spread ${((Math.max(...vs) - Math.min(...vs)) /
          Math.min(...vs) * 100).toFixed(2)}%` : "";
  }
  $("t-emotion").onchange = draw;
  draw();
}

/* ---------------- figures ---------------- */

function card(f, label) {
  const src = REPO + f.path;
  if (f.kind === "interactive") {
    return `<figure class="play"><a href="${esc(src)}" target="_blank" rel="noopener">
      ▶ ${esc(f.name)}</a></figure>`;
  }
  return `<figure><img loading="lazy" src="${esc(src)}" alt="${esc(f.name)}"
    data-full="${esc(src)}" data-cap="${esc(f.path)}"><figcaption>${
    label ? `<span class="fam">${esc(label)}</span>` : ""}${
    f.variant ? `<span class="fam">${esc(f.variant)}</span>` : ""}${
    esc(f.name)}</figcaption></figure>`;
}

function renderFigures() {
  const all = groups.flatMap((g) => g.files.filter((f) => f.kind !== "data")
    .map((f) => ({ ...f, figure: g.figure })));

  const figs = [...new Set(all.map((f) => f.figure))].sort();
  $("f-figure").innerHTML = `<option value="">all families</option>` +
    figs.map((v) => `<option value="${esc(v)}">${esc(v)}</option>`).join("");

  function draw() {
    const fg = $("f-figure").value, q = $("f-text").value.trim().toLowerCase();
    const hit = all.filter((f) => (!fg || f.figure === fg) &&
      (!q || f.name.toLowerCase().includes(q)));
    const CAP = 48;
    $("f-grid").innerHTML = hit.slice(0, CAP)
      .map((f) => card(f, f.figure)).join("");
    $("f-count").textContent = hit.length > CAP
      ? `${CAP} of ${hit.length} — narrow the filter` : `${hit.length} figures`;
  }
  $("f-figure").onchange = draw;
  $("f-text").oninput = draw;
  draw();
}

/* ---------------- conflicts ---------------- */

const CONFLICTS = [
  ["Two bandwidth conventions, now told apart by folder",
   `The duplicate Nadaraya–Watson is <b>fixed</b> — one implementation, in
    <code>geometry/smoothers.py</code> — and the two tunings are now siblings
    under <code>surface/</code> rather than separate top-level folders. What
    remains is a real difference to be aware of: <code>isotropic_loo/</code>
    selects one <code>h</code> in <em>raw</em> PCA units by leave-one-out, while
    <code>gaussian_nw/</code> selects <code>hx, hy</code> in <em>standardized</em>
    units by k-fold. On wonder that is h=0.062 raw = 0.31/0.43 sd against
    0.40/0.40 — mean difference across the surface ~0.005, correlation 0.999.
    Both still write <code>field_&lt;emotion&gt;.npz</code>; the parent folder now
    says which is which.`],
  ["The two bandwidth ladders are not merged yet",
   `<code>bandwidth_grid.py</code> sweeps ×0.25–1 and <code>smooth_grid.py</code>
    sweeps ×0.33–3 — the same rows×columns figure over different halves of one
    ladder. They now sit in their own variant directories under
    <code>surface/bandwidth_sweep/</code> instead of colliding, but running
    <code>bandwidth_grid.py --factors 0.25 0.5 1 3</code> would replace both with
    a single figure spanning under- to over-smoothed.`],
  ["Four different terrains are all called “the emotion surface”",
   `geodesics use <code>gaussian_nw</code> h=0.2; arc-on-surface and mood use
    0.15; <code>narrative_arc_3d.py</code> defaults to <code>local_linear</code>
    (which can leave [0,1]); <code>emotion_surface.py</code> picks its own h per
    emotion.`],
  ["Two incompatible mood-axis definitions",
   `<code>narrative_arc_3d.py --z-mode spectrum</code> orders
    <em>danger, sadness, …</em> with a raw score-weighted centre of mass;
    <code>arc_emotion_axis.py</code> orders <em>sadness, danger, …</em> with a
    softmax over rank-normalized prominence.`],
  ["Window parameters differ per book",
   `Alice's arcs use window 10 / stride 5. Script defaults disagree:
    <code>windows.py</code> and <code>narrative-arc/</code> 40/20,
    <code>narrative_arc_3d.py</code> 25/12, <code>arc_comparison.py</code> 10/5.`],
  ["The validation contradicts most of the arc output",
   `<code>arc_comparison</code> finds t-SNE/UMAP arcs are almost entirely
    algorithm, and that windows 40–80 keep under 10% of the variance — yet most
    arc figures are exactly those settings.`],
  ["LOESS <code>frac=0.8</code> is a ladder artifact reported as a tuned result",
   `The search stops at 0.8, so it selected the edge. 80% of the book is not a
    local neighbourhood; <code>frac=0.2</code> scores the same.`],
  ["No tests",
   `<code>geodesic_arrows.py</code> cites a <code>test_geodesic</code> flat-mesh
    control for the exact solver. It was never written.`],
];

function renderFlags() {
  const stale = groups.filter((g) => !g.params.length && g.n_images);
  const n = stale.reduce((s, g) => s + g.n_images, 0);
  const dyn = n ? [[`${n} figures carry no provenance`,
    `Written before <code>visualization/paths.py</code>: no
     <code>params.json</code>, no variant directory, so their settings are not
     recoverable. Re-run the family to replace them.`]] : [];

  $("flags").innerHTML = [...CONFLICTS, ...dyn].map(([t, b]) =>
    `<div class="flag"><b>${t}</b><p>${b}</p></div>`).join("");
}

/* ---------------- chrome ---------------- */

function lightbox() {
  document.body.addEventListener("click", (e) => {
    const img = e.target.closest("img[data-full]");
    if (!img) return;
    $("lb-img").src = img.dataset.full;
    $("lb-cap").textContent = img.dataset.cap;
    $("lightbox").showModal();
  });
}

const corpus = D.corpus.find((c) => c.book === R.book) || {};
$("scope").textContent =
  `${R.book} × ${R.model} · ${corpus.paragraphs} paragraphs · ` +
  `${corpus.chapters} chapters · ${(corpus.models.find((m) =>
    m.name === R.model) || {}).dim}-d · ${corpus.emotions.length} emotions`;

tabs();
renderFlow();
renderAlgos();
renderParams();
renderTuning();
renderFigures();
renderFlags();
lightbox();

$("foot").textContent =
  `${D.totals.images} figures · ${D.totals.interactive} interactive · ` +
  `${Object.keys(R.scripts).length} scripts · ` +
  `rebuild: python site/build.py && python site/export_reference.py`;
