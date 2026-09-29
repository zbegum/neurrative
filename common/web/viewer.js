// Shared building blocks for the viewer pages (story map, surface lab, arc 3-D).
// Inlined into each page at build time by common/viewer.py, ahead of the page's
// own code, inside one <script type="module">. d3 is loaded as a global before.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { CSS2DRenderer, CSS2DObject } from "three/addons/renderers/CSS2DRenderer.js";

const $ = (id) => document.getElementById(id);
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const ramp = () => css("--ramp").split(",").map((s) => s.trim());
const rampAt = (t) => d3.piecewise(d3.interpolateRgb, ramp())(Math.max(0, Math.min(1, t)));

// ---- reading: text panel, timeline, keys --------------------------------------
function setText(book, i, chapterEl, textEl) {
  chapterEl.textContent = book.chapters[book.chapter[i]];
  textEl.textContent = book.text[i];
}

// A timeline of one series over reading order; returns { draw(series, chapter), set(i) }.
function Timeline(el, onSeek) {
  let tx = null;
  const svg = d3.select(el);
  // stops: optional [[value, colour], ...] -- the curve is coloured by its value
  function draw(series, chapter, stops = null) {
    const W = el.clientWidth, H = el.clientHeight, m = { l: 16, r: 16, t: 14, b: 18 };
    const N = series.length;
    tx = d3.scaleLinear([0, N - 1], [m.l, W - m.r]);
    let [lo, hi] = d3.extent(series);
    const flat = !(hi > lo);         // no scores: just the chapter ticks and the cursor
    if (flat) hi = lo + 1;
    const ty = d3.scaleLinear([lo - (hi - lo) * 0.08, hi], [H - m.b, m.t]);
    svg.selectAll("*").remove();
    const starts = chapter.map((c, j) => (j === 0 || c !== chapter[j - 1]) ? j : null)
      .filter((j) => j !== null);
    svg.append("g").attr("stroke", css("--arc")).selectAll("line").data(starts).join("line")
      .attr("x1", tx).attr("x2", tx).attr("y1", H - m.b).attr("y2", H - m.b + 6);
    if (!flat) {
      let paint = css("--line"), fillOpacity = 0.12;
      if (stops) {
        const id = "tl-grad-" + Math.random().toString(36).slice(2, 8);
        const g = svg.append("defs").append("linearGradient").attr("id", id)
          .attr("gradientUnits", "userSpaceOnUse").attr("x1", 0).attr("x2", 0)
          .attr("y1", ty(stops[0][0])).attr("y2", ty(stops[stops.length - 1][0]));
        const span = stops[stops.length - 1][0] - stops[0][0] || 1;
        stops.forEach(([v, c]) => g.append("stop").attr("offset", (v - stops[0][0]) / span).attr("stop-color", c));
        paint = `url(#${id})`; fillOpacity = 0.35;
      }
      svg.append("path").datum(series).attr("fill", paint).attr("fill-opacity", fillOpacity)
        .attr("d", d3.area().x((v, j) => tx(j)).y0(H - m.b).y1(ty).curve(d3.curveMonotoneX));
      svg.append("path").datum(series).attr("fill", "none").attr("stroke", paint)
        .attr("stroke-width", 2).attr("d", d3.line().x((v, j) => tx(j)).y(ty).curve(d3.curveMonotoneX));
    }
    svg.append("line").attr("class", "cursor").attr("y1", m.t - 6).attr("y2", H - m.b)
      .attr("stroke", css("--ink")).attr("stroke-width", 1.5);
  }
  const seek = (ev) => onSeek(Math.round(tx.invert(d3.pointer(ev, el)[0])));
  svg.on("pointerdown", (ev) => { el.setPointerCapture(ev.pointerId); seek(ev); })
    .on("pointermove", (ev) => { if (ev.buttons) seek(ev); });
  return {
    draw,
    set(i) { svg.select(".cursor").attr("x1", tx(i)).attr("x2", tx(i)); },
  };
}

// The page's state as a URL fragment ("book/.../paragraph"): kept in the address,
// and sent to the tabbed page when this view is one of its tabs.
function publish(state) {
  try { history.replaceState(null, "", "#" + state); } catch (e) { /* e.g. about:srcdoc */ }
  if (window.parent !== window) window.parent.postMessage({ viewerState: state }, "*");
}
// Open from the address now, and again whenever the tabbed page sends a state.
function follow(apply) {
  apply(decodeURIComponent(location.hash.slice(1)));
  window.addEventListener("hashchange", () => apply(decodeURIComponent(location.hash.slice(1))));
  window.addEventListener("message", (e) => { if (typeof e.data?.viewerState === "string") apply(e.data.viewerState); });
}

// Arrows step (shift: 20), space plays.
function readingKeys(getI, getN, go) {
  let playing = null;
  document.addEventListener("keydown", (ev) => {
    if (ev.target.tagName === "SELECT") return;
    if (ev.key === "ArrowRight" || ev.key === "ArrowLeft") {
      go(getI() + (ev.key === "ArrowRight" ? 1 : -1) * (ev.shiftKey ? 20 : 1));
      ev.preventDefault();
    } else if (ev.key === " ") {
      ev.preventDefault();
      if (playing) { clearInterval(playing); playing = null; return; }
      playing = setInterval(() => {
        if (getI() >= getN() - 1) { clearInterval(playing); playing = null; } else go(getI() + 1);
      }, 60);
    }
  });
}

// Where paragraph i sits along a path of windows whose last field is the window's
// centre paragraph: { k, t } = the segment [k, k + 1] and the fraction along it.
function pathAt(path, i) {
  const c = (w) => w[w.length - 1];
  if (i <= c(path[0])) return { k: 0, t: 0 };
  for (let k = 1; k < path.length; k++) {
    if (i <= c(path[k])) return { k: k - 1, t: (i - c(path[k - 1])) / (c(path[k]) - c(path[k - 1])) };
  }
  return { k: path.length - 2, t: 1 };
}

// A window path [[x, y, centre], ...] sampled densely along a Catmull-Rom curve;
// each sample keeps its reading position (the interpolated centre).
function denseArc(P, per = 8) {
  const out = [];
  for (let k = 0; k < P.length - 1; k++) {
    const p0 = P[Math.max(0, k - 1)], p1 = P[k], p2 = P[k + 1], p3 = P[Math.min(P.length - 1, k + 2)];
    for (let s = 0; s < per; s++) {
      const t = s / per, t2 = t * t, t3 = t2 * t;
      const cr = (a, b, c, d) => 0.5 * (2 * b + (c - a) * t + (2 * a - 5 * b + 4 * c - d) * t2 + (3 * b - a - 3 * c + d) * t3);
      out.push([cr(p0[0], p1[0], p2[0], p3[0]), cr(p0[1], p1[1], p2[1], p3[1]), p1[2] + t * (p2[2] - p1[2])]);
    }
  }
  out.push(P[P.length - 1]);
  return out;
}

// ---- 3-D -------------------------------------------------------------------------
// Map the PCA extent to a unit square around the origin: x -> world x, y -> world -z.
function Frame(extent) {
  const [x0, x1, y0, y1] = extent, s = Math.max(x1 - x0, y1 - y0) / 2;
  const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
  return { X: (x) => (x - cx) / s, Z: (y) => -(y - cy) / s };
}

// Bilinear sample of an n x n row-major field (rows = y) at PCA (x, y); NaN outside.
function sample(field, n, extent, x, y) {
  const [x0, x1, y0, y1] = extent;
  const u = (x - x0) / (x1 - x0) * (n - 1), v = (y - y0) / (y1 - y0) * (n - 1);
  const i = Math.floor(u), j = Math.floor(v);
  if (i < 0 || j < 0 || i >= n - 1 || j >= n - 1) return NaN;
  const a = u - i, b = v - j, f = (ii, jj) => field[jj * n + ii] ?? NaN;
  return (1 - a) * (1 - b) * f(i, j) + a * (1 - b) * f(i + 1, j) + (1 - a) * b * f(i, j + 1) + a * b * f(i + 1, j + 1);
}

// A renderer + camera + orbit controls that fills `el`, rendering on demand.
function Scene(el) {
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  el.appendChild(renderer.domElement);
  renderer.domElement.className = "scene";
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 100);
  camera.position.set(2.1, 2.0, 2.8);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0.12, 0);
  controls.enableDamping = true;
  scene.add(new THREE.HemisphereLight(0xffffff, 0x777777, 2.2));
  const sun = new THREE.DirectionalLight(0xffffff, 1.2); sun.position.set(1, 3, 2); scene.add(sun);
  const group = new THREE.Group(); scene.add(group);
  // HTML labels placed in 3-D (CSS2DObject), drawn over the canvas
  const labels = new CSS2DRenderer();
  labels.domElement.style.position = "absolute"; labels.domElement.style.inset = "0";
  labels.domElement.style.pointerEvents = "none";
  el.appendChild(labels.domElement);
  let raf = null;
  const render = () => {
    raf = null;
    if (controls.update()) request();
    renderer.render(scene, camera);
    labels.render(scene, camera);
  };
  const request = () => { if (!raf) raf = requestAnimationFrame(render); };
  controls.addEventListener("change", request);
  new ResizeObserver(() => {
    const W = el.clientWidth, H = el.clientHeight;
    if (!W || !H) return;
    renderer.setSize(W, H, false); labels.setSize(W, H);
    camera.aspect = W / H; camera.updateProjectionMatrix(); request();
  }).observe(el);
  return {
    THREE, renderer, scene, camera, controls, group, request,
    clear() {
      while (group.children.length) {
        const o = group.children.pop();
        o.traverse((c) => { c.geometry?.dispose(); if (c.isCSS2DObject) c.element.remove(); });
      }
    },
  };
}

// Height of a value: its place between lo and hi, times the relief (world units).
const RELIEF = 0.45;
const lift = (v, lo, hi, relief = RELIEF) => ((v - lo) / (hi - lo || 1)) * relief;

// Terrain mesh of an n x n field over `extent`, coloured on the ramp between lo and hi
// and raised by `height(v)`; triangles are kept only where `keep(cellIndex)` holds at
// all three corners.
function terrain(field, n, extent, { height = (v) => v, lo = 0, hi = 1, keep = () => true, opacity = 1,
                                     color = null } = {}) {
  const F = Frame(extent), [x0, x1, y0, y1] = extent;
  const pos = new Float32Array(n * n * 3), col = new Float32Array(n * n * 3);
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
    const k = j * n + i, v = field[k];
    const x = x0 + (x1 - x0) * i / (n - 1), y = y0 + (y1 - y0) * j / (n - 1);
    pos.set([F.X(x), height(v ?? lo), F.Z(y)], 3 * k);
    const c = new THREE.Color(color ? color(v ?? lo) : rampAt(((v ?? lo) - lo) / (hi - lo || 1)));
    col.set([c.r, c.g, c.b], 3 * k);
  }
  const idx = [];
  const ok = (k) => field[k] !== null && Number.isFinite(field[k]) && keep(k);
  for (let j = 0; j < n - 1; j++) for (let i = 0; i < n - 1; i++) {
    const a = j * n + i, b = a + 1, c = a + n, d = c + 1;
    if (ok(a) && ok(b) && ok(c)) idx.push(a, c, b);
    if (ok(b) && ok(c) && ok(d)) idx.push(b, c, d);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  g.setAttribute("color", new THREE.BufferAttribute(col, 3));
  g.setIndex(idx); g.computeVertexNormals();
  return new THREE.Mesh(g, new THREE.MeshStandardMaterial({
    vertexColors: true, side: THREE.DoubleSide, roughness: 0.9, metalness: 0,
    transparent: opacity < 1, opacity,
  }));
}

// A smooth thin tube through world-space points.
function curve(points, radius, color) {
  const path = new THREE.CatmullRomCurve3(points.map((p) => new THREE.Vector3(...p)));
  const g = new THREE.TubeGeometry(path, Math.max(64, points.length * 6), radius, 8, false);
  return new THREE.Mesh(g, new THREE.MeshStandardMaterial({ color, roughness: 0.6 }));
}

// A text label at a 3-D point (muted, small).
// `align: "right"` puts the text to the left of the point, ending just before it.
function label(text, p, { align = "right", dot = null } = {}) {
  const div = document.createElement("div");      // positioned by CSS2DRenderer
  const span = document.createElement("span");    // offset inside it
  span.textContent = text;
  if (dot) span.innerHTML += ` <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${dot};margin-left:4px;vertical-align:1px"></span>`;
  span.style.cssText = `display:inline-block;color:${css("--ink-2")};font-size:12px;white-space:nowrap;` +
    (align === "right" ? "transform:translateX(calc(-50% - 10px));" : "transform:translateX(calc(50% + 10px));");
  div.appendChild(span);
  const o = new CSS2DObject(div); o.position.set(...p);
  return o;
}

function ball(p, radius, color) {
  const m = new THREE.Mesh(new THREE.SphereGeometry(radius, 20, 14),
    new THREE.MeshStandardMaterial({ color, roughness: 0.4 }));
  m.position.set(...p);
  return m;
}
