"use strict";

const $ = (id) => document.getElementById(id);
const MAX = 1000;
const fmtPct = (v) => (v == null ? "-" : (v * 100).toFixed(1) + "%");
const fmtMs = (v) => (v == null ? "-" : v < 10 ? v.toFixed(2) + " ms" : v.toFixed(0) + " ms");

const els = {
  form: $("form"), text: $("text"), count: $("count"), err: $("text-err"),
  slider: $("threshold"), sliderVal: $("threshold-val"), sliderLabel: $("slider-label"), sliderHelp: $("slider-help"),
  submit: $("submit"), result: $("result"), loading: $("result-loading"), rErr: $("result-error"), body: $("result-body"),
};

let results = null;
let dataset = "sst2";
let learnedCutoff = null;
let cutoffTouched = false;

/* ---------- input ---------- */
function updateCount() {
  els.count.textContent = `${els.text.value.length} / ${MAX}`;
  if (els.text.value.trim()) setFieldError("");
}
function setFieldError(msg) {
  els.err.textContent = msg;
  els.err.hidden = !msg;
  els.text.setAttribute("aria-invalid", msg ? "true" : "false");
}
els.text.addEventListener("input", updateCount);
document.querySelectorAll(".chip").forEach((chip) =>
  chip.addEventListener("click", () => {
    els.text.value = chip.dataset.example;
    updateCount();
    els.text.focus();
  })
);

function router() {
  return document.querySelector('input[name="router"]:checked').value;
}
function configureSlider() {
  if (router() === "fixed") {
    Object.assign(els.slider, { min: 0.5, max: 1, step: 0.01 });
    els.slider.value = els.slider.dataset.fixed || 0.85;
    els.sliderLabel.textContent = "Confidence threshold";
    els.sliderHelp.textContent = "Escalate when the small model's calibrated confidence is below this.";
  } else {
    Object.assign(els.slider, { min: 0, max: 1, step: 0.01 });
    els.slider.value = els.slider.dataset.learned || (learnedCutoff ?? 0.5);
    els.sliderLabel.textContent = "Escalation cutoff";
    els.sliderHelp.textContent = learnedCutoff == null
      ? "Escalate when the router's predicted benefit of escalating is above this."
      : `Escalate when the router's predicted benefit is above this (tuned value ${learnedCutoff.toFixed(2)}).`;
  }
  els.sliderVal.textContent = Number(els.slider.value).toFixed(2);
}
document.querySelectorAll('input[name="router"]').forEach((r) => r.addEventListener("change", configureSlider));
els.slider.addEventListener("input", () => {
  els.slider.dataset[router()] = els.slider.value;
  if (router() === "learned") cutoffTouched = true;
  els.sliderVal.textContent = Number(els.slider.value).toFixed(2);
});

/* ---------- predict ---------- */
function show(which) {
  els.result.hidden = false;
  els.loading.hidden = which !== "loading";
  els.rErr.hidden = which !== "error";
  els.body.hidden = which !== "body";
  els.result.setAttribute("aria-busy", which === "loading" ? "true" : "false");
}

els.form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = els.text.value;
  if (!text.trim()) {
    setFieldError("Please enter some text to classify.");
    els.text.focus();
    return;
  }
  const payload = { text, router: router() };
  if (payload.router === "fixed") payload.threshold = Number(els.slider.value);
  else if (cutoffTouched) payload.cutoff = Number(els.slider.value);

  els.submit.disabled = true;
  show("loading");
  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      const d = data.detail;
      throw new Error(Array.isArray(d) ? d.map((x) => x.msg).join("; ") : d || `Request failed (${res.status})`);
    }
    renderResult(data);
    loadMetrics();
  } catch (err) {
    els.rErr.textContent = err.message || "Something went wrong.";
    show("error");
  } finally {
    els.submit.disabled = false;
  }
});

function renderResult(r) {
  const label = $("r-label");
  label.textContent = r.label;
  label.className = `big-label ${r.label}`;
  $("r-conf").textContent = `${fmtPct(r.confidence)} confident · answered by the ${r.model} model`;
  const pct = Math.round(r.confidence * 100);
  const bar = $("r-bar");
  bar.setAttribute("aria-valuenow", pct);
  bar.setAttribute("aria-valuetext", `${pct}% confidence`);
  const fill = $("r-bar-fill");
  fill.style.width = "0";
  requestAnimationFrame(() => requestAnimationFrame(() => (fill.style.width = pct + "%")));

  const small = $("step-small"), large = $("step-large");
  small.className = "step visited" + (r.model === "small" ? " answered" : "");
  large.className = "step" + (r.model === "large" ? " visited answered" : "");
  small.setAttribute("aria-current", r.model === "small" ? "step" : "false");
  large.setAttribute("aria-current", r.model === "large" ? "step" : "false");

  $("r-small").textContent = `${r.small_label} (${fmtPct(r.small_confidence)})`;
  $("r-esc").textContent = r.escalated ? "Yes, to the large model" : "No";
  $("r-router").textContent = `${r.router === "fixed" ? "Fixed" : "Learned"} @ ${r.threshold.toFixed(2)}`;
  $("r-lat").textContent = fmtMs(r.latency_ms);
  show("body");
}

/* ---------- metrics ---------- */
async function loadMetrics() {
  try {
    const m = await (await fetch("/metrics")).json();
    $("s-req").textContent = m.requests.toLocaleString();
    $("s-esc").textContent = fmtPct(m.escalation_rate);
    $("s-p50").textContent = fmtMs(m.p50_ms);
    $("s-p95").textContent = fmtMs(m.p95_ms);
  } catch (_) {
    /* leave previous values */
  }
}

/* ---------- performance ---------- */
const SVGNS = "http://www.w3.org/2000/svg";
function svg(tag, attrs, parent) {
  const el = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  if (parent) parent.appendChild(el);
  return el;
}
function niceTicks(lo, hi, n) {
  const span = hi - lo || 1;
  const step0 = span / n;
  const mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= step0);
  const ticks = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) ticks.push(+v.toFixed(10));
  return ticks;
}

function drawChart(d) {
  const host = $("chart");
  host.innerHTML = "";
  const tip = $("tip");
  const W = Math.max(300, Math.min(640, Math.round(host.clientWidth || 640)));
  const H = Math.round(W < 480 ? W * 0.75 : W * 0.5), m = { t: 12, r: 16, b: 40, l: 48 };
  const toPts = (rows) => rows.map(([param, acc, lat, esc]) => ({ param, acc, lat, esc }));
  const fixed = toPts(d.fixed_sweep), learned = toPts(d.learned_sweep);
  const base = [
    { name: "Small only", r: d.small_only },
    { name: "Large only", r: d.large_only },
  ];
  const all = [...fixed, ...learned, ...base.map((b) => ({ acc: b.r.accuracy, lat: b.r.mean_latency_ms }))];
  let x0 = Math.min(...all.map((p) => p.lat)), x1 = Math.max(...all.map((p) => p.lat));
  let y0 = Math.min(...all.map((p) => p.acc)), y1 = Math.max(...all.map((p) => p.acc));
  const xp = (x1 - x0) * 0.06 || 1, yp = (y1 - y0) * 0.12 || 0.01;
  x0 = Math.max(0, x0 - xp); x1 += xp; y0 -= yp; y1 += yp;
  const X = (v) => m.l + ((v - x0) / (x1 - x0)) * (W - m.l - m.r);
  const Y = (v) => H - m.b - ((v - y0) / (y1 - y0)) * (H - m.t - m.b);

  const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, "aria-hidden": "true", focusable: "false" }, host);
  for (const t of niceTicks(y0, y1, 5)) {
    svg("line", { x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t), class: "grid" }, root);
    svg("text", { x: m.l - 8, y: Y(t) + 4, "text-anchor": "end" }, root).textContent = (t * 100).toFixed(t * 100 % 1 ? 1 : 0) + "%";
  }
  for (const t of niceTicks(x0, x1, 6)) {
    svg("text", { x: X(t), y: H - m.b + 18, "text-anchor": "middle" }, root).textContent = t;
  }
  svg("text", { x: (m.l + W - m.r) / 2, y: H - 6, "text-anchor": "middle" }, root).textContent = "Mean latency per request (ms)";

  const series = [
    { pts: fixed, cls: "fixed", name: "Fixed", pname: "threshold", tuned: d.fixed_tuned },
    { pts: learned, cls: "learned", name: "Learned", pname: "cutoff", tuned: d.learned_tuned },
  ];
  const showTip = (evt, html) => {
    tip.innerHTML = html;
    tip.hidden = false;
    const box = host.parentElement.getBoundingClientRect();
    const src = evt.currentTarget.getBoundingClientRect();
    let left = src.left - box.left + src.width / 2 + 10;
    const top = src.top - box.top - 8;
    if (left + tip.offsetWidth > box.width - 8) left = src.left - box.left - tip.offsetWidth - 10;
    tip.style.left = left + "px";
    tip.style.top = Math.max(4, top) + "px";
  };
  const hideTip = () => (tip.hidden = true);
  const tipHtml = (name, p) =>
    `<strong>${name}</strong><br>Accuracy ${fmtPct(p.acc)} · ${fmtMs(p.lat)}<br>Escalated ${fmtPct(p.esc)}`;

  for (const s of series) {
        svg("polyline", {
      points: s.pts.map((p) => `${X(p.lat).toFixed(1)},${Y(p.acc).toFixed(1)}`).join(" "),
      fill: "none", "stroke-width": 2, "stroke-linejoin": "round", class: `line-${s.cls}`,
    }, root);
  }
  for (const s of series) {
    for (const p of s.pts) {
      const hit = svg("circle", { cx: X(p.lat), cy: Y(p.acc), r: 7, class: "hit" }, root);
      const html = tipHtml(`${s.name} router, ${s.pname} ${p.param.toFixed(2)}`, p);
      hit.addEventListener("mouseenter", (e) => showTip(e, html));
      hit.addEventListener("mouseleave", hideTip);
    }
  }
  for (const s of series) {
    const t = s.tuned;
    const p = { acc: t.accuracy, lat: t.mean_latency_ms, esc: t.escalation_rate };
    const c = svg("circle", {
      cx: X(p.lat), cy: Y(p.acc), r: 5.5, class: "tuned", stroke: `var(--${s.cls})`, tabindex: 0,
    }, root);
    const html = tipHtml(`${s.name} router, tuned ${s.pname} ${t.param.toFixed(2)}`, p);
    c.addEventListener("mouseenter", (e) => showTip(e, html));
    c.addEventListener("focus", (e) => showTip(e, html));
    c.addEventListener("mouseleave", hideTip);
    c.addEventListener("blur", hideTip);
  }
  for (const b of base) {
    const p = { acc: b.r.accuracy, lat: b.r.mean_latency_ms, esc: b.r.escalation_rate };
    const g = svg("g", { tabindex: 0 }, root);
    svg("rect", { x: X(p.lat) - 5, y: Y(p.acc) - 5, width: 10, height: 10, rx: 2, class: "base" }, g);
    const right = X(p.lat) > W * 0.7;
    svg("text", {
      x: X(p.lat) + (right ? -10 : 10), y: Y(p.acc) - 8, "text-anchor": right ? "end" : "start", class: "lbl",
    }, g).textContent = b.name;
    const html = tipHtml(b.name, p);
    g.addEventListener("mouseenter", (e) => showTip(e, html));
    g.addEventListener("focus", (e) => showTip(e, html));
    g.addEventListener("mouseleave", hideTip);
    g.addEventListener("blur", hideTip);
  }
  host.setAttribute("aria-label",
    `Accuracy versus mean latency on ${d.title}. Small only ${fmtPct(d.small_only.accuracy)} at ${fmtMs(d.small_only.mean_latency_ms)}; ` +
    `large only ${fmtPct(d.large_only.accuracy)} at ${fmtMs(d.large_only.mean_latency_ms)}. The table below lists the same numbers.`);
}

function drawTable(d) {
  const best = d.best_routed;
  const rows = [
    ["Small only", d.small_only, false],
    ["Large only", d.large_only, false],
    [`Best routed: ${best === "fixed_tuned" ? "fixed" : "learned"} @ ${d[best].param.toFixed(2)}`, d[best], true],
  ];
  const tbody = document.querySelector("#perf-table tbody");
  tbody.innerHTML = "";
  for (const [name, r, isBest] of rows) {
    const tr = document.createElement("tr");
    if (isBest) tr.className = "best";
    const th = document.createElement("th");
    th.scope = "row";
    th.textContent = name;
    tr.appendChild(th);
    for (const v of [fmtPct(r.accuracy), fmtPct(r.escalation_rate), r.mean_latency_ms.toFixed(2),
      r.p95_latency_ms.toFixed(2), r.rel_cost.toFixed(2) + "×"]) {
      const td = document.createElement("td");
      td.textContent = v;
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }
}

function renderPerf() {
  if (!results) return;
  const d = results.datasets[dataset];
  $("perf-sub").textContent =
    `${d.title}, n = ${d.n}. Router settings tuned on ${results.tuning.split}. ` +
    `Large model is ${d.latency_ratio}× the small model's latency here.`;
  drawChart(d);
  drawTable(d);
}

document.querySelectorAll(".ds").forEach((btn) =>
  btn.addEventListener("click", () => {
    dataset = btn.dataset.ds;
    document.querySelectorAll(".ds").forEach((b) => b.setAttribute("aria-pressed", String(b === btn)));
    renderPerf();
  })
);

async function loadResults() {
  try {
    const res = await fetch("/results");
    if (!res.ok) throw new Error();
    results = await res.json();
    learnedCutoff = results.tuning.learned_cutoff;
    if (router() === "learned") configureSlider();
    renderPerf();
  } catch (_) {
    const e = $("perf-error");
    e.textContent = "Evaluation results are not available yet.";
    e.hidden = false;
  }
}

let resizeTimer;
window.addEventListener("resize", () => {
  $("tip").hidden = true;
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(renderPerf, 150);
});
updateCount();
configureSlider();
loadResults();
loadMetrics();
setInterval(loadMetrics, 15000);
