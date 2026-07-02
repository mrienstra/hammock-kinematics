const g = 9.81;
let L = 2.0;
const L_MIN = 0.5, L_MAX = 40, L_STEPS = 1000;
function sliderToL(pos) { return L_MIN * Math.pow(L_MAX / L_MIN, pos / L_STEPS); }
function lToSlider(l) { return Math.round(Math.log(l / L_MIN) / Math.log(L_MAX / L_MIN) * L_STEPS); }
let thetaMax = (45 * Math.PI) / 180;

// --- simulation state ---
let theta = thetaMax,
  omega = 0;
let playing = true;
let mode = "v";
let unit = "m";
const CONV = { m: 1, ft: 3.28084, in: 39.3701 };
function toU(v) { return v * CONV[unit]; }
function fmtLen(m) { return (m * CONV[unit]).toFixed(unit === "in" ? 0 : 1); }
function updateUnitLabels() {
  const s = unit === "in" ? "in" : unit;
  $("u_len").textContent = unit;
  $("u_v").textContent = s + "/s";
  $("u_a").textContent = s + "/s²";
  $("u_j").textContent = s + "/s³";
  $("u_s").textContent = s + "/s⁴";
  $("u_c").textContent = s + "/s⁵";
  $("u_p").textContent = s + "/s⁶";
  $("u_tv").textContent = s + "/s";
  $("u_ta").textContent = s + "/s²";
  $("u_tj").textContent = s + "/s³";
  $("u_ts").textContent = s + "/s⁴";
  $("u_tc").textContent = s + "/s⁵";
  $("u_tp").textContent = s + "/s⁶";
  $("lenInput").value = fmtLen(L);
  $("lenInput").step = unit === "in" ? "1" : "0.1";
}

// peaks (direction-free magnitudes)
let peak = { v: 0, a: 0, j: 0, s: 0, cr: 0, p: 0, gforce: 1, res_v: 0, res_a: 0, res_j: 0, res_s: 0, res_cr: 0, res_p: 0 };

// --- DOM ---
const $ = (id) => document.getElementById(id);
const angle = $("angle"),
  length = $("length");
$("angle").addEventListener("input", (e) => {
  thetaMax = (e.target.value * Math.PI) / 180;
  $("angReadout").textContent = e.target.value;
  relaunch();
});
$("length").addEventListener("input", (e) => {
  L = sliderToL(parseFloat(e.target.value));
  $("lenInput").value = fmtLen(L);
  relaunch();
});
$("lenInput").addEventListener("change", (e) => {
  const v = parseFloat(e.target.value);
  if (isNaN(v)) { e.target.value = fmtLen(L); return; }
  L = Math.max(L_MIN, Math.min(L_MAX, v / CONV[unit]));
  e.target.value = fmtLen(L);
  $("length").value = lToSlider(L);
  relaunch();
});
$("playBtn").addEventListener("click", (e) => {
  playing = !playing;
  e.target.textContent = playing ? "Pause" : "Play";
  if (playing) {
    last = performance.now();
    requestAnimationFrame(loop);
  }
});
$("resetBtn").addEventListener("click", resetPeaks);
$("modeSeg").addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (!b) return;
  mode = b.dataset.mode;
  [...$("modeSeg").children].forEach((c) => c.classList.toggle("on", c === b));
});
$("unitSeg").addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (!b) return;
  unit = b.dataset.unit;
  [...$("unitSeg").children].forEach((c) => c.classList.toggle("on", c === b));
  updateUnitLabels();
});
function relaunch() {
  theta = thetaMax;
  omega = 0;
  resetPeaks();
  computePeriod();
}
function resetPeaks() {
  peak = { v: 0, a: 0, j: 0, s: 0, cr: 0, p: 0, gforce: 1, res_v: 0, res_a: 0, res_j: 0, res_s: 0, res_cr: 0, res_p: 0 };
  histV.fill(0);
  histA.fill(0);
  histAt.fill(0);
  histAc.fill(0);
  histJ.fill(0);
  histJt.fill(0);
  histJr.fill(0);
  histS.fill(0);
  histSt.fill(0);
  histSr.fill(0);
  histC.fill(0);
  histCt.fill(0);
  histCr.fill(0);
  histP.fill(0);
  histPt.fill(0);
  histPr.fill(0);
}

// --- accurate period via arithmetic-geometric mean (complete elliptic K) ---
function computePeriod() {
  const k = Math.sin(thetaMax / 2);
  let a = 1,
    b = Math.sqrt(1 - k * k);
  for (let i = 0; i < 20; i++) {
    const a2 = (a + b) / 2;
    b = Math.sqrt(a * b);
    a = a2;
  }
  const K = Math.PI / (2 * a);
  const T = 4 * Math.sqrt(L / g) * K;
  $("period").textContent = T.toFixed(2);
}

// --- physics derivation at current state ---
function derive() {
  const k = g / L;
  const c = Math.cos(theta),
    s = Math.sin(theta);
  // time-derivatives of angular acceleration α = θ̈ = -(g/L)sinθ
  const alpha = -k * s;
  const alphaDot = -k * c * omega; // α̇
  const alphaDot2 = -k * (c * alpha - s * omega * omega); // α̈
  const alphaDot3 = k * k * omega * (c * c - 3 * s * s) + k * c * omega ** 3; // α⃛
  const alphaDot4 =
    -(k ** 3) * s * (c * c - 3 * s * s) -
    11 * k * k * s * c * omega * omega -
    k * s * omega ** 4; // α⃜
  // velocity (tangential only)
  const vt = L * omega;
  const vmag = Math.abs(vt);
  // acceleration
  const at = L * alpha; // tangential, signed
  const ac = L * omega * omega; // centripetal, toward pivot (+)
  const amag = Math.hypot(at, ac);
  // jerk (3rd derivative of position)
  const jt = L * (alphaDot - omega ** 3); // tangential, signed
  const jr = -3 * L * omega * alpha; // along outward radial, signed
  const jmag = Math.hypot(jt, jr);
  // snap (4th derivative)
  const st = L * (alphaDot2 - 6 * omega * omega * alpha); // tangential, signed
  const sr = L * (omega ** 4 - 3 * alpha * alpha - 4 * omega * alphaDot); // outward radial, signed
  const smag = Math.hypot(st, sr);
  // crackle (5th derivative)
  const crt = L * (alphaDot3 + omega ** 5 - 15 * omega * alpha * alpha - 10 * omega * omega * alphaDot); // tangential
  const crr = L * (10 * omega ** 3 * alpha - 10 * alpha * alphaDot - 5 * omega * alphaDot2); // outward radial
  const cmag = Math.hypot(crt, crr);
  // pop (6th derivative)
  const pt = L * (alphaDot4 + 15 * omega ** 4 * alpha - 15 * alpha ** 3 - 60 * omega * alpha * alphaDot - 15 * omega * omega * alphaDot2); // tangential
  const pr = L * (-(omega ** 6) + 45 * omega * omega * alpha * alpha + 20 * omega ** 3 * alphaDot - 10 * alphaDot * alphaDot - 15 * alpha * alphaDot2 - 6 * omega * alphaDot3); // outward radial
  const pmag = Math.hypot(pt, pr);
  // felt g-force = proper acceleration / g = (g cosθ + Lω²)/g, purely radial
  const gforce = (g * Math.cos(theta) + L * omega * omega) / g;
  return { alpha, at, ac, amag, vt, vmag, jt, jr, jmag, st, sr, smag, crt, crr, cmag, pt, pr, pmag, gforce };
}

// --- RK4 integrator for (theta, omega) ---
function deriv(th, w) {
  return [w, -(g / L) * Math.sin(th)];
}
function step(dt) {
  const [k1a, k1b] = deriv(theta, omega);
  const [k2a, k2b] = deriv(theta + 0.5 * dt * k1a, omega + 0.5 * dt * k1b);
  const [k3a, k3b] = deriv(theta + 0.5 * dt * k2a, omega + 0.5 * dt * k2b);
  const [k4a, k4b] = deriv(theta + dt * k3a, omega + dt * k3b);
  theta += (dt / 6) * (k1a + 2 * k2a + 2 * k3a + k4a);
  omega += (dt / 6) * (k1b + 2 * k2b + 2 * k3b + k4b);
}

// --- canvas setup with devicePixelRatio ---
function fit(cv, cssH) {
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth || cv.parentElement.clientWidth;
  cv.width = w * dpr;
  cv.height = cssH * dpr;
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h: cssH };
}
const stage = $("stage");
let ST;
function sizeStage() {
  stage.style.height = "360px";
  ST = fit(stage, 360);
}
window.addEventListener("resize", () => {
  sizeStage();
  sizeTraces();
});

// arrow helper
function arrow(ctx, x0, y0, x1, y1, color, width) {
  const dx = x1 - x0,
    dy = y1 - y0,
    len = Math.hypot(dx, dy);
  if (len < 2) return;
  const a = Math.atan2(dy, dx),
    hl = Math.min(11, len * 0.5);
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  ctx.moveTo(x0, y0);
  ctx.lineTo(x1 - hl * Math.cos(a), y1 - hl * Math.sin(a));
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(
    x1 - hl * Math.cos(a - Math.PI / 6),
    y1 - hl * Math.sin(a - Math.PI / 6),
  );
  ctx.lineTo(
    x1 - hl * Math.cos(a + Math.PI / 6),
    y1 - hl * Math.sin(a + Math.PI / 6),
  );
  ctx.closePath();
  ctx.fill();
}

function drawStage(d) {
  const { ctx, w, h } = ST;
  ctx.clearRect(0, 0, w, h);
  const px = w / 2,
    py = 44;
  const Lpx = Math.min(h - 120, 210);
  const bx = px + Math.sin(theta) * Lpx,
    by = py + Math.cos(theta) * Lpx;
  // ceiling
  ctx.strokeStyle = "#94a3b8";
  ctx.lineWidth = 5;
  ctx.beginPath();
  ctx.moveTo(px - 70, py);
  ctx.lineTo(px + 70, py);
  ctx.stroke();
  for (let i = -60; i <= 60; i += 15) {
    ctx.beginPath();
    ctx.moveTo(px + i, py);
    ctx.lineTo(px + i - 8, py - 8);
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }
  // arc guide
  ctx.strokeStyle = "#eef1f4";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.arc(px, py, Lpx, Math.PI / 2 - thetaMax, Math.PI / 2 + thetaMax);
  ctx.stroke();
  // suspension
  ctx.strokeStyle = "#64748b";
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  ctx.moveTo(px, py);
  ctx.lineTo(bx, by);
  ctx.stroke();
  // unit directions
  const tx = Math.cos(theta),
    ty = -Math.sin(theta); // tangential (increasing θ)
  const rx = Math.sin(theta),
    ry = Math.cos(theta); // outward radial

  // pick vectors + auto-scale by this mode's peak resultant
  let comps = [],
    resKey;
  if (mode === "v") {
    comps = [["tan", d.vt, "--tan"]];
    resKey = "res_v";
  } else if (mode === "a") {
    comps = [
      ["tan", d.at, "--tan"],
      ["rad", -d.ac, "--cen"],
    ];
    resKey = "res_a";
  } // ac drawn toward pivot
  else if (mode === "j") {
    comps = [
      ["tan", d.jt, "--tan"],
      ["rad", d.jr, "--cen"],
    ];
    resKey = "res_j";
  } else if (mode === "s") {
    comps = [
      ["tan", d.st, "--tan"],
      ["rad", d.sr, "--cen"],
    ];
    resKey = "res_s";
  } else if (mode === "cr") {
    comps = [
      ["tan", d.crt, "--tan"],
      ["rad", d.crr, "--cen"],
    ];
    resKey = "res_cr";
  } else {
    comps = [
      ["tan", d.pt, "--tan"],
      ["rad", d.pr, "--cen"],
    ];
    resKey = "res_p";
  }

  // resultant vector (world units) for scaling + drawing
  let Rx = 0,
    Ry = 0;
  comps.forEach(([kind, val]) => {
    if (kind === "tan") {
      Rx += val * tx;
      Ry += val * ty;
    } else {
      Rx += val * rx;
      Ry += val * ry;
    }
  });
  const resMag = Math.hypot(Rx, Ry);
  peak[resKey] = Math.max(peak[resKey], resMag);
  const scale = 110 / Math.max(peak[resKey], 1e-6);

  const css = getComputedStyle(document.documentElement);
  const col = (n) => css.getPropertyValue(n).trim();
  // draw component arrows
  comps.forEach(([kind, val, c]) => {
    let ux, uy;
    if (kind === "tan") {
      ux = tx;
      uy = ty;
    } else {
      ux = rx;
      uy = ry;
    }
    arrow(ctx, bx, by, bx + ux * val * scale, by + uy * val * scale, col(c), 3);
  });
  // resultant (only meaningful when 2 comps)
  if (comps.length > 1)
    arrow(ctx, bx, by, bx + Rx * scale, by + Ry * scale, col("--res"), 4.5);

  // bob
  ctx.fillStyle = "#334155";
  ctx.beginPath();
  ctx.arc(bx, by, 17, 0, 7);
  ctx.fill();
}

// --- traces ---
const N = 280;
const histV = new Array(N).fill(0),
  histA = new Array(N).fill(0),
  histAt = new Array(N).fill(0),
  histAc = new Array(N).fill(0),
  histJ = new Array(N).fill(0),
  histJt = new Array(N).fill(0),
  histJr = new Array(N).fill(0),
  histS = new Array(N).fill(0),
  histSt = new Array(N).fill(0),
  histSr = new Array(N).fill(0),
  histC = new Array(N).fill(0),
  histCt = new Array(N).fill(0),
  histCr = new Array(N).fill(0),
  histP = new Array(N).fill(0),
  histPt = new Array(N).fill(0),
  histPr = new Array(N).fill(0);
let CV, CA, CJ, CS, CC, CP;
function sizeTraces() {
  CV = fit($("cv"), 120);
  CA = fit($("ca"), 120);
  CJ = fit($("cj"), 120);
  CS = fit($("cs"), 120);
  CC = fit($("cc"), 120);
  CP = fit($("cp"), 120);
}
// series: [{hist, color, width?}, ...]  — drawn in order (last on top)
function drawTrace(T, series, peakVal) {
  const { ctx, w, h } = T;
  ctx.clearRect(0, 0, w, h);
  const max = Math.max(peakVal, 1e-6);
  // peak-hold line
  ctx.strokeStyle = "#e2e8f0";
  ctx.setLineDash([4, 4]);
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, 6);
  ctx.lineTo(w, 6);
  ctx.stroke();
  ctx.setLineDash([]);
  // baseline
  ctx.strokeStyle = "#eef1f4";
  ctx.beginPath();
  ctx.moveTo(0, h - 2);
  ctx.lineTo(w, h - 2);
  ctx.stroke();
  // traces
  for (const { hist, color, width } of series) {
    ctx.strokeStyle = color;
    ctx.lineWidth = width ?? 1.8;
    ctx.beginPath();
    for (let i = 0; i < N; i++) {
      const x = (i / (N - 1)) * w,
        y = h - 2 - (hist[i] / max) * (h - 10);
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    }
    ctx.stroke();
  }
}

// --- main loop ---
let last = performance.now(),
  acc = 0;
const FIXED = 1 / 600; // 600 Hz internal integration
function loop(now) {
  let frame = (now - last) / 1000;
  last = now;
  if (frame > 0.05) frame = 0.05;
  acc += frame;
  while (acc >= FIXED) {
    step(FIXED);
    acc -= FIXED;
  }

  const d = derive();
  // peaks
  peak.v = Math.max(peak.v, d.vmag);
  peak.a = Math.max(peak.a, d.amag);
  peak.j = Math.max(peak.j, d.jmag);
  peak.s = Math.max(peak.s, d.smag);
  peak.cr = Math.max(peak.cr, d.cmag);
  peak.p = Math.max(peak.p, d.pmag);
  peak.gforce = Math.max(peak.gforce, d.gforce);
  // histories
  histV.push(d.vmag);
  histV.shift();
  histA.push(d.amag);
  histA.shift();
  histAt.push(Math.abs(d.at));
  histAt.shift();
  histAc.push(d.ac);
  histAc.shift();
  histJ.push(d.jmag);
  histJ.shift();
  histJt.push(Math.abs(d.jt));
  histJt.shift();
  histJr.push(Math.abs(d.jr));
  histJr.shift();
  histS.push(d.smag);
  histS.shift();
  histSt.push(Math.abs(d.st));
  histSt.shift();
  histSr.push(Math.abs(d.sr));
  histSr.shift();
  histC.push(d.cmag);
  histC.shift();
  histCt.push(Math.abs(d.crt));
  histCt.shift();
  histCr.push(Math.abs(d.crr));
  histCr.shift();
  histP.push(d.pmag);
  histP.shift();
  histPt.push(Math.abs(d.pt));
  histPt.shift();
  histPr.push(Math.abs(d.pr));
  histPr.shift();

  // table
  $("v_t").textContent = toU(d.vmag).toFixed(2);
  $("v_m").textContent = toU(d.vmag).toFixed(2);
  $("v_p").textContent = toU(peak.v).toFixed(2);
  $("a_t").textContent = toU(Math.abs(d.at)).toFixed(2);
  $("a_c").textContent = toU(d.ac).toFixed(2);
  $("a_m").textContent = toU(d.amag).toFixed(2);
  $("a_p").textContent = toU(peak.a).toFixed(2);
  $("j_t").textContent = toU(Math.abs(d.jt)).toFixed(1);
  $("j_c").textContent = toU(Math.abs(d.jr)).toFixed(1);
  $("j_m").textContent = toU(d.jmag).toFixed(1);
  $("j_p").textContent = toU(peak.j).toFixed(1);
  $("s_t").textContent = toU(Math.abs(d.st)).toFixed(1);
  $("s_c").textContent = toU(Math.abs(d.sr)).toFixed(1);
  $("s_m").textContent = toU(d.smag).toFixed(1);
  $("s_p").textContent = toU(peak.s).toFixed(1);
  $("cr_t").textContent = toU(Math.abs(d.crt)).toFixed(1);
  $("cr_c").textContent = toU(Math.abs(d.crr)).toFixed(1);
  $("cr_m").textContent = toU(d.cmag).toFixed(1);
  $("cr_p").textContent = toU(peak.cr).toFixed(1);
  $("p_t").textContent = toU(Math.abs(d.pt)).toFixed(1);
  $("p_c").textContent = toU(Math.abs(d.pr)).toFixed(1);
  $("p_m").textContent = toU(d.pmag).toFixed(1);
  $("p_p").textContent = toU(peak.p).toFixed(1);
  // gforce
  $("gNow").textContent = d.gforce.toFixed(2);
  $("gPk").textContent = peak.gforce.toFixed(2);
  // trace readouts
  $("tv").textContent = toU(d.vmag).toFixed(2);
  $("tvp").textContent = toU(peak.v).toFixed(2);
  $("ta_t").textContent = toU(Math.abs(d.at)).toFixed(2);
  $("ta_c").textContent = toU(d.ac).toFixed(2);
  $("ta_m").textContent = toU(d.amag).toFixed(2);
  $("tap").textContent = toU(peak.a).toFixed(2);
  $("tj_t").textContent = toU(Math.abs(d.jt)).toFixed(1);
  $("tj_c").textContent = toU(Math.abs(d.jr)).toFixed(1);
  $("tj_m").textContent = toU(d.jmag).toFixed(1);
  $("tjp").textContent = toU(peak.j).toFixed(1);
  $("ts_t").textContent = toU(Math.abs(d.st)).toFixed(1);
  $("ts_c").textContent = toU(Math.abs(d.sr)).toFixed(1);
  $("ts_m").textContent = toU(d.smag).toFixed(1);
  $("tsp").textContent = toU(peak.s).toFixed(1);
  $("tc_t").textContent = toU(Math.abs(d.crt)).toFixed(1);
  $("tc_c").textContent = toU(Math.abs(d.crr)).toFixed(1);
  $("tc_m").textContent = toU(d.cmag).toFixed(1);
  $("tcp").textContent = toU(peak.cr).toFixed(1);
  $("tp_t").textContent = toU(Math.abs(d.pt)).toFixed(1);
  $("tp_c").textContent = toU(Math.abs(d.pr)).toFixed(1);
  $("tp_m").textContent = toU(d.pmag).toFixed(1);
  $("tpp").textContent = toU(peak.p).toFixed(1);

  // render
  drawStage(d);
  const css = getComputedStyle(document.documentElement);
  const tanColor = css.getPropertyValue("--tan").trim();
  const cenColor = css.getPropertyValue("--cen").trim();
  const resColor = css.getPropertyValue("--res").trim();
  drawTrace(CV, [{ hist: histV, color: resColor, width: 2 }], peak.v);
  drawTrace(
    CA,
    [
      { hist: histAt, color: tanColor, width: 1.5 },
      { hist: histAc, color: cenColor, width: 1.5 },
      { hist: histA, color: resColor, width: 2.5 },
    ],
    peak.a,
  );
  drawTrace(
    CJ,
    [
      { hist: histJt, color: tanColor, width: 1.5 },
      { hist: histJr, color: cenColor, width: 1.5 },
      { hist: histJ, color: resColor, width: 2.5 },
    ],
    peak.j,
  );
  drawTrace(
    CS,
    [
      { hist: histSt, color: tanColor, width: 1.5 },
      { hist: histSr, color: cenColor, width: 1.5 },
      { hist: histS, color: resColor, width: 2.5 },
    ],
    peak.s,
  );
  drawTrace(
    CC,
    [
      { hist: histCt, color: tanColor, width: 1.5 },
      { hist: histCr, color: cenColor, width: 1.5 },
      { hist: histC, color: resColor, width: 2.5 },
    ],
    peak.cr,
  );
  drawTrace(
    CP,
    [
      { hist: histPt, color: tanColor, width: 1.5 },
      { hist: histPr, color: cenColor, width: 1.5 },
      { hist: histP, color: resColor, width: 2.5 },
    ],
    peak.p,
  );

  if (playing) requestAnimationFrame(loop);
}

// init
sizeStage();
sizeTraces();
computePeriod();
requestAnimationFrame(loop);
