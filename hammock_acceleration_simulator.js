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
  $("u_tv").textContent = s + "/s";
  $("u_ta").textContent = s + "/s²";
  $("u_tj").textContent = s + "/s³";
  $("lenInput").value = fmtLen(L);
  $("lenInput").step = unit === "in" ? "1" : "0.1";
}

// peaks (direction-free magnitudes)
let peak = { v: 0, a: 0, j: 0, gforce: 1, res_v: 0, res_a: 0, res_j: 0 };

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
  peak = { v: 0, a: 0, j: 0, gforce: 1, res_v: 0, res_a: 0, res_j: 0 };
  histV.fill(0);
  histA.fill(0);
  histAt.fill(0);
  histAc.fill(0);
  histJ.fill(0);
  histJt.fill(0);
  histJr.fill(0);
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
  const alpha = -(g / L) * Math.sin(theta);
  const alphaDot = -(g / L) * Math.cos(theta) * omega;
  // velocity (tangential only)
  const vt = L * omega;
  const vmag = Math.abs(vt);
  // acceleration
  const at = L * alpha; // tangential, signed
  const ac = L * omega * omega; // centripetal, toward pivot (+)
  const amag = Math.hypot(at, ac);
  // jerk
  const jt = L * (alphaDot - omega ** 3); // tangential, signed
  const jr = -3 * L * omega * alpha; // along outward radial, signed
  const jmag = Math.hypot(jt, jr);
  // felt g-force = proper acceleration / g = (g cosθ + Lω²)/g, purely radial
  const gforce = (g * Math.cos(theta) + L * omega * omega) / g;
  return { alpha, at, ac, amag, vt, vmag, jt, jr, jmag, gforce };
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
  else {
    comps = [
      ["tan", d.jt, "--tan"],
      ["rad", d.jr, "--cen"],
    ];
    resKey = "res_j";
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
  histJr = new Array(N).fill(0);
let CV, CA, CJ;
function sizeTraces() {
  CV = fit($("cv"), 120);
  CA = fit($("ca"), 120);
  CJ = fit($("cj"), 120);
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

  if (playing) requestAnimationFrame(loop);
}

// =====================================================================
//  MEASURE MODE — recover swing params from phone gyro + accelerometer
// =====================================================================
//
//  Physics used (see the forward model above):
//    • The gyroscope reads the swing's angular velocity ω(t). It's the
//      same everywhere on a rigid body, so it works regardless of where
//      the phone sits. Zero-crossings give the period T; for SHM the
//      peak speed gives amplitude θmax = ωpeak·T/2π.
//    • Period → effective length via T = 2π√(L/g)·(2K(sin θmax/2)/π),
//      inverting computePeriod(). Small-angle first, then one K refine.
//    • The accelerometer (including gravity) reads a purely radial
//      specific force  g·cosθ + r·ω²  (tangential gravity is unfelt in
//      free swing). At the bottom θ≈0 so |a|max ≈ g + r·ωpeak² → phone
//      radius r; at a turning point ω=0 so |a|min = g·cosθmax, an
//      independent amplitude check. r vs L tells you where the phone is.

const MEAS = {
  on: false,
  g: 9.81,
  // rolling buffer of raw angular-velocity vectors (rad/s) for axis fit
  wbuf: [],
  WBUF_MAX: 240,
  axis: [1, 0, 0], // principal swing axis, updated by power iteration
  sinceAxis: 0,
  // signed angular speed about the axis, for the live trace
  sbuf: new Array(N).fill(0),
  // half-cycle accumulators (reset at each turning point)
  prevS: 0,
  peakSpeed: 0, // max |s| this half-cycle
  aMax: 0, // max |accel| this half-cycle
  aMin: Infinity, // min |accel| this half-cycle
  lastCrossT: 0,
  halfPeriods: [], // recent half-period durations (s)
  // per-cycle amplitude log for the decay fit: {t, theta}
  amps: [],
  T: 0,
  L: 0,
  thetaMax: 0,
  r: 0,
  ready: false,
};

const $$ = (id) => document.getElementById(id);
function mStatus(msg, isErr) {
  const el = $$("measStatus");
  el.textContent = msg || "";
  el.classList.toggle("err", !!isErr);
}

// dominant axis of the buffered ω vectors via power iteration on ΣωωT
function principalAxis(buf, seed) {
  let cxx = 0, cyy = 0, czz = 0, cxy = 0, cxz = 0, cyz = 0;
  for (const [x, y, z] of buf) {
    cxx += x * x; cyy += y * y; czz += z * z;
    cxy += x * y; cxz += x * z; cyz += y * z;
  }
  let vx = seed[0], vy = seed[1], vz = seed[2];
  for (let i = 0; i < 16; i++) {
    const nx = cxx * vx + cxy * vy + cxz * vz;
    const ny = cxy * vx + cyy * vy + cyz * vz;
    const nz = cxz * vx + cyz * vy + czz * vz;
    const m = Math.hypot(nx, ny, nz) || 1;
    vx = nx / m; vy = ny / m; vz = nz / m;
  }
  // keep sign continuity with the previous axis
  if (vx * seed[0] + vy * seed[1] + vz * seed[2] < 0) { vx = -vx; vy = -vy; vz = -vz; }
  return [vx, vy, vz];
}

// invert T = 2π√(L/g)·(2K/π) for L, given θmax (K via AGM, as computePeriod)
function periodToLength(T, thetaMax) {
  const k = Math.sin(thetaMax / 2);
  let a = 1, b = Math.sqrt(1 - k * k);
  for (let i = 0; i < 20; i++) { const a2 = (a + b) / 2; b = Math.sqrt(a * b); a = a2; }
  const K = Math.PI / (2 * a); // = π/2 at small angle
  const bigOverSmall = (2 * K) / Math.PI; // ≥ 1, period stretch
  return (MEAS.g * (T / (2 * Math.PI)) ** 2) / (bigOverSmall * bigOverSmall);
}

function onMotion(ev) {
  if (!MEAS.on) return;
  const rr = ev.rotationRate;
  const ai = ev.accelerationIncludingGravity;
  MEAS.gotEvent = true;
  if (!rr || rr.alpha == null || !ai) {
    mStatus(
      "Motion events are arriving but without sensor values" +
        (rr ? "" : " (rotationRate missing)") +
        (ai ? "" : " (accelerationIncludingGravity missing)") +
        ". This device/browser may not expose them.",
      true,
    );
    return;
  }
  if (!MEAS.rawShown) { MEAS.rawShown = true; mStatus("Recording… lie back and let it swing. Numbers refine each cycle."); }
  let dt = ev.interval || 1 / 60;
  if (dt <= 0 || dt > 0.1) dt = 1 / 60;
  const D = Math.PI / 180;
  const w = [rr.beta * D, rr.gamma * D, rr.alpha * D]; // deg/s → rad/s
  const amag = Math.hypot(ai.x, ai.y, ai.z);

  // maintain the principal-axis estimate
  MEAS.wbuf.push(w);
  if (MEAS.wbuf.length > MEAS.WBUF_MAX) MEAS.wbuf.shift();
  if (++MEAS.sinceAxis >= 8 && MEAS.wbuf.length >= 30) {
    MEAS.axis = principalAxis(MEAS.wbuf, MEAS.axis);
    MEAS.sinceAxis = 0;
  }
  const s = w[0] * MEAS.axis[0] + w[1] * MEAS.axis[1] + w[2] * MEAS.axis[2];

  // live trace of signed angular speed
  MEAS.sbuf.push(s); MEAS.sbuf.shift();

  // half-cycle tracking
  const t = ev.timeStamp / 1000;
  const sp = Math.abs(s);
  if (sp > MEAS.peakSpeed) MEAS.peakSpeed = sp;
  if (amag > MEAS.aMax) MEAS.aMax = amag;
  if (amag < MEAS.aMin) MEAS.aMin = amag;

  // detect a turning point: sign flip of s, gated by a min amplitude so
  // noise near rest doesn't register as swings
  const gate = Math.max(0.25, 0.35 * MEAS.peakSpeed);
  if (MEAS.prevS <= 0 && s > 0 && MEAS.peakSpeed > gate) registerHalf(t);
  else if (MEAS.prevS >= 0 && s < 0 && MEAS.peakSpeed > gate) registerHalf(t);
  MEAS.prevS = s;
}

function registerHalf(t) {
  if (MEAS.lastCrossT > 0) {
    const half = t - MEAS.lastCrossT;
    if (half > 0.15 && half < 8) {
      MEAS.halfPeriods.push(half);
      if (MEAS.halfPeriods.length > 8) MEAS.halfPeriods.shift();
      solveFromCycle(t);
    }
  }
  MEAS.lastCrossT = t;
  MEAS.peakSpeed = 0;
  MEAS.aMax = 0;
  MEAS.aMin = Infinity;
}

function solveFromCycle(t) {
  const hp = MEAS.halfPeriods;
  if (hp.length < 2) return;
  const T = 2 * (hp.reduce((a, b) => a + b, 0) / hp.length);
  const wpeak = MEAS.peakSpeed;
  // amplitude, primary estimate: SHM θmax = ωpeak·T/2π
  let thetaMax = wpeak * T / (2 * Math.PI);
  // accelerometer cross-check via |a|min = g·cosθmax (turning point)
  if (MEAS.aMin < MEAS.g && MEAS.aMin > 0) {
    const thAcc = Math.acos(Math.min(1, MEAS.aMin / MEAS.g));
    thetaMax = 0.5 * thetaMax + 0.5 * thAcc; // blend gyro + accel
  }
  thetaMax = Math.min(thetaMax, (89 * Math.PI) / 180);
  const L = periodToLength(T, thetaMax);
  // phone radius from |a|max ≈ g + r·ωpeak² (θ≈0 at max speed)
  const r = wpeak > 0.05 ? Math.max(0, (MEAS.aMax - MEAS.g) / (wpeak * wpeak)) : 0;

  MEAS.T = T; MEAS.thetaMax = thetaMax; MEAS.L = L; MEAS.r = r;
  MEAS.amps.push({ t, theta: thetaMax });
  if (MEAS.amps.length > 40) MEAS.amps.shift();
  MEAS.ready = true;
  renderMeasure();
}

// exponential envelope fit θ = θ0·e^(−t/τ) over recent cycle amplitudes
function decayHint() {
  const a = MEAS.amps;
  if (a.length < 4) return "";
  const t0 = a[0].t;
  let n = 0, sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (const p of a) {
    if (p.theta <= 0) continue;
    const x = p.t - t0, y = Math.log(p.theta);
    n++; sx += x; sy += y; sxx += x * x; sxy += x * y;
  }
  const denom = n * sxx - sx * sx;
  if (n < 4 || denom === 0) return "";
  const slope = (n * sxy - sx * sy) / denom; // d ln θ / dt = −1/τ
  if (slope >= -1e-4) return "Swing looks steady — little decay yet.";
  const half = Math.log(2) / -slope; // time to halve the amplitude
  return `Amplitude halving every ≈ ${half.toFixed(0)} s (fading).`;
}

function renderMeasure() {
  $$("measReadouts").hidden = false;
  $$("mTrace").hidden = false;
  $$("measFoot").hidden = false;
  $$("mPeriod").textContent = MEAS.T.toFixed(2);
  $$("mAngle").textContent = ((MEAS.thetaMax * 180) / Math.PI).toFixed(0);
  $$("mLen").textContent = fmtLen(MEAS.L);
  $$("mLenU").textContent = unit;
  $$("mRad").textContent = MEAS.r > 0 ? fmtLen(MEAS.r) : "—";
  $$("mRadU").textContent = unit;

  const parts = [];
  if (MEAS.r > 0) {
    const ratio = MEAS.r / MEAS.L;
    if (ratio > 1.12) parts.push("Phone reads farther than the effective length → it's below your center of gravity.");
    else if (ratio < 0.88) parts.push("Phone reads shorter → it's above your center of gravity (nearer the pivot).");
    else parts.push("Phone radius ≈ effective length → it's near your center of gravity. Nice placement.");
  }
  const dh = decayHint();
  if (dh) parts.push(dh);
  $$("mHint").textContent = parts.join(" ") || "Swing for a few cycles for a reading…";
  $$("applyBtn").disabled = !(MEAS.ready && MEAS.L > 0);
}

// signed live trace (zero line in the middle)
let MT;
function sizeMTrace() { MT = fit($$("mTrace"), 120); }
function drawMTrace() {
  if (!MT || $$("mTrace").hidden) return;
  const { ctx, w, h } = MT;
  ctx.clearRect(0, 0, w, h);
  const mid = h / 2;
  let max = 0;
  for (const v of MEAS.sbuf) max = Math.max(max, Math.abs(v));
  max = Math.max(max, 0.5);
  // zero line
  ctx.strokeStyle = "#e2e8f0";
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(0, mid); ctx.lineTo(w, mid); ctx.stroke();
  // ω(t)
  const css = getComputedStyle(document.documentElement);
  ctx.strokeStyle = css.getPropertyValue("--accent").trim();
  ctx.lineWidth = 1.8;
  ctx.beginPath();
  for (let i = 0; i < N; i++) {
    const x = (i / (N - 1)) * w;
    const y = mid - (MEAS.sbuf[i] / max) * (mid - 6);
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  }
  ctx.stroke();
}

function measureLoop() {
  if (!MEAS.on) return;
  drawMTrace();
  requestAnimationFrame(measureLoop);
}

async function startMeasuring() {
  // immediate synchronous feedback so a tap always visibly does something
  mStatus("Starting…");
  if (typeof DeviceMotionEvent === "undefined") {
    mStatus("This browser/device doesn't expose DeviceMotionEvent (motion sensors).", true);
    return;
  }
  const hasReq = typeof DeviceMotionEvent.requestPermission === "function";
  mStatus(
    "DeviceMotionEvent found" +
      (hasReq ? ", requesting permission…" : " (no permission prompt on this OS)…") +
      (window.isSecureContext ? "" : "  ⚠︎ page is NOT a secure (https) context — iOS blocks sensors here."),
  );
  // iOS 13+ requires an explicit permission request from a user gesture.
  // Call it FIRST, before any other await, or Safari drops the gesture.
  if (hasReq) {
    let res;
    try {
      res = await DeviceMotionEvent.requestPermission();
    } catch (e) {
      mStatus("requestPermission() threw: " + (e && (e.name + ": " + e.message)) +
        (window.isSecureContext ? "" : " — likely because this isn't an https page."), true);
      return;
    }
    if (res !== "granted") {
      mStatus('Motion access was "' + res + '" (not granted). Enable it and tap Start again.', true);
      return;
    }
  }
  MEAS.on = true;
  MEAS.gotEvent = false;
  MEAS.wbuf = []; MEAS.halfPeriods = []; MEAS.amps = [];
  MEAS.sbuf.fill(0); MEAS.lastCrossT = 0; MEAS.peakSpeed = 0;
  MEAS.aMax = 0; MEAS.aMin = Infinity; MEAS.prevS = 0; MEAS.ready = false;
  MEAS.rawShown = false;
  sizeMTrace();
  window.addEventListener("devicemotion", onMotion);
  mStatus("Permission OK — waiting for motion data…");
  const btn = $$("measBtn");
  btn.textContent = "Stop"; btn.classList.add("recording");
  renderMeasure();
  requestAnimationFrame(measureLoop);
  // watchdog: if no devicemotion events arrive, say so instead of hanging
  setTimeout(() => {
    if (MEAS.on && !MEAS.gotEvent) {
      mStatus(
        "No motion events are arriving. On iPhone check Settings ▸ Safari ▸ " +
          "Motion & Orientation Access is ON, then reload and tap Start.",
        true,
      );
    }
  }, 2000);
}

function stopMeasuring() {
  MEAS.on = false;
  window.removeEventListener("devicemotion", onMotion);
  const btn = $$("measBtn");
  btn.textContent = "Start measuring"; btn.classList.remove("recording");
  mStatus(MEAS.ready ? "Stopped. Tap “Apply to simulator” to replay this swing above." : "Stopped.");
}

// surface any uncaught error into the UI (no desktop console on the phone)
window.addEventListener("error", (e) =>
  mStatus("Script error: " + (e.message || e.error) + (e.filename ? " @ " + e.lineno : ""), true));
window.addEventListener("unhandledrejection", (e) =>
  mStatus("Unhandled promise rejection: " + (e.reason && (e.reason.message || e.reason)), true));

$$("measBtn").addEventListener("click", () => {
  try {
    if (MEAS.on) stopMeasuring();
    else startMeasuring().catch((e) => mStatus("startMeasuring failed: " + (e && (e.name + ": " + e.message)), true));
  } catch (e) {
    mStatus("Click handler threw: " + (e && (e.name + ": " + e.message)), true);
  }
});
$$("applyBtn").addEventListener("click", () => {
  if (!(MEAS.L > 0)) return;
  L = Math.max(L_MIN, Math.min(L_MAX, MEAS.L));
  thetaMax = Math.max((5 * Math.PI) / 180, Math.min((90 * Math.PI) / 180, MEAS.thetaMax));
  const deg = Math.round((thetaMax * 180) / Math.PI);
  $("angle").value = deg;
  $("angReadout").textContent = deg;
  $("length").value = lToSlider(L);
  $("lenInput").value = fmtLen(L);
  relaunch();
  window.scrollTo({ top: 0, behavior: "smooth" });
});
window.addEventListener("resize", () => { if (MEAS.on) sizeMTrace(); });

// init
sizeStage();
sizeTraces();
computePeriod();
requestAnimationFrame(loop);
