// Measure mode — the inverse model: recover swing params from the phone.
// Loaded AFTER simulator.js; relies on these globals from it (classic
// scripts share one global scope): $, fit, N, unit, fmtLen, lToSlider,
// relaunch, computePeriod, and the mutable L / thetaMax bindings.

// =====================================================================
//  MEASURE MODE — recover swing params from phone gyro + accelerometer
// =====================================================================
//
//  Physics used (see the forward model in simulator.js):
//    • The gyroscope reads the swing's angular velocity ω(t). It's the
//      same everywhere on a rigid body, so it works regardless of where
//      the phone sits. Zero-crossings give the period T; amplitude comes
//      from the exact energy relation ωpeak² = (2g/L)(1 − cosθmax).
//    • Period → effective length via T = 2π√(L/g)·(2K(sin θmax/2)/π),
//      inverting computePeriod(). Uses true g = 9.81 (dynamics), never the
//      accelerometer's scale.
//    • The accelerometer (including gravity) reads ≈ g·cosθ + r·ω² at
//      phone radius r (tangential gravity is unfelt at the bob; the
//      residual g·sinθ·(1−r/L) term is ≲0.02 m/s² at these r/L — below
//      sensor noise). Within a half-cycle, energy makes cosθ linear in
//      ω², so regressing |a| on ω² gives  intercept = g_cal·cosθmax
//      (self-calibrates the device's accel scale, typically 1–3% off)
//      and  slope = (g_cal/g)·(L/2 + r) → r. r vs L locates the phone
//      relative to the rider's center of gravity.

// True gravity for the DYNAMICS (period → length). Fixed: the pendulum's
// period depends on real g, not on the phone's accelerometer scale.
const G_TRUE = 9.81;

const MEAS = {
  on: false,
  // g here is the phone's CALIBRATED gravity magnitude (may differ from
  // G_TRUE by the accelerometer's ~1-2% scale error). Used only for the
  // felt-force analysis (r, angle cross-check), never for L. Recalibrated
  // per recording from the |a|-vs-ω² intercept.
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
  // per-half-cycle sums for the |a| = g·cosθmax + (L/2 + r)·ω² regression
  // (cosθ and ω² are collinear via energy, so this single-regressor fit of
  // |a| against ω² is the well-conditioned form): g = intercept/cosθmax,
  // r = slope − L/2. Reset each turning point.
  rw2: 0, rw4: 0, ra: 0, raw2: 0, rn: 0,
  lastCrossT: 0,
  halfPeriods: [], // recent half-period durations (s)
  // per-cycle amplitude log for the decay fit: {t, theta}
  amps: [],
  T: 0,
  L: 0,
  thetaMax: 0,
  r: 0,
  rOk: false, // is r well-enough constrained to report?
  ready: false,
  // full recording for export
  samples: [], // per-event raw + derived
  cycles: [], // per-cycle solved estimates
  t0: null, // analysis time origin (s): first RECORDED sample
  // warm-up: ignore the first skipSec so repositioning / pocketing the
  // phone doesn't pollute the fit
  warm: true,
  skipSec: 3,
  startT: null, // timestamp of the very first event (warm-up clock)
  // cool-down: retroactively drop the last trimSec at Stop (e.g. fishing the
  // phone back out of a pocket)
  trimSec: 3,
  trimApplied: 0, // what was actually cut (0 until Stop applies it)
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
  return (G_TRUE * (T / (2 * Math.PI)) ** 2) / (bigOverSmall * bigOverSmall);
}

// clear the recording / per-cycle analysis state (but keep the axis buffer
// warm so it stays locked). t0 = new analysis time origin.
function resetAnalysis(t0) {
  MEAS.samples = []; MEAS.cycles = []; MEAS.amps = [];
  MEAS.halfPeriods = [];
  MEAS.lastCrossT = 0; MEAS.peakSpeed = 0; MEAS.aMax = 0; MEAS.aMin = Infinity;
  MEAS.rw2 = MEAS.rw4 = MEAS.ra = MEAS.raw2 = MEAS.rn = 0;
  MEAS.prevS = 0; MEAS.primed = false; MEAS.ready = false;
  MEAS.g = 9.81; MEAS.r = 0; MEAS.rOk = false; // recalibrate g per recording
  MEAS.trimApplied = 0;
  MEAS.t0 = t0 === undefined ? null : t0;
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
  let dt = ev.interval || 1 / 60;
  if (dt <= 0 || dt > 0.1) dt = 1 / 60;
  const D = Math.PI / 180;
  const w = [rr.beta * D, rr.gamma * D, rr.alpha * D]; // deg/s → rad/s
  const amag = Math.hypot(ai.x, ai.y, ai.z);
  const t = ev.timeStamp / 1000;
  if (MEAS.startT == null) MEAS.startT = t;

  // maintain the principal-axis estimate + live trace, even during warm-up so
  // the signal visibly moves while the user repositions
  MEAS.wbuf.push(w);
  if (MEAS.wbuf.length > MEAS.WBUF_MAX) MEAS.wbuf.shift();
  if (++MEAS.sinceAxis >= 8 && MEAS.wbuf.length >= 30) {
    MEAS.axis = principalAxis(MEAS.wbuf, MEAS.axis);
    MEAS.sinceAxis = 0;
  }
  const s = w[0] * MEAS.axis[0] + w[1] * MEAS.axis[1] + w[2] * MEAS.axis[2];
  MEAS.sbuf.push(s); MEAS.sbuf.shift();

  // warm-up window: preview the trace but don't record or analyze yet
  if (MEAS.warm) {
    const remain = MEAS.skipSec - (t - MEAS.startT);
    if (remain > 0) {
      mStatus("Warming up — reposition / pocket the phone now. Recording in " + remain.toFixed(1) + " s…");
      return;
    }
    MEAS.warm = false; // discard the warm-up; the clock starts clean here
    resetAnalysis(t);
  }
  if (!MEAS.rawShown) { MEAS.rawShown = true; mStatus("Recording… lie back and let it swing. Numbers refine each cycle."); }
  if (MEAS.t0 == null) MEAS.t0 = t;

  // full recording (raw + derived) for later inspection
  MEAS.samples.push({
    t: +(t - MEAS.t0).toFixed(4),
    dt: +dt.toFixed(4),
    // raw gyro (deg/s, as the API reports) and accel incl. gravity (m/s²)
    rrAlpha: rr.alpha, rrBeta: rr.beta, rrGamma: rr.gamma,
    ax: ai.x, ay: ai.y, az: ai.z,
    amag: +amag.toFixed(4),
    // derived: signed angular speed about the fitted swing axis (rad/s)
    s: +s.toFixed(5),
    axisX: +MEAS.axis[0].toFixed(4), axisY: +MEAS.axis[1].toFixed(4), axisZ: +MEAS.axis[2].toFixed(4),
  });
  const sp = Math.abs(s);
  if (sp > MEAS.peakSpeed) MEAS.peakSpeed = sp;
  if (amag > MEAS.aMax) MEAS.aMax = amag;
  if (amag < MEAS.aMin) MEAS.aMin = amag;
  // accumulate sums for this half-cycle's |a|-vs-ω² regression
  const w2 = s * s;
  MEAS.rw2 += w2; MEAS.rw4 += w2 * w2; MEAS.ra += amag; MEAS.raw2 += amag * w2; MEAS.rn++;

  // detect a turning point: sign flip of s, gated by a min amplitude so
  // noise near rest doesn't register as swings. Skip detection on the very
  // first sample after (re)start: recording begins mid-swing with s already
  // one sign, so prevS=0 would fabricate a bogus crossing at t≈0 and the
  // first recorded half-period would be a fragment (poisons the T average).
  const gate = Math.max(0.25, 0.35 * MEAS.peakSpeed);
  if (MEAS.primed) {
    if (MEAS.prevS <= 0 && s > 0 && MEAS.peakSpeed > gate) registerHalf(t);
    else if (MEAS.prevS >= 0 && s < 0 && MEAS.peakSpeed > gate) registerHalf(t);
  }
  MEAS.primed = true;
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
  MEAS.rw2 = MEAS.rw4 = MEAS.ra = MEAS.raw2 = MEAS.rn = 0;
}

function solveFromCycle(t) {
  const hp = MEAS.halfPeriods;
  if (hp.length < 2) return;
  const T = 2 * (hp.reduce((a, b) => a + b, 0) / hp.length);
  const wpeak = MEAS.peakSpeed;
  // Amplitude from the EXACT energy relation ωpeak² = (2·G_TRUE/L)(1−cosθmax).
  // It needs L, so seed L from the small-angle amplitude then refine both. This
  // is gyro-only (g-independent), so it doesn't couple to the accelerometer
  // calibration below, and it's exact at large angles unlike θ ≈ ωpeak·T/2π.
  const thSeed = (wpeak * T) / (2 * Math.PI);
  let L = periodToLength(T, thSeed);
  let thetaMax = thSeed;
  const cE = 1 - (L * wpeak * wpeak) / (2 * G_TRUE);
  if (cE > -1 && cE < 1) {
    thetaMax = Math.acos(cE);
    L = periodToLength(T, thetaMax); // refine L with the exact amplitude
  }
  thetaMax = Math.min(thetaMax, (89 * Math.PI) / 180);

  // least-squares fit of this half-cycle's |a| against ω²:
  //   |a|/k = g·cosθmax + (L/2 + r)·ω²  with k the accel scale error, so
  //   g_cal = intercept/cosθmax  (= k·g_true, self-calibrates the scale) and
  //   slope = k·(L/2 + r)  →  r = slope·(G_TRUE/g_cal) − L/2.
  // (cosθ and ω² are collinear via energy, so this 1-regressor form is the
  // well-conditioned one.) A failed/implausible fit records null so it can't
  // pollute the rolling medians.
  let gCyc = null, rCyc = null;
  const den = MEAS.rn * MEAS.rw4 - MEAS.rw2 * MEAS.rw2;
  const cthm = Math.cos(thetaMax);
  if (MEAS.rn >= 5 && Math.abs(den) > 1e-12 && cthm > 0.1) {
    const slope = (MEAS.rn * MEAS.raw2 - MEAS.rw2 * MEAS.ra) / den;
    const intercept = (MEAS.ra - slope * MEAS.rw2) / MEAS.rn;
    const gTry = intercept / cthm;
    // accept only a plausible device scale (within ~±8% of true g)
    if (gTry > 9.0 && gTry < 10.6) {
      gCyc = gTry;
      rCyc = slope * (G_TRUE / gTry) - L / 2; // undo the scale on the slope
    }
  }

  MEAS.T = T; MEAS.thetaMax = thetaMax; MEAS.L = L;
  MEAS.amps.push({ t, theta: thetaMax });
  if (MEAS.amps.length > 400) MEAS.amps.shift();
  MEAS.cycles.push({
    t: MEAS.t0 == null ? 0 : +(t - MEAS.t0).toFixed(3),
    T: +T.toFixed(4),
    thetaMaxDeg: +((thetaMax * 180) / Math.PI).toFixed(2),
    L_m: +L.toFixed(4),
    r_m: rCyc == null ? null : +rCyc.toFixed(4), // per-cycle radius (noisy; null = fit rejected)
    gCal: gCyc == null ? null : +gCyc.toFixed(4),
    wPeak: +wpeak.toFixed(4),
    aMax: +MEAS.aMax.toFixed(4),
    aMin: +(MEAS.aMin === Infinity ? 0 : MEAS.aMin).toFixed(4),
  });
  updateFromRecentCycles();
  MEAS.ready = true;
  renderMeasure();
}

// refresh the g / r / rOk aggregates from the last few solved cycles (shared
// by the live path and the retroactive end-trim so they can't disagree)
function updateFromRecentCycles() {
  const recent = MEAS.cycles.slice(-7);
  // calibrated gravity: rolling median over accepted fits
  const gs = recent.map((c) => c.gCal).filter((v) => v != null);
  if (gs.length) MEAS.g = median(gs);
  // radius: rolling median (noisiest output). Report only when it resolves to
  // a plausible length — small swings leave r below the accel noise floor.
  const rs = recent.map((c) => c.r_m).filter((v) => v != null);
  MEAS.r = rs.length ? median(rs) : 0;
  MEAS.rOk = rs.length >= 3 && MEAS.r > 0.15;
}

function median(a) {
  if (!a.length) return 0;
  const s = [...a].sort((x, y) => x - y);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

// Theil–Sen robust slope (median of pairwise slopes) — resists the odd
// restless cycle where the sitter injects energy
function theilSen(xs, ys) {
  const sl = [];
  for (let i = 0; i < xs.length; i++)
    for (let j = i + 1; j < xs.length; j++) {
      const dx = xs[j] - xs[i];
      if (Math.abs(dx) > 1e-9) sl.push((ys[j] - ys[i]) / dx);
    }
  return sl.length ? median(sl) : 0;
}

function fmtTime(s) {
  if (!isFinite(s)) return "∞";
  return s < 90 ? s.toFixed(0) + " s" : (s / 60).toFixed(1) + " min";
}

// Index where FREE decay begins in an amplitude series: the (last) peak —
// everything before it was powered ramp-up (user pushing). Then skip up to
// 3 leading readings that shed far more in one half-cycle than the rest of
// the series does: those are push-inflated measurements / slosh transients,
// not friction (a free pendulum can't lose 15% in one half-cycle and then
// settle to ~1%/cycle).
function freeDecayStart(A) {
  let m = 0;
  for (let i = 1; i < A.length; i++) if (A[i] >= A[m]) m = i;
  for (let extra = 0; extra < 3 && m < A.length - 2; extra++) {
    const decs = [];
    for (let i = m + 1; i < A.length - 1; i++) decs.push(Math.max(0, A[i] - A[i + 1]));
    const med = Math.max(median(decs), 0.002); // rad; floor for near-flat tails
    if (A[m] - A[m + 1] > 5 * med) m++;
    else break;
  }
  return m;
}

// Adaptive decay analysis over the per-cycle amplitude log.
//  • powered ramp-up (amplitude rising to a peak) is detected and ignored
//  • always: robust exponential → τ, half-life, Q
//  • enough decay: two-channel  dθ/dt = −γθ − c  (viscous + Coulomb)
//    → mechanism split + finite settling forecast
//  • no downward trend: report "steady / powered" instead of fitting noise
function analyzeDecay() {
  let a = MEAS.amps.filter((p) => p.theta > 0);
  if (a.length < 4) return { status: "insufficient", hint: "" };
  const start = freeDecayStart(a.map((p) => p.theta));
  const risingIgnoredSec = start > 0 ? a[start].t - a[0].t : 0;
  a = a.slice(start);
  if (a.length < 4) {
    return {
      status: "powered",
      risingIgnoredSec,
      hint: "Amplitude is still rising (being pushed?) — free decay hasn't started yet.",
    };
  }
  const t = a.map((p) => p.t), A = a.map((p) => p.theta); // seconds, radians
  const n = a.length, t0 = t[0];
  const x = t.map((v) => v - t0);
  const span = x[n - 1] - x[0] || 1;
  const T = MEAS.T || 1;

  // robust noise floor: residual about a robust linear detrend (MAD → σ)
  const sA = theilSen(x, A), medA = median(A), medX = median(x);
  const icA = medA - sA * medX;
  const resid = A.map((v, i) => Math.abs(v - (icA + sA * x[i])));
  const sigma = 1.4826 * median(resid) || 1e-4;

  // steady / powered guard: decay must clear the noise and point downward
  if (sA >= 0 || Math.abs(sA * span) < 2 * sigma) {
    return {
      status: "steady",
      sigma,
      risingIgnoredSec,
      hint: "Swing looks steady — not enough decay to estimate yet (or it's being powered).",
    };
  }

  // robust exponential envelope: slope of ln θ vs t = −1/τ
  const slope = theilSen(x, A.map((v) => Math.log(v)));
  const tau = slope < 0 ? -1 / slope : Infinity;
  const halfLife = Math.log(2) * tau;
  const Q = (Math.PI * tau) / T; // amplitude e-fold in radians of phase
  const dropFrac = (A[0] - A[n - 1]) / A[0];
  const out = { status: "ok", tau, halfLife, Q, sigma, dropFrac, nCycles: n, risingIgnoredSec };

  // upgrade: separate viscous (γ) from dry-friction (c) once the envelope
  // has enough curvature to constrain two parameters. Tentative from 15%
  // (kept honest by the "swing longer to firm up" note below 30%).
  if (dropFrac >= 0.15 && n >= 12) {
    const xs = [], ys = [];
    for (let i = 0; i < n - 1; i++) {
      const dt = t[i + 1] - t[i];
      if (dt > 0) { xs.push((A[i] + A[i + 1]) / 2); ys.push((A[i + 1] - A[i]) / dt); }
    }
    // channel model: dA/dt = −c − γ·A − β·A²  (Coulomb, linear-viscous,
    // quadratic/air drag). A and A² are collinear over a narrow amplitude
    // range, so the β channel is enabled only with ≥30% decay & ≥20 cycles;
    // below that it's the classic two-channel fit.
    const useQuad = dropFrac >= 0.3 && n >= 20;
    const { c, gamma, beta } = decayChannels(xs, ys, useQuad);
    const Anow = A[n - 1];
    const rates = [c, gamma * Anow, beta * Anow * Anow];
    const tot = rates[0] + rates[1] + rates[2];
    out.c = c; out.gamma = gamma; out.beta = beta;
    out.shares = tot > 0 ? { coulomb: rates[0] / tot, viscous: rates[1] / tot, quadratic: rates[2] / tot } : null;
    out.mechanism = mechanismText(out.shares);
    // settling forecast to a small target angle (numeric — no closed form
    // once the quadratic channel is in play)
    const targetDeg = 3;
    out.settleSec = settleTime(Anow, c, gamma, beta, (targetDeg * Math.PI) / 180);
    out.targetDeg = targetDeg;
    out.confident = dropFrac >= 0.3 && n >= 15;
  }
  out.hint = decayHintText(out);
  return out;
}

// Non-negative fit of dA/dt = −(c + γA + βA²): ordinary least squares on the
// active columns, then drop the worst-offending negative channel and refit
// (poor man's NNLS — fine for 3 parameters).
function decayChannels(xs, ys, useQuad) {
  let active = [true, true, useQuad]; // [const, A, A²]
  for (;;) {
    const cols = [];
    if (active[0]) cols.push(xs.map(() => 1));
    if (active[1]) cols.push(xs.slice());
    if (active[2]) cols.push(xs.map((v) => v * v));
    if (!cols.length) return { c: 0, gamma: 0, beta: 0 };
    const b = lsqSolve(cols, ys);
    if (!b) return { c: 0, gamma: 0, beta: 0 };
    // map solved coefficients back to channels; expected sign is negative
    const full = [0, 0, 0];
    let j = 0;
    for (let k = 0; k < 3; k++) if (active[k]) full[k] = b[j++];
    let worst = -1, worstVal = 0;
    for (let k = 0; k < 3; k++) {
      if (active[k] && -full[k] < worstVal) { worstVal = -full[k]; worst = k; }
    }
    if (worst < 0) return { c: -full[0] + 0, gamma: -full[1] + 0, beta: -full[2] + 0 }; // +0 kills -0
    active[worst] = false; // that channel wanted a negative rate — drop it
  }
}

// least squares for y ≈ Σ b_j·col_j via normal equations + Gaussian
// elimination (k ≤ 3). Returns null if singular.
function lsqSolve(cols, ys) {
  const k = cols.length, n = ys.length;
  const M = [], v = [];
  for (let i = 0; i < k; i++) {
    M.push(new Array(k).fill(0));
    let s = 0;
    for (let p = 0; p < n; p++) s += cols[i][p] * ys[p];
    v.push(s);
    for (let j = 0; j < k; j++) {
      let q = 0;
      for (let p = 0; p < n; p++) q += cols[i][p] * cols[j][p];
      M[i][j] = q;
    }
  }
  // gaussian elimination with partial pivoting
  for (let i = 0; i < k; i++) {
    let piv = i;
    for (let r = i + 1; r < k; r++) if (Math.abs(M[r][i]) > Math.abs(M[piv][i])) piv = r;
    if (Math.abs(M[piv][i]) < 1e-12) return null;
    [M[i], M[piv]] = [M[piv], M[i]];
    [v[i], v[piv]] = [v[piv], v[i]];
    for (let r = i + 1; r < k; r++) {
      const f = M[r][i] / M[i][i];
      for (let cIdx = i; cIdx < k; cIdx++) M[r][cIdx] -= f * M[i][cIdx];
      v[r] -= f * v[i];
    }
  }
  const b = new Array(k);
  for (let i = k - 1; i >= 0; i--) {
    let s = v[i];
    for (let j = i + 1; j < k; j++) s -= M[i][j] * b[j];
    b[i] = s / M[i][i];
  }
  return b;
}

function mechanismText(shares) {
  if (!shares) return "";
  const entries = [
    ["dry friction — it'll come to a full stop", shares.coulomb],
    ["linear drag (viscous) — steady exponential fade", shares.viscous],
    ["quadratic air drag — damping fades as it slows (τ ∝ 1/A)", shares.quadratic],
  ];
  entries.sort((a, b) => b[1] - a[1]);
  if (entries[0][1] > 0.5) return entries[0][0];
  return `mixed losses: ${(shares.coulomb * 100).toFixed(0)}% friction / ` +
    `${(shares.viscous * 100).toFixed(0)}% linear / ${(shares.quadratic * 100).toFixed(0)}% v²`;
}

// numeric settle-time for dA/dt = −(c + γA + βA²) down to `target` rad
function settleTime(A0, c, gamma, beta, target) {
  if (A0 <= target) return 0;
  let A = A0, t = 0;
  const dt = 0.05, cap = 7200;
  while (A > target && t < cap) {
    A -= (c + gamma * A + beta * A * A) * dt;
    t += dt;
  }
  return t >= cap ? Infinity : t;
}

function decayHintText(o) {
  if (o.status !== "ok") return o.hint || "";
  const parts = [`Q ≈ ${o.Q.toFixed(0)}, half-life ≈ ${fmtTime(o.halfLife)}`];
  if (o.risingIgnoredSec > 1) parts.push(`(ignored ${fmtTime(o.risingIgnoredSec)} of ramp-up)`);
  if (o.mechanism) {
    parts.push(o.mechanism);
    if (o.shares && !o.mechanism.startsWith("mixed"))
      parts.push(`(${(o.shares.quadratic * 100).toFixed(0)}% v² / ${(o.shares.viscous * 100).toFixed(0)}% linear / ${(o.shares.coulomb * 100).toFixed(0)}% friction)`);
    if (isFinite(o.settleSec)) parts.push(`≈ ${fmtTime(o.settleSec)} to settle below ${o.targetDeg}°`);
    if (!o.confident) parts.push("(swing longer to firm up the split)");
  } else {
    parts.push("(estimate — swing longer for a mechanism breakdown)");
  }
  return parts.join(" · ");
}

function renderMeasure() {
  $$("measReadouts").hidden = false;
  $$("mTrace").hidden = false;
  $$("measFoot").hidden = false;
  $$("mPeriod").textContent = MEAS.ready ? MEAS.T.toFixed(2) : "—";
  $$("mAngle").textContent = MEAS.ready ? ((MEAS.thetaMax * 180) / Math.PI).toFixed(0) : "—";
  $$("mLen").textContent = MEAS.ready ? fmtLen(MEAS.L) : "—";
  $$("mLenU").textContent = unit;
  $$("mRad").textContent = MEAS.rOk ? fmtLen(MEAS.r) : "—";
  $$("mRadU").textContent = unit;

  const parts = [];
  if (MEAS.rOk) {
    const ratio = MEAS.r / MEAS.L;
    if (ratio > 1.12) parts.push("Phone reads farther than the effective length → it's below your center of gravity.");
    else if (ratio < 0.88) parts.push("Phone reads shorter → it's above your center of gravity (nearer the pivot).");
    else parts.push("Phone radius ≈ effective length → it's near your center of gravity. Nice placement.");
    // r comes from a small centripetal signal; warn when the swing is gentle
    if ((MEAS.thetaMax * 180) / Math.PI < 15) parts.push("(placement is low-confidence at small swing angles)");
  } else if (MEAS.ready) {
    parts.push("Placement: swing bigger — the centripetal signal is below the accelerometer noise here.");
  }
  const dh = analyzeDecay().hint;
  if (dh) parts.push(dh);
  // leave the hint blank until there's a real reading — during warm-up the
  // top status line already narrates the countdown
  $$("mHint").textContent = parts.join(" · ");
  $$("applyBtn").disabled = !(MEAS.ready && MEAS.L > 0);
  $$("downloadBtn").disabled = MEAS.samples.length === 0;
}

// bundle raw samples + per-cycle analysis + final estimate as JSON
function downloadData() {
  if (MEAS.samples.length === 0) { mStatus("No data recorded yet.", true); return; }
  const s = MEAS.samples;
  const dur = s.length ? s[s.length - 1].t : 0;
  const data = {
    meta: {
      generatedAt: new Date().toISOString(),
      userAgent: navigator.userAgent,
      secureContext: window.isSecureContext,
      gTrue: G_TRUE, // used for the dynamics (period → length)
      gCalibrated: +MEAS.g.toFixed(4), // accelerometer scale, from |a|-vs-ω² fit
      skippedSec: MEAS.skipSec,
      trimmedEndSec: MEAS.trimApplied, // 0 on a mid-recording download
      units: "gyro deg/s, accel m/s², angle rad unless noted, lengths m",
      nSamples: s.length,
      durationSec: +dur.toFixed(2),
      avgRateHz: dur > 0 ? +(s.length / dur).toFixed(1) : null,
    },
    finalEstimate: {
      periodSec: +MEAS.T.toFixed(4),
      swingAngleDeg: +((MEAS.thetaMax * 180) / Math.PI).toFixed(2),
      effectiveLength_m: +MEAS.L.toFixed(4),
      phoneRadius_m: MEAS.rOk ? +MEAS.r.toFixed(4) : null,
      radiusOverLength: MEAS.rOk && MEAS.L > 0 ? +(MEAS.r / MEAS.L).toFixed(3) : null,
    },
    decay: analyzeDecay(),
    cycles: MEAS.cycles,
    samples: s,
  };
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
  a.href = url;
  a.download = "hammock-swing-" + stamp + ".json";
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  mStatus("Saved " + s.length + " samples over " + dur.toFixed(1) + " s to your downloads.");
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
  MEAS.rawShown = false;
  // warm-up: ignore the first N seconds so repositioning doesn't pollute the fit
  MEAS.skipSec = Math.max(0, Math.min(120, parseFloat($("skipInput").value) || 0));
  MEAS.warm = true;
  MEAS.startT = null;
  MEAS.wbuf = []; MEAS.axis = [1, 0, 0]; MEAS.sinceAxis = 0; MEAS.sbuf.fill(0);
  resetAnalysis(null);
  sizeMTrace();
  window.addEventListener("devicemotion", onMotion);
  mStatus("Permission OK — waiting for motion data…");
  const btn = $$("measBtn");
  btn.textContent = "Stop"; btn.classList.add("recording");
  $("skipInput").disabled = true;
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

// retroactively drop the last trimSec of the recording and re-derive the
// final estimate from what remains. Returns seconds actually trimmed, or
// -1 if that would leave too little to keep.
function applyTrim(trimSec) {
  if (!(trimSec > 0) || MEAS.samples.length === 0) return 0;
  const durRel = MEAS.samples[MEAS.samples.length - 1].t;
  const cutoff = durRel - trimSec;
  // need at least a couple of seconds of usable data left
  if (cutoff < 2) return -1;
  MEAS.samples = MEAS.samples.filter((s) => s.t <= cutoff);
  MEAS.cycles = MEAS.cycles.filter((c) => c.t <= cutoff);
  const cutAbs = (MEAS.t0 || 0) + cutoff; // amps carry absolute timestamps
  MEAS.amps = MEAS.amps.filter((a) => a.t <= cutAbs);
  const cyc = MEAS.cycles;
  if (cyc.length) {
    const last = cyc[cyc.length - 1];
    MEAS.T = last.T;
    MEAS.thetaMax = (last.thetaMaxDeg * Math.PI) / 180;
    MEAS.L = last.L_m;
    updateFromRecentCycles(); // re-derive g / r / rOk from the surviving cycles
    MEAS.ready = MEAS.L > 0;
  } else {
    MEAS.T = 0; MEAS.thetaMax = 0; MEAS.L = 0; MEAS.r = 0; MEAS.rOk = false; MEAS.ready = false;
  }
  return trimSec;
}

function stopMeasuring() {
  MEAS.on = false;
  window.removeEventListener("devicemotion", onMotion);
  MEAS.trimSec = Math.max(0, Math.min(120, parseFloat($("trimInput").value) || 0));
  const trimmed = applyTrim(MEAS.trimSec);
  if (trimmed < 0) MEAS.trimSec = 0; // couldn't trim — keep everything
  MEAS.trimApplied = trimmed > 0 ? MEAS.trimSec : 0;
  renderMeasure();
  const btn = $$("measBtn");
  btn.textContent = "Start measuring"; btn.classList.remove("recording");
  $("skipInput").disabled = false;
  $$("downloadBtn").disabled = MEAS.samples.length === 0;
  const trimNote =
    trimmed > 0 ? "  (dropped last " + trimmed + " s)"
    : trimmed < 0 ? "  (ignore-last too long — kept all data)"
    : "";
  mStatus(
    (MEAS.ready ? "Stopped. Tap “Apply to simulator” to replay this swing above." : "Stopped.") +
      (MEAS.samples.length ? "  " + MEAS.samples.length + " samples ready to download." : "") +
      trimNote,
  );
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
$$("downloadBtn").addEventListener("click", () => {
  try { downloadData(); }
  catch (e) { mStatus("Download failed: " + (e && (e.name + ": " + e.message)), true); }
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
