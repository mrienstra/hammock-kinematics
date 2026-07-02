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
  // full recording for export
  samples: [], // per-event raw + derived
  cycles: [], // per-cycle solved estimates
  t0: null, // first event timestamp (s), for relative time
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

  MEAS.T = T; MEAS.thetaMax = thetaMax; MEAS.L = L;
  MEAS.amps.push({ t, theta: thetaMax });
  if (MEAS.amps.length > 400) MEAS.amps.shift();
  MEAS.cycles.push({
    t: MEAS.t0 == null ? 0 : +(t - MEAS.t0).toFixed(3),
    T: +T.toFixed(4),
    thetaMaxDeg: +((thetaMax * 180) / Math.PI).toFixed(2),
    L_m: +L.toFixed(4),
    r_m: +r.toFixed(4), // raw per-cycle radius (noisy)
    wPeak: +wpeak.toFixed(4),
    aMax: +MEAS.aMax.toFixed(4),
    aMin: +(MEAS.aMin === Infinity ? 0 : MEAS.aMin).toFixed(4),
  });
  // r is the noisiest output — report a rolling median of recent cycles
  // instead of the latest single (division-amplified) value
  MEAS.r = median(MEAS.cycles.slice(-7).map((c) => c.r_m).filter((v) => v > 0));
  MEAS.ready = true;
  renderMeasure();
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

// Adaptive decay analysis over the per-cycle amplitude log.
//  • always: robust exponential → τ, half-life, Q
//  • enough decay: two-channel  dθ/dt = −γθ − c  (viscous + Coulomb)
//    → mechanism split + finite settling forecast
//  • no downward trend: report "steady / powered" instead of fitting noise
function analyzeDecay() {
  const a = MEAS.amps.filter((p) => p.theta > 0);
  if (a.length < 4) return { status: "insufficient", hint: "" };
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
      hint: "Swing looks steady — not enough decay to estimate yet (or it's being powered).",
    };
  }

  // robust exponential envelope: slope of ln θ vs t = −1/τ
  const slope = theilSen(x, A.map((v) => Math.log(v)));
  const tau = slope < 0 ? -1 / slope : Infinity;
  const halfLife = Math.log(2) * tau;
  const Q = (Math.PI * tau) / T; // amplitude e-fold in radians of phase
  const dropFrac = (A[0] - A[n - 1]) / A[0];
  const out = { status: "ok", tau, halfLife, Q, sigma, dropFrac, nCycles: n };

  // upgrade: separate viscous (γ) from dry-friction (c) once the envelope
  // has enough curvature to constrain two parameters
  if (dropFrac >= 0.2 && n >= 12) {
    const xs = [], ys = [];
    for (let i = 0; i < n - 1; i++) {
      const dt = t[i + 1] - t[i];
      if (dt > 0) { xs.push((A[i] + A[i + 1]) / 2); ys.push((A[i + 1] - A[i]) / dt); }
    }
    const m = xs.length;
    let sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (let i = 0; i < m; i++) { sx += xs[i]; sy += ys[i]; sxx += xs[i] * xs[i]; sxy += xs[i] * ys[i]; }
    const den = m * sxx - sx * sx;
    let gamma = 0, c = 0;
    if (den !== 0) {
      const sl2 = (m * sxy - sx * sy) / den; // = −γ
      gamma = Math.max(0, -sl2);
      c = Math.max(0, -((sy - sl2 * sx) / m)); // intercept = −c
    }
    const Anow = A[n - 1];
    const viscRate = gamma * Anow, coulRate = c, tot = viscRate + coulRate;
    const coulombShare = tot > 0 ? coulRate / tot : 0;
    out.gamma = gamma; out.c = c; out.coulombShare = coulombShare;
    out.mechanism =
      coulombShare > 0.66 ? "dry-friction dominated — it'll come to a full stop"
      : coulombShare < 0.33 ? "viscous/air dominated — long asymptotic tail"
      : "mixed dry-friction + viscous";
    // settling forecast to a small target angle
    const targetDeg = 3, target = (targetDeg * Math.PI) / 180;
    let settle = Infinity;
    if (gamma > 1e-6) {
      const K = Anow + c / gamma, val = (target + c / gamma) / K;
      settle = val > 0 && val < 1 ? Math.log(1 / val) / gamma : val >= 1 ? 0 : Infinity;
    } else if (c > 0) settle = Math.max(0, (Anow - target) / c);
    out.settleSec = settle; out.targetDeg = targetDeg;
    out.confident = dropFrac >= 0.3 && n >= 15;
  }
  out.hint = decayHintText(out);
  return out;
}

function decayHintText(o) {
  if (o.status !== "ok") return o.hint || "";
  const parts = [`Q ≈ ${o.Q.toFixed(0)}, half-life ≈ ${fmtTime(o.halfLife)}`];
  if (o.mechanism) {
    parts.push(o.mechanism);
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
  const dh = analyzeDecay().hint;
  if (dh) parts.push(dh);
  $$("mHint").textContent = parts.join(" · ") || "Swing for a few cycles for a reading…";
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
      g: MEAS.g,
      units: "gyro deg/s, accel m/s², angle rad unless noted, lengths m",
      nSamples: s.length,
      durationSec: +dur.toFixed(2),
      avgRateHz: dur > 0 ? +(s.length / dur).toFixed(1) : null,
    },
    finalEstimate: {
      periodSec: +MEAS.T.toFixed(4),
      swingAngleDeg: +((MEAS.thetaMax * 180) / Math.PI).toFixed(2),
      effectiveLength_m: +MEAS.L.toFixed(4),
      phoneRadius_m: +MEAS.r.toFixed(4),
      radiusOverLength: MEAS.L > 0 ? +(MEAS.r / MEAS.L).toFixed(3) : null,
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
  MEAS.samples = []; MEAS.cycles = []; MEAS.t0 = null;
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
  $$("downloadBtn").disabled = MEAS.samples.length === 0;
  mStatus(
    (MEAS.ready ? "Stopped. Tap “Apply to simulator” to replay this swing above." : "Stopped.") +
      (MEAS.samples.length ? "  " + MEAS.samples.length + " samples ready to download." : ""),
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
