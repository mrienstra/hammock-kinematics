#!/usr/bin/env python3
"""Analyze a hammock-swing recording exported by the Measure panel.

Usage:
    python3 analyze_data.py [path/to/recording.json]

With no argument it picks the newest data/hammock-swing-*.json file.

Everything here is an INDEPENDENT re-derivation from the raw samples, meant
to cross-check the numbers the web app reports (finalEstimate / decay). It
mirrors the app's math (median, Theil–Sen, the primed zero-cross guard, the
two-channel γ+c decay fit) so the two should agree; where they don't, the raw
samples are the source of truth. Standard library only.
"""

import sys, json, math, glob, statistics as st

G = 9.81


# ------------------------------------------------------------------ helpers
def median(a):
    a = sorted(a)
    n = len(a)
    if n == 0:
        return 0.0
    m = n // 2
    return a[m] if n % 2 else (a[m - 1] + a[m]) / 2


def theil_sen(xs, ys):
    """Robust slope: median of pairwise slopes (matches the app)."""
    sl = [(ys[j] - ys[i]) / (xs[j] - xs[i])
          for i in range(len(xs)) for j in range(i + 1, len(xs))
          if abs(xs[j] - xs[i]) > 1e-9]
    return median(sl) if sl else 0.0


def ols(xs, ys):
    """Ordinary least squares → (slope, intercept)."""
    n = len(xs)
    sx, sy = sum(xs), sum(ys)
    sxx = sum(v * v for v in xs)
    sxy = sum(xs[i] * ys[i] for i in range(n))
    den = n * sxx - sx * sx
    if den == 0:
        return 0.0, (sy / n if n else 0.0)
    slope = (n * sxy - sx * sy) / den
    return slope, (sy - slope * sx) / n


def fmt_time(s):
    if not math.isfinite(s):
        return "∞"
    return f"{s:.0f} s" if s < 90 else f"{s / 60:.1f} min"


def header(title):
    print("\n" + title)
    print("-" * len(title))


# ------------------------------------------------------------------ sections
def report_summary(d):
    m, fe = d["meta"], d["finalEstimate"]
    header("SUMMARY")
    print(f"  file generated : {m.get('generatedAt', '?')}")
    print(f"  device         : {m.get('userAgent', '?')[:60]}")
    print(f"  duration       : {m.get('durationSec')} s, {m.get('nSamples')} "
          f"samples @ {m.get('avgRateHz')} Hz")
    print(f"  warm-up / trim : ignored first {m.get('skippedSec')} s, "
          f"last {m.get('trimmedEndSec')} s")
    print(f"  app estimate   : T={fe['periodSec']} s  angle={fe['swingAngleDeg']}°  "
          f"L={fe['effectiveLength_m']} m  r={fe['phoneRadius_m']} m  "
          f"(r/L={fe.get('radiusOverLength')})")
    dec = d.get("decay", {})
    if dec:
        print(f"  app decay      : {dec.get('hint', '')}")


def report_cycles(d):
    C = d["cycles"]
    header(f"PER-CYCLE ({len(C)} half-cycles)")
    num = lambda v, fmt: (fmt % v) if isinstance(v, (int, float)) else "  ---"
    print("   t      T     angle   L_m    r_m    gCal   wPeak")
    for c in C:
        print(f"  {c['t']:5.2f}  {c['T']:5.3f}  {c['thetaMaxDeg']:5.1f}  "
              f"{c['L_m']:5.3f}  {num(c.get('r_m'), '%5.3f')}  "
              f"{num(c.get('gCal'), '%5.3f')}  {c['wPeak']:5.3f}")
    L = [c["L_m"] for c in C]
    r = [c.get("r_m") for c in C if isinstance(c.get("r_m"), (int, float))]
    print(f"\n  L: median={median(L):.3f} m  sd={st.pstdev(L):.4f}   "
          f"(large sd => startup ramp / instability)")
    if r:
        print(f"  r: median={median(r):.3f} m  sd={st.pstdev(r):.3f}   "
              f"(noisiest output; app reports a rolling median)")


def verify_raw(d):
    """Re-derive period/axis/planarity straight from the raw sample stream."""
    S = d["samples"]
    header("INDEPENDENT RE-DERIVATION (from raw samples)")
    if not S:
        print("  no samples")
        return

    # period via gated zero-crossings of s, replicating the app's primed guard
    peak, prev, primed, last = 0.0, 0.0, False, None
    hps = []
    for x in S:
        s = x["s"]
        peak = max(peak, abs(s))
        gate = max(0.25, 0.35 * peak)
        if primed and ((prev <= 0 < s) or (prev >= 0 > s)) and peak > gate:
            if last is not None:
                half = x["t"] - last
                if 0.15 < half < 8:
                    hps.append(half)
            last = x["t"]
            peak = 0.0
        primed = True
        prev = s
    if hps:
        T = 2 * st.mean(hps)
        print(f"  zero-cross period : T={T:.3f} s from {len(hps)} half-periods "
              f"(sd {st.pstdev(hps) * 1000:.0f} ms)")
        print(f"  first 3 half-per. : {[round(h, 3) for h in hps[:3]]}  "
              f"(a short leading value => startup fragment)")
        print(f"  small-angle L     : {G * (T / (2 * math.pi)) ** 2:.3f} m")

    # axis stability + planarity (skip the first ~1 s of warm-up settling)
    D = math.pi / 180
    tail = S[60:] if len(S) > 120 else S
    ax = [(x["axisX"], x["axisY"], x["axisZ"]) for x in tail]
    for i, nm in enumerate("XYZ"):
        vals = [a[i] for a in ax]
        print(f"  axis {nm}            : mean={st.mean(vals):+.3f}  sd={st.pstdev(vals):.3f}")
    onax = []
    for x in tail:
        w = (x["rrBeta"] * D, x["rrGamma"] * D, x["rrAlpha"] * D)
        tot = math.hypot(*w)
        if tot > 0.2:
            n = (x["axisX"], x["axisY"], x["axisZ"])
            onax.append(abs(sum(w[i] * n[i] for i in range(3))) / tot)
    if onax:
        print(f"  planarity         : on-axis fraction {st.mean(onax):.3f} "
              f"(1.0 = pure planar swing, no twist)")


def amplitude_noise(C):
    """1-sigma per-cycle amplitude scatter about a robust linear trend (deg)."""
    t = [c["t"] for c in C]
    A = [c["thetaMaxDeg"] for c in C]
    sl = theil_sen(t, A)
    ic = median(A) - sl * median(t)
    resid = [abs(A[i] - (ic + sl * t[i])) for i in range(len(A))]
    return 1.4826 * median(resid)


def free_decay_start(A):
    """Index where free decay begins (mirrors the app's freeDecayStart).

    The (last) amplitude peak — everything before it was powered ramp-up —
    plus up to 3 extra skips for push-inflated leading readings that shed far
    more in one half-cycle than the rest of the series does (push residue /
    slosh transients, not friction). A in radians.
    """
    m = 0
    for i in range(1, len(A)):
        if A[i] >= A[m]:
            m = i
    for _ in range(3):
        if m >= len(A) - 2:
            break
        decs = [max(0.0, A[i] - A[i + 1]) for i in range(m + 1, len(A) - 1)]
        med = max(median(decs), 0.002)
        if A[m] - A[m + 1] > 5 * med:
            m += 1
        else:
            break
    return m


def fit_decay(d):
    """Full decay model comparison: exponential, Coulomb-linear, quadratic-
    hyperbolic, plus the combined viscous+Coulomb (gamma, c) split."""
    C = d["cycles"]
    header("DECAY ANALYSIS")
    if len(C) < 4:
        print("  too few cycles")
        return
    T = d["finalEstimate"]["periodSec"] or (2 * (C[1]["t"] - C[0]["t"]))
    t = [c["t"] for c in C]
    A = [math.radians(c["thetaMaxDeg"]) for c in C]
    m = free_decay_start(A)
    if m > 0:
        print(f"  push phase : ignored first {m} half-cycles ({t[m] - t[0]:.1f} s) — "
              f"amplitude rising / push-inflated")
        t, A, C = t[m:], A[m:], C[m:]
    if len(A) < 4:
        print("  amplitude still rising at the end — no free decay to fit")
        return
    n = len(A)
    x0 = t[0]
    x = [ti - x0 for ti in t]
    drop = (A[0] - A[-1]) / A[0]
    ups = sum(1 for i in range(1, n) if A[i] > A[i - 1])
    sigma_deg = amplitude_noise(C)
    print(f"  amplitude  : {math.degrees(A[0]):.1f} deg -> {math.degrees(A[-1]):.1f} deg  "
          f"({100 * drop:.1f}% over {x[-1]:.0f} s)")
    print(f"  noise/monotonicity : per-cycle sigma={sigma_deg:.2f} deg, "
          f"{ups}/{n - 1} up-ticks "
          f"({'clean decay' if ups <= n // 4 else 'noisy / possibly powered'})")

    # robust exponential envelope
    slope = theil_sen(x, [math.log(a) for a in A])
    tau = -1 / slope if slope < 0 else float("inf")
    Q = math.pi * tau / T
    print(f"  exponential: tau={tau:.0f} s  half-life={fmt_time(math.log(2) * tau)}  "
          f"Q~{Q:.0f}")

    # shape comparison via RMS residual (which envelope fits best?)
    def rms(pred):
        return math.sqrt(sum((A[i] - pred(x[i])) ** 2 for i in range(n)) / n)

    b, a0 = ols(x, [math.log(a) for a in A])
    r_exp = rms(lambda u: math.exp(a0 + b * u))
    sl_lin, ic_lin = ols(x, A)
    r_lin = rms(lambda u: ic_lin + sl_lin * u)
    sl_q, ic_q = ols(x, [1 / a for a in A])           # 1/A linear => hyperbolic
    r_quad = rms(lambda u: 1 / (ic_q + sl_q * u) if (ic_q + sl_q * u) > 0 else 0)
    best = min([("exponential/viscous", r_exp), ("linear/Coulomb", r_lin),
                ("hyperbolic/quadratic-drag", r_quad)], key=lambda p: p[1])
    print(f"  shape RMS  : exp={r_exp:.5f}  linear={r_lin:.5f}  hyperbolic={r_quad:.5f}"
          f"  -> best fit: {best[0]}")

    # combined two-channel  dA/dt = -gamma*A - c
    xs = [(A[i] + A[i + 1]) / 2 for i in range(n - 1) if t[i + 1] > t[i]]
    ys = [(A[i + 1] - A[i]) / (t[i + 1] - t[i]) for i in range(n - 1) if t[i + 1] > t[i]]
    sl2, ic2 = ols(xs, ys)
    gamma, c = max(0.0, -sl2), max(0.0, -ic2)
    anow = A[-1]
    visc, coul = gamma * anow, c
    tot = visc + coul
    share = coul / tot if tot > 0 else 0.0
    mech = ("dry-friction dominated (comes to a full stop)" if share > 0.66
            else "viscous/air dominated (asymptotic tail)" if share < 0.33
            else "mixed friction + viscous")
    print(f"  two-channel: gamma={gamma:.4f}/s  c={c:.5f} rad/s  "
          f"Coulomb share={100 * share:.0f}%  -> {mech}")

    # settling forecast from the combined model
    target = math.radians(3)
    if gamma > 1e-6:
        K = anow + c / gamma
        val = (target + c / gamma) / K
        settle = math.log(1 / val) / gamma if 0 < val < 1 else 0.0
    else:
        settle = (anow - target) / c if c > 0 else float("inf")
    conf = ("confident" if (drop >= 0.30 and n >= 15)
            else "tentative - swing longer (>30% decay) to firm up" if drop >= 0.15
            else "too little decay to trust the mechanism split")
    print(f"  forecast   : ~{fmt_time(settle)} to settle below 3 deg   [{conf}]")


def _split_cycles(S):
    """Split the sample stream into half-cycles at gated zero-crossings of s,
    replicating the app's primed guard."""
    segs, cur, peak, prev, primed = [], [], 0.0, 0.0, False
    for x in S:
        v = x["s"]
        peak = max(peak, abs(v))
        gate = max(0.25, 0.35 * peak)
        if primed and ((prev <= 0 < v) or (prev >= 0 > v)) and peak > gate:
            if cur:
                segs.append(cur)
            cur, peak = [], 0.0
        primed = True
        prev = v
        cur.append(x)
    if cur:
        segs.append(cur)
    return segs


def accel_check(d):
    """Self-calibrate g and recover phone radius r from the felt-force model.

    For a pendulum, energy makes cosθ = cosθmax + (L/2g)·ω², so within a
    constant-amplitude half-cycle  |a| = g·cosθmax + (L/2 + r)·ω². Regressing
    |a| on ω² (well-conditioned) gives g = intercept/cosθmax and r = slope−L/2.
    g calibrates the accelerometer scale; L (from the period) supplies the L/2.
    """
    S = d["samples"]
    header("ACCELEROMETER: g self-calibration + phone radius")
    amag = [x["amag"] for x in S if "amag" in x]
    if not amag:
        print("  no accel samples")
        return
    L = d["finalEstimate"]["effectiveLength_m"]
    T = d["finalEstimate"]["periodSec"]
    print(f"  measured |a|: mean={st.mean(amag):.3f}  min={min(amag):.3f}  "
          f"max={max(amag):.3f}  sd={st.pstdev(amag):.3f} m/s^2")

    # pushing violates the free-pendulum force model (an external force adds
    # to the specific force), so calibrate only on the free-decay phase
    cutoff = 0.0
    C = d.get("cycles", [])
    if len(C) >= 4:
        m = free_decay_start([math.radians(c["thetaMaxDeg"]) for c in C])
        if m > 0:
            cutoff = C[m]["t"]
            print(f"  push filter : fitting only samples after t={cutoff:.1f} s (free decay)")

    gs, rs = [], []
    for seg in _split_cycles(S):
        if len(seg) < 5 or seg[0]["t"] < cutoff:
            continue
        xs = [p["s"] ** 2 for p in seg]
        ys = [p["amag"] for p in seg]
        slope, intercept = ols(xs, ys)
        wpk = max(abs(p["s"]) for p in seg)
        # exact (g-independent) amplitude from energy: w^2 = (2G/L)(1-cos th)
        cc = 1 - (L * wpk ** 2) / (2 * G)
        thmax = math.acos(cc) if -1 < cc < 1 else wpk * T / (2 * math.pi)
        c = math.cos(thmax)
        if c > 0.1:
            g_cyc = intercept / c
            if 9.0 < g_cyc < 10.6:  # plausible device scale only
                gs.append(g_cyc)
                # slope = (g_cyc/G)*(L/2 + r): undo the accel scale before
                # subtracting L/2, else r comes out ~(g_cyc/G) too small
                rs.append(slope * (G / g_cyc) - L / 2)
    if gs:
        gmed = median(gs)
        rmed = median(rs)
        print(f"  calibrated g : {gmed:.3f} m/s^2  (sd {st.pstdev(gs):.3f})  "
              f"{'— device reads %+.1f%% vs 9.81' % (100 * (gmed - 9.81) / 9.81)}")
        # centripetal signal vs noise, to judge whether r is trustworthy
        wpk = max(abs(x["s"]) for x in S)
        snr = max(0.0, rmed) * wpk ** 2 / (st.pstdev(amag) or 1e-6)
        ok = rmed > 0.15 and snr > 2
        verdict = (f"r/L={rmed / L:.2f} " +
                   ("(below CG)" if rmed / L > 1.12 else
                    "(above CG / near pivot)" if rmed / L < 0.88 else "(near CG)")) \
            if ok else "UNRESOLVED — swing bigger (centripetal signal ~ noise)"
        print(f"  phone radius : r={rmed:.3f} m  (sd {st.pstdev(rs):.3f}, "
              f"centripetal SNR~{snr:.1f})  {verdict}")
    else:
        print("  (not enough free-decay data to calibrate g / recover r)")


# ------------------------------------------------------------------ main
def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        files = sorted(glob.glob("data/hammock-swing-*.json"))
        if not files:
            sys.exit("no data/hammock-swing-*.json found; pass a path explicitly")
        path = files[-1]
    print(f"Analyzing: {path}")
    with open(path) as f:
        d = json.load(f)
    report_summary(d)
    report_cycles(d)
    verify_raw(d)
    fit_decay(d)
    accel_check(d)


if __name__ == "__main__":
    main()
