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

import sys, os, json, math, glob, statistics as st

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


def lsq_solve(cols, ys):
    """Least squares y ~ sum b_j * col_j via normal equations + Gaussian
    elimination (k <= 3). Returns None if singular."""
    k, n = len(cols), len(ys)
    M = [[sum(cols[i][p] * cols[j][p] for p in range(n)) for j in range(k)]
         for i in range(k)]
    v = [sum(cols[i][p] * ys[p] for p in range(n)) for i in range(k)]
    for i in range(k):
        piv = max(range(i, k), key=lambda r: abs(M[r][i]))
        if abs(M[piv][i]) < 1e-12:
            return None
        M[i], M[piv] = M[piv], M[i]
        v[i], v[piv] = v[piv], v[i]
        for r in range(i + 1, k):
            f = M[r][i] / M[i][i]
            for cc in range(i, k):
                M[r][cc] -= f * M[i][cc]
            v[r] -= f * v[i]
    b = [0.0] * k
    for i in range(k - 1, -1, -1):
        b[i] = (v[i] - sum(M[i][j] * b[j] for j in range(i + 1, k))) / M[i][i]
    return b


def decay_channels(xs, ys, use_quad):
    """Non-negative fit of dA/dt = -(c + gamma*A + beta*A^2): OLS on the
    active columns, then drop the worst-offending negative channel and refit
    (poor man's NNLS — fine for 3 parameters). Mirrors the app."""
    active = [True, True, use_quad]  # [const, A, A^2]
    while True:
        cols = []
        if active[0]:
            cols.append([1.0] * len(xs))
        if active[1]:
            cols.append(list(xs))
        if active[2]:
            cols.append([v * v for v in xs])
        if not cols:
            return 0.0, 0.0, 0.0
        b = lsq_solve(cols, ys)
        if b is None:
            return 0.0, 0.0, 0.0
        full = [0.0, 0.0, 0.0]
        j = 0
        for kk in range(3):
            if active[kk]:
                full[kk] = b[j]
                j += 1
        worst, worst_val = -1, 0.0
        for kk in range(3):
            if active[kk] and -full[kk] < worst_val:
                worst_val = -full[kk]
                worst = kk
        if worst < 0:
            return -full[0] + 0.0, -full[1] + 0.0, -full[2] + 0.0  # +0.0 kills -0.0
        active[worst] = False  # that channel wanted a negative rate


def settle_time(A0, c, gamma, beta, target):
    """Numeric settle-time for dA/dt = -(c + gamma*A + beta*A^2)."""
    if A0 <= target:
        return 0.0
    A, t, dt, cap = A0, 0.0, 0.05, 7200.0
    while A > target and t < cap:
        A -= (c + gamma * A + beta * A * A) * dt
        t += dt
    return float("inf") if t >= cap else t


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

    # channel model  dA/dt = -(c + gamma*A + beta*A^2): Coulomb, linear-
    # viscous, and quadratic/air drag. A and A^2 are collinear over a narrow
    # amplitude range, so the beta channel needs >=30% decay & >=20 cycles.
    xs = [(A[i] + A[i + 1]) / 2 for i in range(n - 1) if t[i + 1] > t[i]]
    ys = [(A[i + 1] - A[i]) / (t[i + 1] - t[i]) for i in range(n - 1) if t[i + 1] > t[i]]
    use_quad = drop >= 0.30 and n >= 20
    c, gamma, beta = decay_channels(xs, ys, use_quad)
    anow = A[-1]
    rates = [c, gamma * anow, beta * anow * anow]
    tot = sum(rates)
    shares = [r / tot for r in rates] if tot > 0 else [0, 0, 0]
    names = ["dry friction (full stop)", "linear drag (viscous)",
             "quadratic air drag (tau ~ 1/A)"]
    dom = max(range(3), key=lambda i: shares[i])
    mech = (names[dom] + " dominated") if shares[dom] > 0.5 else "mixed losses"
    print(f"  channels   : c={c:.5f} rad/s  gamma={gamma:.4f}/s  beta={beta:.4f}/rad/s"
          f"{'' if use_quad else '  (quadratic channel disabled: needs >=30% decay & >=20 cycles)'}")
    print(f"  loss split : {100 * shares[0]:.0f}% friction / {100 * shares[1]:.0f}% linear"
          f" / {100 * shares[2]:.0f}% v^2  -> {mech}")

    # settling forecast: numeric (no closed form with the quadratic channel)
    settle = settle_time(anow, c, gamma, beta, math.radians(3))
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

    gs, rs, mw2s = [], [], []
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
                mw2s.append(sum(xs) / len(xs))
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
        # elastic-suspension signature: per-cycle apparent g falls as the
        # swing grows (stretch amplifies the ω² part of |a|, deflating the
        # intercept). Flag it; quantifying needs the joint --stretch fit.
        if len(gs) >= 8 and _pearson(gs, mw2s) < -0.5:
            print("  note        : per-cycle g falls as amplitude grows — elastic-suspension"
                  " signature; run `analyze_data.py --stretch <files…>` to quantify")
    else:
        print("  (not enough free-decay data to calibrate g / recover r)")


def _pearson(a, b):
    n = len(a)
    ma, mb = st.mean(a), st.mean(b)
    num = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    da = math.sqrt(sum((v - ma) ** 2 for v in a))
    db = math.sqrt(sum((v - mb) ** 2 for v in b))
    return num / (da * db) if da > 0 and db > 0 else 0.0


# ------------------------------------------------------- stretch (joint fit)
def _stretch_rows(d):
    """Per free-decay half-cycle rows (intercept, cosθmax, slope, ⟨ω²⟩) for
    the elastic-suspension model. Looser per-cycle g gate than accel_check:
    stretch legitimately drags the apparent per-cycle g below 9."""
    S = d["samples"]
    L = d["finalEstimate"]["effectiveLength_m"]
    T = d["finalEstimate"]["periodSec"]
    C = d.get("cycles", [])
    cutoff = 0.0
    if len(C) >= 4:
        m = free_decay_start([math.radians(c["thetaMaxDeg"]) for c in C])
        if m > 0:
            cutoff = C[m]["t"]
    rows = []
    for seg in _split_cycles(S):
        if len(seg) < 5 or seg[0]["t"] < cutoff:
            continue
        xs = [p["s"] ** 2 for p in seg]
        ys = [p["amag"] for p in seg]
        slope, intercept = ols(xs, ys)
        wpk = max(abs(p["s"]) for p in seg)
        cc = 1 - (L * wpk ** 2) / (2 * G)
        thmax = math.acos(cc) if -1 < cc < 1 else wpk * T / (2 * math.pi)
        c = math.cos(thmax)
        if c > 0.1 and intercept > 0 and 8.0 < intercept / c < 11.5:
            rows.append((intercept, c, slope, sum(xs) / len(xs)))
    return rows


def stretch_report(paths):
    """Joint elastic-suspension fit across recordings, grouped per device.

    Physics: an elastic suspension breathes at 2× the swing frequency. Below
    the bounce resonance ω_b this amplifies the ω²-oscillating part of |a| by
    A_f = 1/(1 − (2ω₀/ω_b)²), inflating the per-cycle slope and deflating the
    intercept:  intercept_i = g_cal·cosθmax_i − ε·slope_i·⟨ω²⟩_i  with
    ε = (A_f−1)/A_f = (2ω₀/ω_b)²·A_f⁻¹·A_f = (2ω₀/ω_b)².

    One recording alone can't separate g_cal from ε (both regressors track
    amplitude), so we share g_cal across all recordings of a device (it's a
    device property) and give each recording its own ε (a rig-loading
    property). ε = 0 is the rigid model; the residual comparison says which
    model the data prefers. Returns a dict for testability.
    """
    header("STRETCH MODEL — joint fit per device (shared g_cal, per-recording ε)")
    groups = {}
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        ua = d.get("meta", {}).get("userAgent", "?")
        dev = "iPhone" if "iPhone" in ua else "Android" if "Android" in ua else ua[:24]
        groups.setdefault(dev, []).append((p, d))

    result = {}
    for dev, files in groups.items():
        entries = []
        for p, d in files:
            rows = _stretch_rows(d)
            C = d.get("cycles", [])
            amps = [c["thetaMaxDeg"] for c in C]
            m = free_decay_start([math.radians(a) for a in amps]) if len(amps) >= 4 else 0
            entries.append({
                "path": p,
                "rows": rows,
                "T": d["finalEstimate"]["periodSec"],
                "L": d["finalEstimate"]["effectiveLength_m"],
                "ampHi": amps[m] if amps else 0,
                "ampLo": amps[-1] if amps else 0,
            })
        entries = [e for e in entries if len(e["rows"]) >= 4]
        if not entries:
            print(f"  {dev}: no usable recordings")
            continue
        nrows = sum(len(e["rows"]) for e in entries)
        print(f"\n  device: {dev}  ({len(entries)} recordings, {nrows} half-cycles)")
        if len(entries) == 1:
            print("  (single recording — g_cal and ε are partially degenerate; add a"
                  " gentle-swing recording from the same phone to pin g_cal)")

        # joint LSQ: ic = g·cos − ε_f·(slope·⟨ω²⟩); clamp ε_f ≥ 0 by dropping
        y, cosCol, zCols, owner = [], [], [[] for _ in entries], []
        for fi, e in enumerate(entries):
            for (ic, cs, sl, mw2) in e["rows"]:
                y.append(ic)
                cosCol.append(cs)
                owner.append(fi)
                for fj in range(len(entries)):
                    zCols[fj].append(-sl * mw2 if fj == fi else 0.0)
        active = [True] * len(entries)
        for _ in range(len(entries) + 1):
            cols = [cosCol] + [zCols[f] for f in range(len(entries)) if active[f]]
            b = lsq_solve(cols, y)
            if b is None:
                print("  (singular fit)")
                break
            gcal = b[0]
            eps = [0.0] * len(entries)
            j = 1
            for f in range(len(entries)):
                if active[f]:
                    eps[f] = b[j]
                    j += 1
            bad = [f for f in range(len(entries)) if active[f] and eps[f] < 0]
            if not bad:
                break
            for f in bad:
                active[f] = False
        else:
            b = None
        if b is None:
            continue

        # residuals: stretch model vs rigid (all ε = 0)
        pred_st = [gcal * cosCol[i] - eps[owner[i]] * (-zCols[owner[i]][i])
                   for i in range(len(y))]
        # note: zCols holds −slope·⟨ω²⟩, so −zCols[...] = slope·⟨ω²⟩
        g_rigid = sum(y[i] * cosCol[i] for i in range(len(y))) / sum(c * c for c in cosCol)
        rms_st = math.sqrt(sum((y[i] - pred_st[i]) ** 2 for i in range(len(y))) / len(y))
        rms_ri = math.sqrt(sum((y[i] - g_rigid * cosCol[i]) ** 2 for i in range(len(y))) / len(y))
        print(f"  g_cal = {gcal:.3f} m/s^2 ({100 * (gcal - G) / G:+.1f}% vs 9.81)   "
              f"residual RMS: rigid {rms_ri:.3f} -> stretch {rms_st:.3f}")
        print("  recording                    amp(deg)   n   eps    bounce  sag     r rigid->corr   stretch")

        out_files = []
        for fi, e in enumerate(entries):
            ep = eps[fi]
            T, L = e["T"], e["L"]
            w0 = 2 * math.pi / T
            med_sl = median([r[2] for r in e["rows"]])
            r_rigid = med_sl * (G / gcal) - L / 2
            r_corr = med_sl * (1 - ep) * (G / gcal) - L / 2
            fb = sag = None
            label = "rigid"
            if ep > 1e-3:
                fb = 2 / (T * math.sqrt(ep))
                sag = G * ep / (4 * w0 * w0)
                label = ("low" if sag < 0.04 else "medium" if sag < 0.10 else "high")
                if ep > 0.8:
                    label += " (near resonance — unreliable)"
            if len(entries) == 1:
                label += " (unpinned)"
            name = os.path.basename(e["path"]).replace("hammock-swing-", "").replace(".json", "")
            print(f"  {name:28s} {e['ampHi']:3.0f}-{e['ampLo']:<3.0f}  {len(e['rows']):3d}"
                  f"  {ep:5.3f}  {('%4.1fHz' % fb) if fb else '  —  '}"
                  f"  {('%4.1fcm' % (100 * sag)) if sag else '  —  '}"
                  f"  {r_rigid:5.2f}->{r_corr:5.2f}     {label}")
            out_files.append({"path": e["path"], "T": T, "eps": ep, "fb": fb,
                              "sag": sag, "r_corr": r_corr, "r_rigid": r_rigid})
        result[dev] = {"gcal": gcal, "rms_rigid": rms_ri, "rms_stretch": rms_st,
                       "files": out_files}
    return result


# ------------------------------------------------------------------ main
def main():
    args = sys.argv[1:]
    if args and args[0] == "--stretch":
        paths = args[1:] or sorted(glob.glob("data/hammock-swing-*.json"))
        if not paths:
            sys.exit("no recordings found for --stretch")
        stretch_report(paths)
        return
    if args:
        path = args[0]
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
