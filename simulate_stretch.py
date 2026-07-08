#!/usr/bin/env python3
"""Time-domain validation of the elastic-suspension (stretch) model.

analyze_data.py --stretch infers stretch from how the ⟨|a|⟩-vs-⟨ω²⟩
REGRESSION changes across recordings (a cycle-aggregate method). This
script tests the same hypothesis a different way: actually simulate the
radial spring-mass dynamics, forced by the REAL measured swing, and compare
the predicted |a|(t) waveform against what was actually recorded — sample
by sample, in the time domain. If a physically plausible bounce frequency
explains the raw waveform much better than the rigid model, AND that
frequency agrees with what the amplitude-based fit found independently,
that's two very different methods triangulating on the same physical
constant — good evidence the "stretch" story is real, not a fitting
artifact of one particular method.

Key simplification: the swing itself (theta) is NOT re-simulated — that
would mean reproducing the push and the exact decay, which is unnecessary
work and a source of avoidable error, since we already MEASURED it. Instead
this script reconstructs cos(theta(t)) directly from the recorded gyro
speed via the exact pendulum energy relation

    cos(theta) = cos(thetamax_i) + (L / 2g) * omega(t)^2

using each half-cycle's already-fitted amplitude thetamax_i (same method
used throughout analyze_data.py). Only the radial (bounce) degree of
freedom r(t) is actually integrated forward in time, driven by that
reconstructed cos(theta(t)) and the real omega(t). Because only cos(theta)
is needed (never theta's sign), there's no sign ambiguity to resolve and no
integration drift to accumulate — a meaningfully more honest test than
re-simulating everything from scratch.

Model (single radial spring DOF, bob = phone, equilibrium anchored at the
fitted period-length L — see the docstring note in fit_bounce() for why):
    r'' = r*omega^2 + g*cos(theta) - wb^2*(r - L0) - 2*zeta*wb*r'
    amag_predicted = k_scale * |wb^2*(r - L0) + 2*zeta*wb*r'|
(the spring's contact force IS what an accelerometer riding the bob reads;
matches the model already validated in synthetic_check.py's spring case.)

Run:  python3 simulate_stretch.py [path/to/recording.json]
      python3 simulate_stretch.py --pushcheck [path/to/recording.json]
"""
import sys, os, json, math, glob
import analyze_data as ad

G = ad.G


def reconstruct_drive(d):
    """Per-sample (t, omega, cos_theta, amag) for every sample in a "good"
    (len>=5) half-cycle segment within the free-decay window, sorted by t."""
    S = d["samples"]
    L = d["finalEstimate"]["effectiveLength_m"]
    C = d.get("cycles", [])
    cutoff = 0.0
    if len(C) >= 4:
        m = ad.free_decay_start([math.radians(c["thetaMaxDeg"]) for c in C])
        if m > 0:
            cutoff = C[m]["t"]
    rows = []
    for seg in ad._split_cycles(S):
        if len(seg) < 5 or seg[0]["t"] < cutoff:
            continue
        wpk = max(abs(p["s"]) for p in seg)
        cos_thmax = 1 - (L * wpk ** 2) / (2 * G)
        if not (-1 < cos_thmax < 1):
            continue
        for p in seg:
            om = p["s"]
            cth = cos_thmax + (L / (2 * G)) * om * om
            cth = max(-1.0, min(1.0, cth))
            rows.append((p["t"], om, cth, p["amag"]))
    rows.sort(key=lambda r: r[0])
    return rows, L, cutoff


def simulate_bounce(rows, L, k_scale, fb, zeta):
    """Drive the radial spring ODE with the REAL (t, omega, cos_theta) rows;
    return predicted |a| at each row's timestamp.

    Equilibrium is anchored at r_eq = L (the fitted pendulum length), not at
    the separately-fitted rigid phone radius: L is what determines the
    swing period used to reconstruct cos(theta(t)) in the first place, so
    it's the self-consistent choice for "where the effective bob sits."
    Phone PLACEMENT relative to that (the r vs L distinction used
    elsewhere) is a separate geometric effect this single-DOF toy model
    doesn't try to capture — this script is testing bounce DYNAMICS, not
    re-deriving placement.
    """
    wb = 2 * math.pi * fb
    L0 = L - G / (wb * wb)
    r, vr = L, 0.0  # start at rest at the anchor; transient dies out via zeta
    t_prev = rows[0][0]
    out = []
    # interpolate the forcing linearly between consecutive real samples,
    # sub-stepped finely enough to resolve the bounce frequency (>=20 steps/cycle)
    substeps = max(4, int(math.ceil((1.0 / fb) / 0.01)))
    for i, (t, om, cth, amag) in enumerate(rows):
        if i == 0:
            out.append(k_scale * abs(wb * wb * (r - L0) + 2 * zeta * wb * vr))
            continue
        om0, cth0 = rows[i - 1][1], rows[i - 1][2]
        dt_total = t - t_prev
        if dt_total <= 0 or dt_total > 0.5:  # gap (trimmed/discontinuous) — reset softly
            t_prev = t
            out.append(k_scale * abs(wb * wb * (r - L0) + 2 * zeta * wb * vr))
            continue
        n = max(1, int(math.ceil(dt_total / ((1.0 / fb) / 20))))
        dt = dt_total / n
        for k in range(n):
            frac = (k + 0.5) / n
            omi = om0 + (om - om0) * frac
            cthi = cth0 + (cth - cth0) * frac

            def f(rr, vv):
                return vv, rr * omi * omi + G * cthi - wb * wb * (rr - L0) - 2 * zeta * wb * vv

            k1r, k1v = f(r, vr)
            k2r, k2v = f(r + .5 * dt * k1r, vr + .5 * dt * k1v)
            k3r, k3v = f(r + .5 * dt * k2r, vr + .5 * dt * k2v)
            k4r, k4v = f(r + dt * k3r, vr + dt * k3v)
            r += dt / 6 * (k1r + 2 * k2r + 2 * k3r + k4r)
            vr += dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
        out.append(k_scale * abs(wb * wb * (r - L0) + 2 * zeta * wb * vr))
        t_prev = t
    return out


def rms(a, b, skip=0):
    """RMS over indices [skip:] — lets the ODE's arbitrary rest-start
    transient (no real initial radial velocity/position is known, since the
    push itself isn't simulated) settle before it's scored."""
    n = len(a)
    if n - skip < 10:
        skip = 0
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(skip, n)) / (n - skip))


def fit_bounce(path, fb_range=None, zeta_range=None, g_cal_override=None):
    """Grid-search (bounce_hz, zeta) to minimize the |a|(t) residual against
    the REAL recording, then compare against the rigid model. Returns a
    result dict; prints a report as a side effect.

    g_cal_override: use this for the bounce model's scale (k_scale) instead
    of the single-file fit_g_and_r() g_cal. Tried using the properly-
    anchored multi-recording value here (analyze_data.py --stretch) on the
    theory that fit_g_and_r's g_cal is itself stretch-biased — but the gap
    between the two (0.2-0.7 m/s^2) turned out to be AS LARGE AS the whole
    |a|(t) RMS residual being fit (0.09-0.46 m/s^2), so it swamped the
    within-cycle oscillation-SHAPE signal this function is actually trying
    to isolate, and made every comparison worse, not better. Left available
    for inspection (main() can wire it back in) but not used by default —
    the honest, apples-to-apples comparison keeps calibration source
    IDENTICAL between the bounce and rigid predictions, so what's left is
    purely a difference in oscillation shape.
    """
    with open(path) as f:
        d = json.load(f)
    ad.header(f"TIME-DOMAIN BOUNCE FIT: {os.path.basename(path)}")

    calib = ad.fit_g_and_r(d)
    if not calib:
        print("  not enough free-decay data to calibrate — skipping")
        return None
    g_cal, r_rigid = calib["gcal"], calib["r"]
    g_bounce = g_cal_override if g_cal_override else g_cal
    k_scale = g_bounce / G

    rows, L, cutoff = reconstruct_drive(d)
    if len(rows) < 50:
        print(f"  only {len(rows)} usable samples — too few for a waveform fit")
        return None
    real = [r[3] for r in rows]
    print(f"  drive        : {len(rows)} samples, L={L:.3f} m, "
          f"g_cal(rigid,single-file)={g_cal:.3f}"
          + (f", g_cal(anchored,used)={g_bounce:.3f}" if g_cal_override else "")
          + f", r_rigid={r_rigid:.3f} m")

    # The ODE starts at rest (r=L, vr=0) since the true radial state at the
    # free-decay cutoff isn't known (the push itself isn't simulated) — that
    # arbitrary start produces a settling transient. Give it ~2.5s (several
    # bounce periods at the frequencies seen so far) before scoring RMS, so
    # a short recording isn't unfairly dominated by an unmodeled startup
    # mismatch. The grid search below still integrates through the full
    # window (needed for correct dynamics); only the SCORING window shifts.
    t_start = rows[0][0]
    warmup_n = sum(1 for r_ in rows if r_[0] - t_start < 2.5)

    # rigid-model prediction, for baseline comparison
    rigid_pred = [g_cal * r_[2] + r_rigid * r_[1] ** 2 for r_ in rows]
    rms_rigid = rms(real, rigid_pred, warmup_n)

    fb_range = fb_range or [round(0.7 + 0.1 * i, 2) for i in range(24)]   # 0.7–3.0 Hz
    zeta_range = zeta_range or [round(0.05 + 0.05 * i, 2) for i in range(20)]  # 0.05–1.00

    best = None
    for fb in fb_range:
        for zeta in zeta_range:
            pred = simulate_bounce(rows, L, k_scale, fb, zeta)
            e = rms(real, pred, warmup_n)
            if best is None or e < best[0]:
                best = (e, fb, zeta)
    e0, fb0, z0 = best
    # local refine around the coarse best point
    for fb in [fb0 + d_ for d_ in (-0.06, -0.03, 0, 0.03, 0.06)]:
        for zeta in [z0 + d_ for d_ in (-0.02, -0.01, 0, 0.01, 0.02)]:
            if fb <= 0.2 or zeta <= 0.01:
                continue
            pred = simulate_bounce(rows, L, k_scale, fb, zeta)
            e = rms(real, pred, warmup_n)
            if e < best[0]:
                best = (e, fb, zeta)
    rms_bounce, fb_best, zeta_best = best

    improvement = 100 * (1 - rms_bounce / rms_rigid) if rms_rigid > 0 else 0
    print(f"  rigid model  : |a|(t) RMS residual = {rms_rigid:.4f} m/s^2")
    print(f"  bounce model : |a|(t) RMS residual = {rms_bounce:.4f} m/s^2  "
          f"at fb={fb_best:.2f} Hz, zeta={zeta_best:.2f}  ({improvement:+.0f}% vs rigid)")
    sag_cm = 100 * G / (2 * math.pi * fb_best) ** 2
    print(f"  implied sag  : {sag_cm:.1f} cm static stretch under load at fb={fb_best:.2f} Hz")

    # cross-check against the independent amplitude-based (--stretch) estimate
    amp_eps = None
    rows_stretch = ad._stretch_rows(d)
    if len(rows_stretch) >= 6:
        # single-file version of the joint fit (unanchored — see analyze_data.py
        # --stretch for the proper multi-recording version); still useful as an
        # independent second read on this one recording's data.
        y = [r[0] for r in rows_stretch]
        cs = [r[1] for r in rows_stretch]
        z = [r[2] * r[3] for r in rows_stretch]
        gtry = ad.median([y[i] / cs[i] for i in range(len(y))])
        b = ad.lsq_solve([cs, [-v for v in z]], y)
        if b:
            amp_eps = max(0.0, b[1])
    if amp_eps and amp_eps > 1e-3:
        fb_amp = 2 / d["finalEstimate"]["periodSec"] / math.sqrt(amp_eps)
        print(f"  cross-check  : amplitude-based (single-file) fit implies "
              f"fb~{fb_amp:.2f} Hz (eps={amp_eps:.3f})  vs waveform fit fb={fb_best:.2f} Hz")

    return {"path": path, "rms_rigid": rms_rigid, "rms_bounce": rms_bounce,
            "fb": fb_best, "zeta": zeta_best, "improvement": improvement,
            "g_cal": g_cal, "r_rigid": r_rigid, "sag_cm": sag_cm}


def push_ringdown_check(path, fb_hint=None):
    """Look for a decaying oscillation near the bounce frequency in the
    PUSH-PHASE window (before free decay starts) — a plucked spring should
    ring at its own natural frequency right after an impulsive push. This
    is exploratory: push forcing is irregular (a human/hand), so a clean
    ringdown is not guaranteed to be visible."""
    with open(path) as f:
        d = json.load(f)
    S = d["samples"]
    C = d.get("cycles", [])
    if len(C) < 4:
        print(f"  {os.path.basename(path)}: too few cycles"); return
    m = ad.free_decay_start([math.radians(c["thetaMaxDeg"]) for c in C])
    cutoff = C[m]["t"] if m > 0 else 0
    if cutoff < 3:
        print(f"  {os.path.basename(path)}: push phase too short ({cutoff:.1f}s) to analyze")
        return
    push = [x for x in S if x["t"] < cutoff]
    # remove the slow swing-envelope trend with a wide moving-average, leaving
    # high-frequency residual (ringdown candidate)
    win = 15  # ~0.25s at 60Hz — shorter than a bounce period, longer than sample noise
    amag = [x["amag"] for x in push]
    trend = []
    for i in range(len(amag)):
        lo, hi = max(0, i - win), min(len(amag), i + win + 1)
        trend.append(sum(amag[lo:hi]) / (hi - lo))
    resid = [amag[i] - trend[i] for i in range(len(amag))]
    t = [x["t"] for x in push]

    ad.header(f"PUSH-PHASE RINGDOWN CHECK: {os.path.basename(path)}")
    print(f"  push window  : {cutoff:.1f} s, {len(push)} samples")
    # coarse DFT scan of the residual for a spectral peak
    fgrid = [0.5 + 0.02 * k for k in range(126)]  # 0.5-3.0 Hz
    mags = []
    t0 = t[0]
    for fq in fgrid:
        re = im = 0.0
        for i in range(len(resid)):
            ph = 2 * math.pi * fq * (t[i] - t0)
            re += resid[i] * math.cos(ph)
            im += resid[i] * math.sin(ph)
        mags.append(math.hypot(re, im) / len(resid))
    peak_i = max(range(len(mags)), key=lambda i: mags[i])
    print(f"  residual spectrum peak: {fgrid[peak_i]:.2f} Hz (mag {mags[peak_i]:.4f})")
    if fb_hint:
        near = abs(fgrid[peak_i] - fb_hint) < 0.25
        print(f"  vs waveform-fit bounce freq {fb_hint:.2f} Hz: "
              f"{'MATCHES (supports ringdown)' if near else 'no clear match — inconclusive'}")
    print("  note: push forcing is irregular (a human push), so a clean ringdown line")
    print("        is not guaranteed even if the bounce model is correct — this check")
    print("        is exploratory corroboration, not a standalone proof.")


def _device_of(path):
    with open(path) as f:
        ua = json.load(f).get("meta", {}).get("userAgent", "?")
    return "iPhone" if "iPhone" in ua else "Android" if "Android" in ua else ua[:24]


def main():
    args = sys.argv[1:]
    pushcheck = "--pushcheck" in args
    args = [a for a in args if a != "--pushcheck"]
    if args:
        paths = args
    else:
        paths = sorted(glob.glob("data/hammock-swing-*.json"))
        if not paths:
            sys.exit("no recordings found")

    # FYI only: print the anchored (multi-recording) g_cal per device for
    # comparison, but do NOT feed it into fit_bounce — see fit_bounce's
    # docstring for why that comparison turned out to be apples-to-oranges.
    if len(paths) > 1:
        stretch_res = ad.stretch_report(paths)
        for dev, info in stretch_res.items():
            print(f"\n  [FYI] anchored g_cal for {dev}: {info['gcal']:.3f} m/s^2 "
                  "(not used below — see fit_bounce docstring)")
        print()

    results = []
    for p in paths:
        r = fit_bounce(p)
        if r:
            results.append(r)
        if pushcheck:
            push_ringdown_check(p, fb_hint=r["fb"] if r else None)
    if len(results) > 1:
        ad.header("SUMMARY across recordings")
        for r in results:
            print(f"  {os.path.basename(r['path']):32s} fb={r['fb']:.2f} Hz  "
                  f"zeta={r['zeta']:.2f}  sag={r['sag_cm']:.1f} cm  "
                  f"improvement={r['improvement']:+.0f}%")


if __name__ == "__main__":
    main()
