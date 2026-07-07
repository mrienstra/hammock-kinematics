#!/usr/bin/env python3
"""Ground-truth regression test for the swing-analysis pipeline.

Simulates a damped pendulum with KNOWN parameters, renders it into the same
JSON format the app exports (including realistic imperfections: accelerometer
scale error, sensor noise, and the tangential force term the |a| model
neglects), runs analyze_data.py on it, and asserts the recovered numbers.

Run:  python3 synthetic_check.py        (prints PASS/FAIL per quantity)
"""

import json, math, os, random, re, subprocess, sys, tempfile

# ------------------------------------------------------------ ground truth
G = 9.81
L = 1.80          # effective pendulum length, m
R_PHONE = 2.30    # phone radius from pivot, m  (below CG: r > L)
K_SCALE = 0.975   # accelerometer scale error (device reads 2.5% low)
BETA = 0.010      # viscous damping: amplitude ~ e^(-BETA t)  → tau = 100 s
THETA0 = math.radians(28)
RATE = 60.0       # Hz
DUR = 60.0        # s
NOISE_GYRO = 0.005   # rad/s
NOISE_ACC = 0.05     # m/s^2
random.seed(42)

TAU_TRUE = 1 / BETA
# small-angle period with elliptic stretch at THETA0 (for reference bounds)
T_SMALL = 2 * math.pi * math.sqrt(L / G)


def simulate():
    """RK4 on theta'' = -(G/L) sin(theta) - 2*BETA*theta'."""
    def f(th, om):
        return om, -(G / L) * math.sin(th) - 2 * BETA * om

    th, om, t, dt = THETA0, 0.0, 0.0, 1.0 / 600
    out, next_sample = [], 0.0
    while t <= DUR:
        if t >= next_sample:
            out.append((t, th, om))
            next_sample += 1.0 / RATE
        k1a, k1b = f(th, om)
        k2a, k2b = f(th + 0.5 * dt * k1a, om + 0.5 * dt * k1b)
        k3a, k3b = f(th + 0.5 * dt * k2a, om + 0.5 * dt * k2b)
        k4a, k4b = f(th + dt * k3a, om + dt * k3b)
        th += (dt / 6) * (k1a + 2 * k2a + 2 * k3a + k4a)
        om += (dt / 6) * (k1b + 2 * k2b + 2 * k3b + k4b)
        t += dt
    return out


def build_export(states, tangential):
    """Rigid-pendulum export. `tangential` toggles the g·sinθ(1−r/L) term the
    analysis model neglects: OFF proves the pipeline algebra is exact; ON
    proves robustness to the documented approximation."""
    rows = []
    for (t, th, om) in states:
        f_r = G * math.cos(th) + R_PHONE * om * om
        f_t = G * math.sin(th) * (1 - R_PHONE / L) if tangential else 0.0
        amag = K_SCALE * math.hypot(f_r, f_t) + random.gauss(0, NOISE_ACC)
        rows.append((t, th, om, amag))
    return assemble(rows)


def assemble(rows):
    """Build the app-export dict from (t, theta, omega, amag_reading) rows."""
    D = 180 / math.pi
    samples, cycles = [], []
    prev_om, last_cross, halfs = 0.0, None, []
    for (t, th, om, amag) in rows:
        s = om + random.gauss(0, NOISE_GYRO)
        samples.append({
            "t": round(t, 4), "dt": round(1 / RATE, 4),
            "rrBeta": s * D, "rrGamma": random.gauss(0, 0.05),
            "rrAlpha": random.gauss(0, 0.05),
            "ax": 0, "ay": 0, "az": amag,  # analyzer uses amag, not the axes
            "amag": round(amag, 4),
            "s": round(s, 5),
            "axisX": 1.0, "axisY": 0.0, "axisZ": 0.0,
        })
        # cycle log from TRUE state (tests fit_decay against known decay)
        if prev_om != 0 and prev_om * om < 0 and last_cross is not None:
            halfs.append(t - last_cross)
            if len(halfs) >= 2:
                T = 2 * sum(halfs[-8:]) / len(halfs[-8:])
                cycles.append({
                    "t": round(t, 3), "T": round(T, 4),
                    "thetaMaxDeg": round(abs(th) * D, 2),  # om=0 ⇒ th = amplitude
                    "L_m": L, "r_m": None, "gCal": None,
                    "wPeak": 0, "aMax": 0, "aMin": 0,
                })
        if prev_om != 0 and prev_om * om < 0:
            last_cross = t
        elif last_cross is None and om != 0:
            last_cross = t
        prev_om = om
    T_avg = 2 * sum(halfs) / len(halfs)
    return {
        "meta": {"generatedAt": "synthetic", "userAgent": "synthetic",
                 "gTrue": G, "skippedSec": 0, "trimmedEndSec": 0,
                 "nSamples": len(samples), "durationSec": DUR,
                 "avgRateHz": RATE},
        "finalEstimate": {"periodSec": round(T_avg, 4),
                          "swingAngleDeg": round(cycles[-1]["thetaMaxDeg"], 2),
                          "effectiveLength_m": L,
                          "phoneRadius_m": None, "radiusOverLength": None},
        "cycles": cycles,
        "samples": samples,
    }


def run_case(states, tangential, tol_g, tol_r, verbose):
    data = build_export(states, tangential)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        path = f.name
    try:
        out = subprocess.run(
            [sys.executable, os.path.join(os.path.dirname(__file__), "analyze_data.py"), path],
            capture_output=True, text=True, check=True).stdout
    finally:
        os.unlink(path)
    if verbose:
        print(out)

    def grab(pattern):
        m = re.search(pattern, out)
        return float(m.group(1)) if m else None

    label = "tangential ON (realistic)" if tangential else "tangential OFF (ideal)"
    checks = [
        ("period T", grab(r"zero-cross period : T=([0-9.]+)"),
         data["finalEstimate"]["periodSec"], 0.03),
        ("calibrated g", grab(r"calibrated g : ([0-9.]+)"), K_SCALE * G, tol_g),
        ("phone radius r", grab(r"phone radius : r=([0-9.]+)"), R_PHONE, tol_r),
        ("decay tau", grab(r"tau=([0-9]+) s"), TAU_TRUE, 20),
    ]
    print(f"--- {label} ---")
    fails = 0
    for name, got, want, tol in checks:
        ok = got is not None and abs(got - want) <= tol
        fails += 0 if ok else 1
        print(f"  {'PASS' if ok else 'FAIL'}  {name:16s} got {got}  want {want:.3f} ±{tol}")
    # the simulation damps with a linear −2βθ' term, so the three-channel
    # fit must attribute the loss to the linear channel
    visc = "linear drag (viscous) dominated" in out
    print(f"  {'PASS' if visc else 'FAIL'}  mechanism        linear/viscous expected")
    return fails + (0 if visc else 1)


def simulate_spring(theta0, fb, dur):
    """Elastic-suspension (spring) pendulum: bounce frequency fb, static
    length L. Returns (t, theta, omega, amag_reading) rows — the phone rides
    the bob, so |a| is the spring's specific contact force (scale + noise
    applied here)."""
    wb = 2 * math.pi * fb
    zeta = 0.20                    # bounce-mode damping (fabric is lossy)
    L0 = L - G / wb ** 2           # natural length so static length = L

    def f(th, om, r, vr):
        return (om,
                (-G * math.sin(th) - 2 * vr * om) / r - 2 * BETA * om,
                vr,
                r * om * om + G * math.cos(th) - wb * wb * (r - L0) - 2 * zeta * wb * vr)

    th, om = theta0, 0.0
    r, vr = L0 + G * math.cos(theta0) / wb ** 2, 0.0  # static at release angle
    rows, t, dt, nxt = [], 0.0, 1.0 / 1200, 0.0
    while t <= dur:
        if t >= nxt:
            tension = wb * wb * (r - L0) + 2 * zeta * wb * vr  # specific contact force
            amag = K_SCALE * abs(tension) + random.gauss(0, NOISE_ACC)
            rows.append((t, th, om, amag))
            nxt += 1.0 / RATE
        k1 = f(th, om, r, vr)
        k2 = f(th + .5 * dt * k1[0], om + .5 * dt * k1[1], r + .5 * dt * k1[2], vr + .5 * dt * k1[3])
        k3 = f(th + .5 * dt * k2[0], om + .5 * dt * k2[1], r + .5 * dt * k2[2], vr + .5 * dt * k2[3])
        k4 = f(th + dt * k3[0], om + dt * k3[1], r + dt * k3[2], vr + dt * k3[3])
        th += dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        om += dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        r += dt / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        vr += dt / 6 * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])
        t += dt
    return rows


def run_spring_case():
    """Joint stretch fit must recover g_cal, the stretch parameter, and the
    corrected r from three same-device recordings (two gentle to pin g_cal,
    one big to expose stretch)."""
    import shutil
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import analyze_data as ad
    fb = 1.6
    tmpd = tempfile.mkdtemp()
    paths = []
    try:
        for i, th0 in enumerate((12, 14, 40)):
            data = assemble(simulate_spring(math.radians(th0), fb, 45.0))
            p = os.path.join(tmpd, f"spring-{i}-{th0}deg.json")
            with open(p, "w") as f:
                json.dump(data, f)
            paths.append(p)
        res = ad.stretch_report(paths)
    finally:
        shutil.rmtree(tmpd, ignore_errors=True)
    print("--- elastic suspension (joint stretch fit) ---")
    dev = res.get("synthetic")
    if not dev or len(dev["files"]) != 3:
        print("  FAIL  joint fit produced no usable result")
        return 1
    big = dev["files"][2]
    eps_true = (2 / (big["T"] * fb)) ** 2  # (2ω₀/ω_b)²
    checks = [
        ("joint g_cal", dev["gcal"], K_SCALE * G, 0.06),
        ("stretch eps (40deg)", big["eps"], eps_true, max(0.35 * eps_true, 0.03)),
        ("corrected r (40deg)", big["r_corr"], L, 0.30),
    ]
    fails = 0
    for name, got, want, tol in checks:
        ok = got is not None and abs(got - want) <= tol
        fails += 0 if ok else 1
        print(f"  {'PASS' if ok else 'FAIL'}  {name:20s} got {got:.3f}  want {want:.3f} ±{tol:.3f}")
    # and the stretch model must actually fit better than rigid
    better = dev["rms_stretch"] < 0.6 * dev["rms_rigid"]
    print(f"  {'PASS' if better else 'FAIL'}  model comparison     stretch RMS {dev['rms_stretch']:.3f}"
          f" vs rigid {dev['rms_rigid']:.3f} (expect clearly better)")
    return fails + (0 if better else 1)


def main():
    states = simulate()
    fails = 0
    # ideal: the model's assumptions hold exactly → recovery must be tight
    fails += run_case(states, tangential=False, tol_g=0.02, tol_r=0.08,
                      verbose="-v" in sys.argv)
    # realistic: include the neglected g·sinθ(1−r/L) term at a deliberately
    # harsh r/L=1.28, θ0=28° → small documented biases allowed
    fails += run_case(states, tangential=True, tol_g=0.10, tol_r=0.20,
                      verbose=False)
    fails += run_spring_case()
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
