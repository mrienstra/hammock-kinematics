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
    D = 180 / math.pi
    samples, cycles = [], []
    prev_om, last_cross, halfs = 0.0, None, []
    for (t, th, om) in states:
        s = om + random.gauss(0, NOISE_GYRO)
        # true specific force at radius R_PHONE. `tangential` toggles the
        # g·sinθ(1−r/L) term the analysis model neglects: OFF proves the
        # pipeline algebra is exact; ON proves robustness to the known
        # (documented) approximation at a deliberately harsh r/L and angle.
        f_r = G * math.cos(th) + R_PHONE * om * om
        f_t = G * math.sin(th) * (1 - R_PHONE / L) if tangential else 0.0
        amag = K_SCALE * math.hypot(f_r, f_t) + random.gauss(0, NOISE_ACC)
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
    visc = "viscous/air dominated" in out
    print(f"  {'PASS' if visc else 'FAIL'}  mechanism        viscous expected")
    return fails + (0 if visc else 1)


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
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
