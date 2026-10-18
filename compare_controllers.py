"""Compare the original controllers with the fixed ones in the MuJoCo sim.

All controllers face the same push on the pendulum tip at t = 1 s, in two conditions:
  - ideal:     1 kHz control, true state, no sensor noise, no cogging
  - realistic: 100 Hz control, Kalman-filtered quantised/noisy encoders, motor cogging

The plant is the corrected model from balance_demo.py (voltage-driven motor).
Gains are applied as volts: u = -K x, clipped to +-12 V.

Usage: uv run python compare_controllers.py [output_dir]
"""

import numpy as np
import control
import mujoco

import balance_demo as bd

KtR = bd.K_t / bd.R  # stall torque per volt


def original_notebook_gains():
    """tau_1 row of the LQR in furata_pendulum_dynamics.py, exactly as derived there."""
    g, m1, m2, l1, l2, L1, L2, J1, J2 = 9.81, 0.02, 0.01, 0.02, 0.03279, 0.0384, 0.0703, 0.00053, 0.0000053
    J0 = J1 + m1 * l1**2 + m2 * L1**2
    J2h = J2 + m2 * l2**2
    den = J0 * J2h - m2**2 * L1**2 * l2**2
    A = np.array([[0, 0, 1, 0], [0, 0, 0, 1],
                  [0, g * m2**2 * l2**2 * L1 / den, 0, 0],
                  [0, g * m2 * L2 * J0 / den, 0, 0]])  # L2 bug kept on purpose
    B = np.array([[0, 0], [0, 0],
                  [J2h / den, m2 * L1 * l2 / den],
                  [m2 * L1 * l2 / den, J0 / den]])
    K = control.lqr(A, B, np.diag([1, 1, 50, 100]), np.diag([1e-4] * 2))[0]
    return K[0]


def fixed_notebook_gains(dt=None):
    """Bryson's-rule LQR from furata_pendulum_dynamics_fixed.py (continuous, or dlqr at dt)."""
    A, B = bd.linear_model()
    Q = np.diag(1 / np.square([1.0, 0.05, 10.0, 2.0]))
    R = np.diag(1 / np.square([6.0]))
    if dt is None:
        return control.lqr(A, B, Q, R)[0][0]
    Ad, Bd = bd.discretise(A, B, dt)
    return control.dlqr(Ad, Bd, Q, R)[0][0]


def configure(ideal):
    if ideal:
        bd.CONTROL_HZ, bd.ENC_BITS, bd.ENC_NOISE, bd.COG_AMPLITUDE = 1000, (40, 40), np.zeros(2), 0.0
    else:
        bd.CONTROL_HZ, bd.ENC_BITS, bd.ENC_NOISE, bd.COG_AMPLITUDE = 100, (14, 12), np.radians([0.02, 0.05]), 2e-3


def run(K, ideal):
    configure(ideal)
    model = mujoco.MjModel.from_xml_string(bd.mujoco_xml())
    A, B = bd.linear_model()
    Ad, Bd = bd.discretise(A, B, 1 / bd.CONTROL_HZ)
    C, M = bd.design_kalman(Ad, Bd)
    _, h = bd.simulate(model, K, Ad, Bd, C, M, render=False, full_state=ideal)
    return h


def compare_estimators(out):
    """Kalman filter vs differentiating the encoder angles, realistic conditions, 100 Hz discrete LQR."""
    from matplotlib import pyplot as plt
    configure(False)
    model = mujoco.MjModel.from_xml_string(bd.mujoco_xml())
    A, B = bd.linear_model()
    Ad, Bd = bd.discretise(A, B, 1 / bd.CONTROL_HZ)
    C, M = bd.design_kalman(Ad, Bd)
    K = fixed_notebook_gains(1 / bd.CONTROL_HZ)
    estimators = {
        "Kalman filter": dict(estimator="kalman"),
        "Differentiated angles, 15 Hz low-pass": dict(estimator="diff", diff_cutoff_hz=15),
    }
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(10, 7), layout="constrained")
    for (name, kw), colour in zip(estimators.items(), ("C2", "C4")):
        h = bd.simulate(model, K, Ad, Bd, C, M, render=False, **kw)[1]
        t, x, xh, u = h["t"], h["x"], h["x_hat"], h["u"]
        err = np.degrees(np.sqrt(np.mean((xh[t > 4, 3] - x[t > 4, 3]) ** 2)))
        print(f"[estimator] {name}: max pendulum {np.degrees(np.abs(x[:, 1])).max():.1f} deg, "
              f"V rms (t>4 s) {u[t > 4].std():.3f} V, pendulum velocity estimate error rms {err:.1f} deg/s")
        ax[0].plot(t, np.degrees(x[:, 1]), color=colour, label=name)
        ax[1].plot(t, np.degrees(xh[:, 3]), color=colour, lw=0.8, label=f"{name} (estimate, {err:.0f} deg/s rms error)")
        ax[2].plot(t, u, color=colour, lw=0.8, label=name)
    ax[1].plot(t, np.degrees(x[:, 3]), "k", lw=0.8, label="true")
    ax[0].set_ylabel("Pendulum $\\theta_2$ (deg)")
    ax[1].set_ylabel("Pendulum $\\dot\\theta_2$ (deg/s)")
    ax[1].set_ylim(-150, 150)
    ax[2].set_ylabel("Motor voltage (V)")
    ax[2].set_xlabel("Time (s)")
    for a in ax:
        a.axvspan(1.0, 1.05, color="red", alpha=0.15)
        a.legend(fontsize=8, loc="upper right")
    fig.suptitle("Velocity estimation at 100 Hz with noisy 14/12-bit encoders and cogging")
    fig.savefig(f"{out}/compare_estimators.png", dpi=110)


def controllers(ideal):
    rate = 1000 if ideal else 100
    return {
        "Original notebook LQR (x2.61, as volts)": original_notebook_gains() * 2.61,
        "Original mujoco_sim.py gains (torque)": np.array([0.066, 2.299, 0.656, 18.894]) / KtR,
        "Fixed notebook: continuous LQR": fixed_notebook_gains(),
        f"Fixed notebook: discrete LQR ({rate} Hz)": fixed_notebook_gains(1 / rate),
    }


if __name__ == "__main__":
    import sys
    from matplotlib import pyplot as plt

    out = sys.argv[1] if len(sys.argv) > 1 else "."
    colours = ["C3", "C1", "C0", "C2"]
    styles = ["-", "-", "-", "--"]
    fig, ax = plt.subplots(3, 2, sharex=True, sharey="row", figsize=(13, 8), layout="constrained")
    fall_deg = 60

    for col, ideal in enumerate((True, False)):
        title = "Ideal: 1 kHz, true state, no noise or cogging" if ideal else \
            "Realistic: 100 Hz, Kalman filter on noisy 14/12-bit encoders, 2 mN m cogging"
        ax[0, col].set_title(title, fontsize=10)
        for (name, K), colour, ls in zip(controllers(ideal).items(), colours, styles):
            h = run(K, ideal)
            t, x, u = h["t"], h["x"], h["u"]
            fell = np.flatnonzero(np.abs(np.degrees(x[:, 1])) > fall_deg)
            end = fell[0] if fell.size else len(t)
            status = f"fell at {t[end]:.2f} s" if fell.size else \
                f"balanced, max pendulum {np.degrees(np.abs(x[:, 1])).max():.1f} deg"
            print(f"[{'ideal' if ideal else 'realistic'}] {name}: K={np.round(K, 3)} -> {status}")
            label = f"{name}: {status}"
            ax[0, col].plot(t[:end], np.degrees(x[:end, 1]), ls, color=colour, label=label)
            ax[1, col].plot(t[:end], np.degrees(x[:end, 0]), ls, color=colour)
            ax[2, col].plot(t[:end], u[:end], ls, color=colour, lw=0.8)
            if fell.size:
                ax[0, col].plot(t[end], np.sign(x[end, 1]) * fall_deg, "x", color=colour, ms=10, mew=2)
        ax[0, col].legend(fontsize=7.5, loc="lower right")
        for a in ax[:, col]:
            a.axvspan(1.0, 1.05, color="red", alpha=0.15)

    ax[0, 0].set_ylim(-fall_deg - 5, fall_deg + 5)
    ax[1, 0].set_ylim(-200, 200)
    ax[2, 0].set_ylim(-13, 13)
    ax[0, 0].set_ylabel("Pendulum $\\theta_2$ (deg)")
    ax[1, 0].set_ylabel("Arm $\\theta_1$ (deg)")
    ax[2, 0].set_ylabel("Motor voltage (V)")
    for a in ax[2]:
        a.set_xlabel("Time (s)")
    fig.suptitle("Same push on the pendulum tip at t = 1 s (red band). x = fell over")
    fig.savefig(f"{out}/compare_controllers.png", dpi=110)

    compare_estimators(out)
