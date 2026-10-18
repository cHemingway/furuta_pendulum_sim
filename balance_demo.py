"""Balance the Furuta pendulum in MuJoCo using the corrected model, and render a video.

Fixes applied relative to furata_pendulum_dynamics.py / mujoco_sim.py (see the
annotated notebook for the reasoning):
  - Controller uses the motor only (tau_1); tau_2 is a disturbance.
  - A_42 uses l_2 (pivot to CoM), not L_2.
  - Motor is voltage-driven: tau = (K_t/R) V - (K_t^2/R) theta_1_dot, so the
    gains come out in volts and back-EMF damping is modelled.
  - J_1 is a realistic estimate rather than 5.3e-4 (still needs measuring).
  - b_2 = q * J_2_hat from the ringdown, not q.
  - MuJoCo model is built from the same constants (explicit inertials), so
    the sim matches the linear model.

Non-idealities simulated:
  - Controller runs at CONTROL_HZ with zero-order hold on the voltage.
  - Encoders: quantised angles plus Gaussian noise; velocities are not measured.
  - State is estimated with a steady-state Kalman filter (angles -> full state).
  - Motor cogging torque, periodic in theta_1.

Usage: MUJOCO_GL=egl uv run python balance_demo.py [output_dir]
"""

from contextlib import nullcontext

import numpy as np
import mujoco
import control
from scipy.linalg import solve_discrete_are

# Physical constants (from furata_pendulum_dynamics.py unless noted)
g = 9.81
m_1, m_2 = 0.02, 0.01  # kg
l_1, l_2 = 0.02, 0.03279  # m, pivot to CoM
L_1, L_2 = 0.0384, 0.0703  # m, pivot to pendulum axis / pendulum end
J_1 = 1.0e-5  # kg m^2, arm about its CoM. ESTIMATE (was 5.3e-4), measure this
J_rotor = 1.0e-5  # kg m^2, GM3506 rotor. ESTIMATE, measure this
J_2 = 5.3e-6  # kg m^2, pendulum about its CoM (confirmed by ringdown)
q_2 = 0.85  # 1/s, ringdown decay rate (pendulum_damping.py)

# GM3506 motor (iFlight spec sheet: R = 5.6 ohm, ~140 RPM/V, 24N22P)
R = 5.6  # ohm
K_t = 1 / (140 * 2 * np.pi / 60)  # N m/A (= V s/rad)
V_max = 12.0  # V, supply
COG_PERIODS = 264  # cogging cycles per rev = lcm(24 slots, 22 poles)
COG_AMPLITUDE = 2e-3  # N m. ESTIMATE (~2% of rated torque), measure this

# Sensors and controller timing
CONTROL_HZ = 100  # pendulum_ringdown.csv shows ~10 ms cycle time
ENC_BITS = (14, 12)  # (motor, pendulum). Ringdown data has 2*pi/4096 steps -> 12 bit
ENC_NOISE = np.radians([0.02, 0.05])  # rad, 1-sigma noise on top of quantisation. ESTIMATE

J_0_hat = J_1 + J_rotor + m_1 * l_1**2 + m_2 * L_1**2
J_2_hat = J_2 + m_2 * l_2**2
b_1 = K_t**2 / R  # back-EMF damping dominates the arm bearing friction
b_2 = q_2 * J_2_hat


def linear_model():
    """Cazzolato & Prime linearisation about upright, input = motor voltage."""
    den = J_0_hat * J_2_hat - m_2**2 * L_1**2 * l_2**2
    A = np.array([
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, g * m_2**2 * l_2**2 * L_1 / den, -b_1 * J_2_hat / den, -b_2 * m_2 * l_2 * L_1 / den],
        [0, g * m_2 * l_2 * J_0_hat / den, -b_1 * m_2 * l_2 * L_1 / den, -b_2 * J_0_hat / den],
    ])
    B_tau = np.array([[0], [0], [J_2_hat / den], [m_2 * L_1 * l_2 / den]])
    return A, B_tau * (K_t / R)  # torque -> volts


def mujoco_xml():
    return f"""
<mujoco model="furuta_pendulum_measured">
  <compiler angle="radian"/>
  <option timestep="0.001"/>
  <visual>
    <headlight ambient="0.5 0.5 0.5" diffuse="0.6 0.6 0.6" specular="0.1 0.1 0.1"/>
    <map znear="0.001"/>
    <global offwidth="1280" offheight="720"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.26 0.12 0.36" rgb2="0.23 0.09 0.33"
      width="512" height="512" mark="cross" markrgb=".8 .8 .8"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance="0.2"/>
  </asset>
  <worldbody>
    <light pos="0.3 -0.3 0.8" dir="-0.3 0.3 -0.8"/>
    <camera name="view" pos="0.26 -0.12 0.27" mode="targetbody" target="arm1" fovy="45"/>
    <geom name="floor" type="plane" size="0.5 0.5 0.01" material="grid"/>
    <geom type="cylinder" size="0.02 0.065" pos="0 0 0.065" rgba="0.3 0.3 0.35 1"/>
    <geom type="cylinder" size="0.02 0.009" pos="0 0 0.139" rgba="0.15 0.15 0.15 1"/>
    <body name="arm1" pos="0 0 0.15">
      <joint name="motor" axis="0 0 1" damping="{b_1}" armature="{J_rotor}"/>
      <inertial pos="{l_1} 0 0" mass="{m_1}" diaginertia="1e-7 {J_1} {J_1}"/>
      <geom type="capsule" fromto="0 0 0 {L_1} 0 0" size="0.004" rgba="0.85 0.55 0.2 1"
        contype="0" conaffinity="0"/>
      <body name="arm2" pos="{L_1} 0 0">
        <joint name="pendulum" axis="1 0 0" damping="{b_2}"/>
        <inertial pos="0 0 {l_2}" mass="{m_2}" diaginertia="{J_2} {J_2} 1e-8"/>
        <geom type="capsule" fromto="0 0 0 0 0 {L_2}" size="0.003" rgba="0.85 0.85 0.9 1"
          contype="0" conaffinity="0"/>
        <site name="tip" pos="0 0 {L_2}" rgba="0 0 0 0"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <!-- ctrl is motor voltage; gear converts to stall torque -->
    <motor name="voltage" joint="motor" gear="{K_t / R}" ctrllimited="true"
      ctrlrange="{-V_max} {V_max}"/>
  </actuator>
</mujoco>
"""


def mujoco_linearisation(model):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    Ad = np.zeros((4, 4))
    Bd = np.zeros((4, 1))
    mujoco.mjd_transitionFD(model, data, 1e-6, True, Ad, Bd, None, None)
    dt = model.opt.timestep
    return (Ad - np.eye(4)) / dt, Bd / dt


def discretise(A, B, dt):
    sys_d = control.c2d(control.ss(A, B, np.eye(4), 0), dt, method="zoh")
    return np.asarray(sys_d.A), np.asarray(sys_d.B)


def design_lqr(Ad, Bd):
    # Bryson's rule: max acceptable theta_1, theta_2, theta_1_dot, theta_2_dot, and volts
    z_max = np.array([1.0, 0.05, 10.0, 2.0])
    u_max = np.array([6.0])
    K, _, poles = control.dlqr(Ad, Bd, np.diag(1 / z_max**2), np.diag(1 / u_max**2))
    return K[0], poles


def encoder_variance():
    lsb = 2 * np.pi / 2.0 ** np.array(ENC_BITS)
    return ENC_NOISE**2 + lsb**2 / 12  # Gaussian noise + uniform quantisation error


def design_kalman(Ad, Bd):
    """Steady-state Kalman filter gain M for x_hat = x_pred + M (y - C x_pred)."""
    C = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    # Process noise: unmodelled torque entering through the input direction, plus
    # some on the velocities. Tuned by sweep: setting the torque term as large as
    # the cogging amplitude gives an estimator pole near z = -1, which makes the
    # voltage chatter at Nyquist and (without sensor noise) drops the pendulum.
    torque_noise_V = 0.5e-3 / (K_t / R)  # 0.5 mN m expressed as volts
    Qn = Bd @ Bd.T * torque_noise_V**2 + np.diag([1e-9, 1e-9, 1e-3, 1e-3])
    Rn = np.diag(encoder_variance())
    P = solve_discrete_are(Ad.T, C.T, Qn, Rn)  # prior covariance
    M = P @ C.T @ np.linalg.inv(C @ P @ C.T + Rn)
    return C, M


def read_encoders(data, rng):
    """Quantised, noisy angle measurements [theta_1, theta_2]."""
    lsb = 2 * np.pi / 2.0 ** np.array(ENC_BITS)
    noisy = data.qpos[:2] + rng.normal(0, ENC_NOISE)
    return np.round(noisy / lsb) * lsb


def simulate(model, K, Ad, Bd, C, M, duration=8.0, push_time=1.0, push_len=0.05,
             push_force=0.035, fps=60, width=1280, height=720, seed=0, render=True, full_state=False,
             estimator="kalman", diff_cutoff_hz=None):
    rng = np.random.default_rng(seed)
    data = mujoco.MjData(model)
    arm1 = model.body("arm1").id
    arm2 = model.body("arm2").id
    tip = model.site("tip").id
    steps = int(round(duration / model.opt.timestep))
    ctrl_every = int(round(1 / (CONTROL_HZ * model.opt.timestep)))
    hist = {k: np.zeros((steps, 4)) for k in ("x", "x_hat")}
    hist["y"] = np.zeros((steps, 2))
    hist["u"] = np.zeros(steps)
    hist["t"] = np.zeros(steps)
    frames = []

    x_hat = np.zeros(4)
    y_prev = np.zeros(2)
    dt_ctrl = ctrl_every * model.opt.timestep
    u = 0.0
    y = np.zeros(2)

    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 30)
    except OSError:
        font = ImageFont.load_default()

    def push_dir():
        return data.xmat[arm1].reshape(3, 3)[:, 1]  # tangential to arm 1

    renderer_ctx = mujoco.Renderer(model, height=height, width=width) if render else nullcontext()
    with renderer_ctx as renderer:
        for i in range(steps):
            # Controller: runs at CONTROL_HZ, sees only quantised noisy angles
            if i % ctrl_every == 0:
                y = read_encoders(data, rng)
                if estimator == "kalman":
                    x_pred = Ad @ x_hat + Bd[:, 0] * u
                    x_hat = x_pred + M @ (y - C @ x_pred)
                else:  # "diff": backward-difference velocities, optional first-order low-pass
                    vel = (y - y_prev) / dt_ctrl
                    if diff_cutoff_hz is not None:
                        alpha = 1 - np.exp(-2 * np.pi * diff_cutoff_hz * dt_ctrl)
                        vel = x_hat[2:] + alpha * (vel - x_hat[2:])
                    x_hat = np.r_[y, vel]
                y_prev = y
                if full_state:  # ideal sensing: controller sees the true state
                    x_hat = np.r_[data.qpos[:2], data.qvel[:2]]
                fallen = abs(x_hat[1]) > 0.5  # pendulum angle, not arm angle
                u = 0.0 if fallen else float(np.clip(-K @ x_hat, -V_max, V_max))
                data.ctrl[0] = u

            # Disturbances: motor cogging, plus a push on the pendulum tip
            data.qfrc_applied[:] = 0
            data.qfrc_applied[0] = -COG_AMPLITUDE * np.sin(COG_PERIODS * data.qpos[0])
            pushing = push_time <= data.time < push_time + push_len
            if pushing:
                mujoco.mj_applyFT(model, data, push_force * push_dir(), np.zeros(3),
                                  data.site_xpos[tip], arm2, data.qfrc_applied)

            hist["t"][i] = data.time
            hist["x"][i] = np.r_[data.qpos[:2], data.qvel[:2]]
            hist["x_hat"][i] = x_hat
            hist["y"][i] = y
            hist["u"][i] = u
            mujoco.mj_step(model, data)

            if render and len(frames) < data.time * fps:
                renderer.update_scene(data, camera="view")
                show_arrow = push_time - 0.05 <= data.time < push_time + push_len + 0.05
                if show_arrow:
                    end = data.site_xpos[tip]
                    scn = renderer.scene
                    geom = scn.geoms[scn.ngeom]
                    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3),
                                        np.zeros(3), np.eye(3).flatten(),
                                        np.array([0.9, 0.1, 0.1, 1], dtype=np.float32))
                    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, 0.003,
                                         end - 0.06 * push_dir(), end - 0.004 * push_dir())
                    scn.ngeom += 1
                img = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(img)
                draw.text((20, 15), f"t = {data.time:4.2f} s", fill="white", font=font)
                draw.text((20, 55), f"pendulum {np.degrees(data.qpos[1]):+6.1f} deg", fill="white", font=font)
                draw.text((20, 95), f"arm      {np.degrees(data.qpos[0]):+6.1f} deg", fill="white", font=font)
                draw.text((20, 135), f"motor    {u:+6.2f} V", fill="white", font=font)
                draw.text((20, height - 50),
                          f"{CONTROL_HZ} Hz control, {ENC_BITS[0]}/{ENC_BITS[1]}-bit encoders + noise, "
                          f"cogging {COG_AMPLITUDE * 1e3:.0f} mN m", fill=(200, 200, 200), font=font)
                if show_arrow:
                    draw.text((width - 200, 15), "PUSH", fill=(255, 80, 80), font=font)
                frames.append(np.asarray(img))

    return frames, hist


if __name__ == "__main__":
    import sys
    import mediapy as media
    from matplotlib import pyplot as plt

    out = sys.argv[1] if len(sys.argv) > 1 else "."
    model = mujoco.MjModel.from_xml_string(mujoco_xml())

    A, B = linear_model()
    A_mj, B_mj = mujoco_linearisation(model)
    np.set_printoptions(precision=3, suppress=False, linewidth=120)
    print("Analytic A rows 3-4:\n", A[2:], "\nMuJoCo A rows 3-4:\n", A_mj[2:])
    print("Analytic B:", B.ravel(), "\nMuJoCo B: ", B_mj.ravel())

    dt = 1 / CONTROL_HZ
    Ad, Bd = discretise(A, B, dt)
    K, poles = design_lqr(Ad, Bd)
    C, M = design_kalman(Ad, Bd)
    print("K (V per [rad, rad, rad/s, rad/s]):", K)
    print("Closed-loop poles (z):", poles, "-> |z| max", np.abs(poles).max())
    print("Estimator poles (z):", np.linalg.eigvals((np.eye(4) - M @ C) @ Ad))

    frames, h = simulate(model, K, Ad, Bd, C, M)
    media.write_video(f"{out}/balance_demo.mp4", frames, fps=60)
    x, t, u = h["x"], h["t"], h["u"]
    settled = t > 4
    print(f"max |theta_2| = {np.degrees(np.abs(x[:, 1]).max()):.1f} deg, "
          f"max |theta_1| = {np.degrees(np.abs(x[:, 0]).max()):.1f} deg, "
          f"max |V| = {np.abs(u).max():.2f} V")
    print(f"after 4 s: theta_2 rms {np.degrees(x[settled, 1].std()):.2f} deg, "
          f"theta_1 range {np.degrees(np.ptp(x[settled, 0])):.2f} deg, "
          f"V rms {u[settled].std():.3f} V")

    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(9, 7), layout="tight")
    for a in ax:
        a.axvspan(1.0, 1.05, color="red", alpha=0.2)
    ax[0].plot(t, np.degrees(x[:, 0]), label=r"$\theta_1$ arm (true)")
    ax[0].plot(t, np.degrees(h["x_hat"][:, 0]), "--", label=r"$\hat\theta_1$ estimate")
    ax[0].set_ylabel("Arm (deg)")
    ax[0].legend(loc="upper right")
    ax[1].plot(t, np.degrees(h["y"][:, 1]), color="0.7", lw=0.8, label="encoder")
    ax[1].plot(t, np.degrees(x[:, 1]), label=r"$\theta_2$ pendulum (true)")
    ax[1].plot(t, np.degrees(h["x_hat"][:, 1]), "--", label=r"$\hat\theta_2$ estimate")
    ax[1].set_ylabel("Pendulum (deg)")
    ax[1].legend(loc="upper right")
    ax[2].plot(t, u)
    ax[2].set_ylabel("Motor voltage (V)")
    ax[2].set_xlabel("Time (s)")
    fig.savefig(f"{out}/balance_demo.png", dpi=110)
