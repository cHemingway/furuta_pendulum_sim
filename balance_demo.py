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

Usage: MUJOCO_GL=egl uv run python balance_demo.py
"""

import numpy as np
import mujoco
import control

# Physical constants (from furata_pendulum_dynamics.py unless noted)
g = 9.81
m_1, m_2 = 0.02, 0.01  # kg
l_1, l_2 = 0.02, 0.03279  # m, pivot to CoM
L_1, L_2 = 0.0384, 0.0703  # m, pivot to pendulum axis / pendulum end
J_1 = 1.0e-5  # kg m^2, arm about its CoM. ESTIMATE (was 5.3e-4), measure this
J_rotor = 1.0e-5  # kg m^2, GM3506 rotor. ESTIMATE, measure this
J_2 = 5.3e-6  # kg m^2, pendulum about its CoM (confirmed by ringdown)
q_2 = 0.85  # 1/s, ringdown decay rate (pendulum_damping.py)

# GM3506 motor (iFlight spec sheet: R = 5.6 ohm, ~140 RPM/V)
R = 5.6  # ohm
K_t = 1 / (140 * 2 * np.pi / 60)  # N m/A (= V s/rad)
V_max = 12.0  # V, supply

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


def design_lqr(A, B):
    # Bryson's rule: max acceptable theta_1, theta_2, theta_1_dot, theta_2_dot, and volts
    z_max = np.array([1.0, 0.05, 10.0, 2.0])
    u_max = np.array([6.0])
    Q = np.diag(1 / z_max**2)
    Rm = np.diag(1 / u_max**2)
    K, _, poles = control.lqr(A, B, Q, Rm)
    return K[0], poles


def simulate(model, K, duration=8.0, push_time=1.0, push_len=0.05, push_force=0.07,
             fps=60, width=1280, height=720):
    data = mujoco.MjData(model)
    arm1 = model.body("arm1").id
    arm2 = model.body("arm2").id
    steps = int(round(duration / model.opt.timestep))
    t_hist, x_hist, u_hist = np.zeros(steps), np.zeros((steps, 4)), np.zeros(steps)
    frames = []

    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 30)
    except OSError:
        font = ImageFont.load_default()

    with mujoco.Renderer(model, height=height, width=width) as renderer:
        for i in range(steps):
            x = np.r_[data.qpos[:2], data.qvel[:2]]
            fallen = abs(x[1]) > 0.5  # pendulum angle, not arm angle
            data.ctrl[0] = 0.0 if fallen else -K @ x

            # External push: force on the pendulum CoM, tangential to arm 1
            pushing = push_time <= data.time < push_time + push_len
            data.xfrc_applied[arm2] = 0
            if pushing:
                tangent = data.xmat[arm1].reshape(3, 3)[:, 1]
                data.xfrc_applied[arm2, :3] = push_force * tangent

            t_hist[i], x_hist[i], u_hist[i] = data.time, x, data.ctrl[0]
            mujoco.mj_step(model, data)

            if len(frames) < data.time * fps:
                renderer.update_scene(data, camera="view")
                show_arrow = push_time - 0.05 <= data.time < push_time + push_len + 0.05
                if show_arrow:
                    tangent = data.xmat[arm1].reshape(3, 3)[:, 1]
                    com = data.xipos[arm2]
                    scn = renderer.scene
                    geom = scn.geoms[scn.ngeom]
                    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3),
                                        np.zeros(3), np.eye(3).flatten(),
                                        np.array([0.9, 0.1, 0.1, 1], dtype=np.float32))
                    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, 0.003,
                                         com - 0.06 * tangent, com - 0.008 * tangent)
                    scn.ngeom += 1
                img = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(img)
                draw.text((20, 15), f"t = {data.time:4.2f} s", fill="white", font=font)
                draw.text((20, 55), f"pendulum {np.degrees(data.qpos[1]):+6.1f} deg", fill="white", font=font)
                draw.text((20, 95), f"arm      {np.degrees(data.qpos[0]):+6.1f} deg", fill="white", font=font)
                draw.text((20, 135), f"motor    {data.ctrl[0]:+6.2f} V", fill="white", font=font)
                if show_arrow:
                    draw.text((width - 200, 15), "PUSH", fill=(255, 80, 80), font=font)
                frames.append(np.asarray(img))

    return frames, t_hist, x_hist, u_hist


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

    K, poles = design_lqr(A, B)
    print("K (V per [rad, rad, rad/s, rad/s]):", K)
    print("Closed-loop poles (analytic):", poles)
    print("Closed-loop poles (MuJoCo lin.):", np.linalg.eigvals(A_mj - B_mj @ K[None]))

    frames, t, x, u = simulate(model, K)
    media.write_video(f"{out}/balance_demo.mp4", frames, fps=60)
    print(f"max |theta_2| = {np.degrees(np.abs(x[:, 1]).max()):.1f} deg, "
          f"max |theta_1| = {np.degrees(np.abs(x[:, 0]).max()):.1f} deg, "
          f"max |V| = {np.abs(u).max():.2f} V, final state = {x[-1]}")

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(9, 5), layout="tight")
    ax[0].plot(t, np.degrees(x[:, 0]), label=r"$\theta_1$ arm")
    ax[0].plot(t, np.degrees(x[:, 1]), label=r"$\theta_2$ pendulum")
    ax[0].axvspan(1.0, 1.05, color="red", alpha=0.2, label="push")
    ax[0].set_ylabel("Angle (deg)")
    ax[0].legend()
    ax[1].plot(t, u)
    ax[1].axvspan(1.0, 1.05, color="red", alpha=0.2)
    ax[1].set_ylabel("Motor voltage (V)")
    ax[1].set_xlabel("Time (s)")
    fig.savefig(f"{out}/balance_demo.png", dpi=110)
