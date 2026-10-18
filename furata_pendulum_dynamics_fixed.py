import marimo

__generated_with = "0.11.5"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Symbolic State Space model of Furuta Pendulum (fixed)

        A corrected copy of `furata_pendulum_dynamics.py`. The issues are explained in
        `furata_pendulum_dynamics_annotated.py`. We build a linear state space model of the
        Furuta pendulum about the upright position, symbolically, following
        "On the Dynamics of the Furuta Pendulum"[^1], then design a controller for it.

        [^1]:  Cazzolato, Benjamin Seth, Prime, Zebb, On the Dynamics of the Furuta Pendulum, Journal of Control Science and Engineering, 2011, 528341, 8 pages, 2011. https://doi.org/10.1155/2011/528341
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **What changed from the original**

    1. The controller is designed with the **motor only** ($\tau_1$). $\tau_2$ stays in the model as a disturbance.
    2. $A_{42}$ uses $l_2$, not $L_2$.
    3. The motor is modelled as **voltage-driven**, so back-EMF damping $K_t^2/R$ is included and the gains come out directly in volts.
    4. $J_1$ is a realistic estimate, and the motor rotor inertia is included.
    5. The damping coefficients are in the right units ($b_2 = q\hat J_2$), and the damping terms are kept in $A$.
    6. The plotting and simulation bugs are fixed. Initial states are near upright ($\theta_2 \approx 0$).
    7. $R$ for LQR comes from the motor's voltage limit (Bryson's rule), and the controller is **discretised for the 100 Hz loop**.
    8. A **Kalman filter with an arm disturbance state** estimates the velocities from the encoder angles, and cancels cogging.

    Values marked *estimate* still need measuring.
    """), kind="success")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## Overall state-space model""")
    return


@app.cell
def _(Matrix, S, diff, dynamicsymbols, mo, symbols, vlatex):
    # Write it out in the form x_dot = Ax + Bu
    mo.output.append("The generic state space model is")

    mo.output.append(mo.md(f"$$\\dot x = {vlatex(S('A*x + B*u'))}$$"))

    t = symbols('t')

    θ_1, θ_2 = dynamicsymbols('θ_1 θ_2') # Angles (x)
    τ_1, τ_2 = dynamicsymbols('τ_1 τ_2') # Torques (u)

    def furuta_symbolic_system():
        ''' Return the symbolic system for the Furuta pendulum, as described in the paper'''
        A31, A32, A33, A34 = symbols('A_31 A_32 A_33 A_34')
        A41, A42, A43, A44 = symbols('A_41 A_42 A_43 A_44')

        B31, B32, B41, B42 = symbols('B_31 B_32 B_41 B_42')

        A = Matrix([[0, 0, 1, 0],
                    [0, 0, 0, 1],
                    [A31, A32, A33, A34],
                    [A41, A42, A43, A44]])

        x = Matrix([θ_1, θ_2, diff(θ_1), diff(θ_2)])

        x_dot = diff(x)

        B = Matrix([[0, 0],
                    [0, 0],
                    [B31, B32],
                    [B41, B42]])

        u = Matrix([τ_1, τ_2])

        return A, x, x_dot, B, u


    _A, x, _x_dot, _B, u = furuta_symbolic_system()

    mo.output.append("For our Furuta pendulum, these are given by: ")
    mo.output.append(
        mo.hstack([
            mo.md(f"$$\\dot x= {vlatex(_x_dot)}$$"),
            mo.md(f"$A= {vlatex(_A)}$"),
            mo.md(f"State $x= {vlatex(x)}$"),
            mo.md(f"$B= {vlatex(_B)}$"),
            mo.md(f"Input $u= {vlatex(u)}$")
        ],
        justify="center", gap=4, align="center")
    )

    mo.output.append(mo.md(f"""With scalar variables:

    - $\\theta_1$, $\\theta_2$ are the angles of the first arm (motor) and second arm (pendulum). $\\theta_2 = 0$ is **upright**.
    - $\\tau_1$ is the motor torque, which we control. $\\tau_2$ is a disturbance torque on the pendulum, which we **cannot** control.
    """))
    return furuta_symbolic_system, t, u, x, θ_1, θ_2, τ_1, τ_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Values of constants $A, B$""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""### Physical parameters""")
    return


@app.cell
def _(mo, np, symbols):
    g = symbols('g') # Gravity

    m_1, m_2 = symbols('m_1 m_2') # Mass of each arm
    l_1, l_2 = symbols('l_1 l_2') # Lengths to centre of mass for each arm
    L_1, L_2 = symbols('L_1 L_2') # Lengths to _end_ for each arm

    J_1 = symbols('J_1') # Inertia of first arm about its CoM
    J_2 = symbols('J_2') # Inertia of second arm about its CoM
    J_r = symbols('J_r') # Motor rotor inertia

    q_2 = symbols('q_2') # Pendulum ringdown decay rate [1/s]
    b_1f = symbols('b_1f') # Arm bearing friction (excluding back-EMF)

    K_t, R_m = symbols('K_t R_m') # Motor torque constant, winding resistance

    constant_values = {
        g: 9.81,
        m_1: 0.02, # Guessing 50% density of ABS for 3d prints, plus 1g * 11 screws
        m_2: 0.01, # From solidworks
        l_1: 0.02, # Bit of a guess here
        l_2: 0.03279,
        L_1: 0.0384, #  From centre to centre in solidworks
        L_2: 0.0703,  # From axis to end face in solidworks (not used in the linear model)
        J_1: 1.0e-5, # ESTIMATE: ~m L^2/12 for the arm plus hub. Measure this
        J_r: 1.0e-5, # ESTIMATE: GM3506 rotor. Measure this
        J_2: 0.0000053, # From solidworks; confirmed by the ringdown (equivalent length 49 mm)
        q_2: 0.85, # See pendulum_damping.py for how we measured this
        b_1f: 0, # Assumed small compared with back-EMF damping
        K_t: 1 / (140 * 2 * np.pi / 60), # GM3506: 140 RPM/V -> 0.0682 N m/A
        R_m: 5.6, # GM3506 winding resistance, ohms
    }

    V_max = 12.0 # Supply voltage
    f_control = 100 # Hz, control loop rate (pendulum_ringdown.csv shows ~10 ms cycle time)

    mo.hstack(
        [mo.md(f"${k}={v:.4g}$") for k, v in constant_values.items()],
        gap=0, align="center", wrap=True
    )
    return (
        J_1,
        J_2,
        J_r,
        K_t,
        L_1,
        L_2,
        R_m,
        V_max,
        b_1f,
        constant_values,
        f_control,
        g,
        l_1,
        l_2,
        m_1,
        m_2,
        q_2,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""### Simplified inertia and damping values""")
    return


@app.cell(hide_code=True)
def _(J_1, J_2, J_r, K_t, L_1, R_m, b_1f, l_1, l_2, latex, m_1, m_2, mo, q_2):
    J_1_hat = J_1 + J_r + m_1*l_1**2
    mo.output.append(mo.md(f"Total moment of inertia of arm 1 (with rotor) about the motor axis: $\\hat{{J_1}} = {latex(J_1_hat)}$"))

    J_2_hat = J_2 + m_2*l_2**2
    mo.output.append(mo.md(f"Total moment of inertia of arm 2 about its pivot: $\\hat{{J_2}} = {latex(J_2_hat)}$"))

    J_0_hat = J_1_hat + m_2 * L_1**2
    mo.output.append(mo.md(f"Arm 1 with the pendulum mass lumped at its pivot: $\\hat{{J_0}} = {latex(J_0_hat)}$"))

    b_1 = b_1f + K_t**2 / R_m
    mo.output.append(mo.md(f"Arm damping, mostly motor back-EMF under voltage drive: $b_1 = {latex(b_1)}$"))

    b_2 = q_2 * J_2_hat
    mo.output.append(mo.md(f"Pendulum damping from the ringdown decay rate: $b_2 = {latex(b_2)}$"))
    return J_0_hat, J_1_hat, J_2_hat, b_1, b_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Values of A and B matrices""")
    return


@app.cell
def _(J_0_hat, J_2_hat, K_t, L_1, Matrix, R_m, b_1, b_2, g, l_2, m_2):
    den = J_0_hat * J_2_hat - m_2**2 * L_1**2 * l_2**2

    A_31 = 0
    A_32 = (g * m_2**2 * l_2**2 * L_1) / den
    A_33 = -b_1 * J_2_hat / den
    A_34 = -b_2 * m_2 * l_2 * L_1 / den

    A_41 = 0
    A_42 = (g * m_2 * l_2 * J_0_hat) / den
    A_43 = -b_1 * m_2 * l_2 * L_1 / den
    A_44 = -b_2 * J_0_hat / den

    A = Matrix([
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [A_31, A_32, A_33, A_34],
        [A_41, A_42, A_43, A_44]
    ])

    # The paper switches the order it lists these between A and B, watch out when comparing
    B_31 = J_2_hat / den
    B_41 = (m_2 * L_1 * l_2) / den
    B_32 = (m_2 * L_1 * l_2) / den
    B_42 = J_0_hat / den

    B = Matrix([
        [0,   0],
        [0,   0],
        [B_31, B_32],
        [B_41, B_42]
    ])

    # The input we actually control is the motor voltage V: tau_1 = (K_t/R) V (back-EMF is in A)
    B_V = B[:, 0] * K_t / R_m

    A, B, B_V
    return (
        A,
        A_31,
        A_32,
        A_33,
        A_34,
        A_41,
        A_42,
        A_43,
        A_44,
        B,
        B_31,
        B_32,
        B_41,
        B_42,
        B_V,
        den,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Simulate open loop (no control)

        Our open loop equation is $\dot{x} = \mathbf{A}x$, with solution $x(t) = e^{\mathbf{A}t}x(0)$.

        We use the matrix exponential directly. The eigenvector form $T e^{Dt} T^{-1}$ fails here, because $A$ has a repeated eigenvalue at 0 with only one eigenvector, so $T$ is singular.
        """
    )
    return


@app.cell(hide_code=True)
def _(A, B_V, constant_values, mo, np, np_to_latex, sympy):
    A_sub = sympy.matrix2numpy(A.subs(constant_values), dtype=float)
    B_sub = sympy.matrix2numpy(B_V.subs(constant_values), dtype=float) # Motor voltage input only

    eigenvalues, eigenvectors = np.linalg.eig(A_sub) # numpy returns (values, vectors)

    mo.md(f"""
    Substituting `constant_values` in: $$A={np_to_latex(A_sub)} \\quad B_V={np_to_latex(B_sub)}$$

    Open-loop eigenvalues: ${np_to_latex(eigenvalues[None, :], dp=2)}$

    The positive eigenvalue is the pendulum falling over. Its time constant is {1000/eigenvalues.real.max():.0f} ms.
    """)
    return A_sub, B_sub, eigenvalues, eigenvectors


@app.cell
def _(A_sub, expm, np, plt):
    x_0 = np.array([0, 0.05, 0, 0]) # Pendulum 0.05 rad (~3 deg) from upright, no velocity

    _t = np.linspace(0, 0.3, 300)
    _x_sim = np.array([expm(A_sub * _tt) @ x_0 for _tt in _t])

    _fig, _ax = plt.subplots(figsize=(7, 3), layout="tight")
    _ax.plot(_t, np.degrees(_x_sim[:, 0]), label=r"$\theta_1$ arm")
    _ax.plot(_t, np.degrees(_x_sim[:, 1]), label=r"$\theta_2$ pendulum")
    _ax.set_xlabel("Time (s)")
    _ax.set_ylabel("Angle (deg)")
    _ax.set_title("Open loop: falls over (linear model, so only valid for small angles)")
    _ax.legend()
    _fig
    return (x_0,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Check for Controllability

        Is the system controllable from the **motor alone**? If $\text{rank}(\mathcal{C}) = 4$, a stabilising controller exists.
        """
    )
    return


@app.cell
def _(A_sub, B_sub, control, np):
    f"{np.linalg.matrix_rank(control.ctrb(A_sub, B_sub))=}"
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Sign check: which gains can possibly work?

        With $u = -Kx$, the closed-loop characteristic polynomial's constant term is

        $$c_0 = k_1(a_{32}b_4 - a_{42}b_3) = -k_1\frac{g m_2 l_2}{\Delta}\frac{K_t}{R}$$

        All coefficients must be positive for stability, so **$k_1$ (the arm-angle gain) must be negative**. This holds for any masses, inertias or damping. Every gain set in the original notebook and `mujoco_sim.py` had $k_1 > 0$, so none of them could ever balance the pendulum.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## Make Pole Placement Controller""")
    return


@app.cell
def _(A_sub, B_sub, control, mo, np, np_to_latex):
    # The open-loop unstable pole is at about +14 rad/s; place the closed-loop poles a bit faster
    poles_desired = [-18, -20, -22, -24]
    K_place = control.place(A_sub, B_sub, poles_desired)

    mo.md(f"""
    Using `control.place(A, B_V, {poles_desired})`

    Gains (volts per rad, rad/s): $K={np_to_latex(K_place, dp=3)}$

    Closed-loop eigenvalues: ${np_to_latex(np.linalg.eigvals(A_sub - B_sub @ K_place)[None, :], dp=2)}$
    """)
    return K_place, poles_desired


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Make LQR Controller""")
    return


@app.cell
def _(A_sub, B_sub, K_t, R_m, constant_values, control, mo, np, np_to_latex):
    def brysons_rule(z_max, u_max):
        Q = np.diag(1/np.power(z_max, 2))
        R = np.diag(1/np.power(u_max, 2))
        return Q, R

    # Q from "Swing-up and LQR stabilization of rotary inverted pendulum", Park, Kim, Lee,
    # with R from our motor's torque limit. Their Q weights velocities 50-100x more than
    # angles, which suits their rig but not ours: the gains it gives are enormous.
    _KtR = float((K_t / R_m).subs(constant_values))
    K_park = control.lqr(A_sub, B_sub / _KtR, np.diag([1, 1, 50, 100]), np.diag([1 / 0.1**2]))[0] / _KtR

    # Bryson's rule: the largest acceptable value of each state, and of the input
    Q, R = brysons_rule(
        z_max=[1.0, 0.05, 10.0, 2.0], # theta_1 [rad], theta_2 [rad], theta_1_dot, theta_2_dot [rad/s]
        u_max=[6.0],                  # volts: half the supply, leaving headroom
    )
    K_lqr = control.lqr(A_sub, B_sub, Q, R)[0]

    mo.md(f"""
    Gains are in volts per [rad, rad, rad/s, rad/s].

    Park et al. $Q$: $K={np_to_latex(K_park, dp=2)}$. The ~{K_park[0, 1]:.0f} V/rad on $\\theta_2$ saturates the 12 V supply at a fraction of a degree.

    Bryson's rule: $K_{{LQR}}={np_to_latex(K_lqr, dp=3)}$. Note the sign pattern: $k_1 < 0$, as the sign check requires.

    Closed-loop eigenvalues: ${np_to_latex(np.linalg.eigvals(A_sub - B_sub @ K_lqr)[None, :], dp=2)}$
    """)
    return K_lqr, K_park, Q, R, brysons_rule


@app.cell(hide_code=True)
def _(mo):
    mo.md("""Closed-loop response from 0.05 rad (~3°). Should return to zero!""")
    return


@app.cell
def _(A_sub, B_sub, K_lqr, K_place, control, np, plt, x_0):
    _fig, _ax = plt.subplots(2, 1, sharex=True, figsize=(7, 5), layout="tight")
    for _name, _K, _style in (("LQR", K_lqr, "-"), ("Pole placement", K_place, "--")):
        _cl = control.ss(A_sub - B_sub @ _K, B_sub, np.eye(4), 0) # closed loop: A - B K
        _resp = control.initial_response(_cl, T=np.linspace(0, 3, 1000), X0=x_0)
        _u = -(_K @ _resp.states)[0]
        _ax[0].plot(_resp.time, np.degrees(_resp.states[0]), _style, color="C0", label=rf"$\theta_1$ arm ({_name})")
        _ax[0].plot(_resp.time, np.degrees(_resp.states[1]), _style, color="C1", label=rf"$\theta_2$ pendulum ({_name})")
        _ax[1].plot(_resp.time, _u, _style, color="C2", label=_name)
    _ax[0].set_ylabel("Angle (deg)")
    _ax[0].legend(fontsize=8)
    _ax[1].set_ylabel("Motor voltage (V)")
    _ax[1].set_xlabel("Time (s)")
    _ax[1].legend(fontsize=8)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Discretise for the real control loop

        The controller runs at a fixed rate, with the voltage held constant between updates (zero-order hold). A continuous-time design assumes instant feedback. At 100 Hz the delay is significant compared with the pendulum's ~70 ms falling time constant, so we design directly on the discretised model with `dlqr`.
        """
    )
    return


@app.cell
def _(A_sub, B_sub, K_lqr, Q, R, control, f_control, mo, np, np_to_latex):
    dt = 1 / f_control
    _sys_d = control.c2d(control.ss(A_sub, B_sub, np.eye(4), 0), dt, method="zoh")
    A_d, B_d = np.asarray(_sys_d.A), np.asarray(_sys_d.B)

    _z_cont = np.linalg.eigvals(A_d - B_d @ K_lqr)
    K_d = control.dlqr(A_d, B_d, Q, R)[0]
    _z_disc = np.linalg.eigvals(A_d - B_d @ K_d)

    mo.md(f"""
    At {f_control} Hz, the **continuous** $K_{{LQR}}$ gives discrete closed-loop poles with
    $|z|_{{max}} = {np.abs(_z_cont).max():.3f}$ ({"unstable" if np.abs(_z_cont).max() > 1 else "stable"}; $|z| < 1$ is needed).
    That's before adding any sensor noise or cogging.

    The **discrete** design: $K_d = {np_to_latex(K_d, dp=3)}$ V per [rad, rad, rad/s, rad/s],
    with $|z|_{{max}} = {np.abs(_z_disc).max():.3f}$.

    **This is the controller to use on the hardware:** $V = -K_d\\hat x$, updated at {f_control} Hz.
    The encoders only measure angles, so $\\hat x$ must come from an estimator. `balance_demo.py`
    uses a steady-state Kalman filter.
    """)
    return A_d, B_d, K_d, dt


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## State estimation

        $V = -K_d x$ needs the whole state, but the encoders only measure the two **angles**, and with limited resolution:
        14-bit on the motor, and 12-bit on the pendulum (`pendulum_ringdown.csv` steps in $2\pi/4096$). At 100 Hz, one
        pendulum count per sample is already $0.088°/0.01\,\text{s} \approx 9°/\text{s}$ of velocity, so just differentiating the
        angles gives noisy velocities.

        A **Kalman filter** uses the model instead. Each cycle it predicts the state from the last estimate and the voltage it
        applied, then corrects the prediction with the measured angles:

        $$\hat x^- = A_d \hat x + B_d u, \qquad \hat x = \hat x^- + M\,(y - C\hat x^-), \qquad y = [\theta_1, \theta_2]$$

        The steady-state gain $M$ comes from the discrete Riccati equation. Its inputs are the measurement noise covariance (from the
        encoders) and a process noise covariance, which is a tuning knob for how much to trust the model.

        ### Adding a disturbance state

        The motor's **cogging** torque (and friction, and any error in $J_1$) acts like an unknown, slowly changing extra
        voltage $d$ at the motor. If we add it to the state as a random walk,

        $$\begin{bmatrix} x \\ d \end{bmatrix}_{k+1} = \begin{bmatrix} A_d & B_d \\ 0 & 1 \end{bmatrix}\begin{bmatrix} x \\ d \end{bmatrix}_k + \begin{bmatrix} B_d \\ 0 \end{bmatrix} u_k,$$

        the filter estimates $\hat d$, and the controller cancels it with $V = -K_d\hat x - \hat d$. That acts like **integral
        action** on the arm, so the arm stops resting wherever a cogging detent holds it.
        """
    )
    return


@app.cell
def _(B_d, K_t, R_m, constant_values, np, solve_discrete_are):
    # Sensor model
    enc_bits = np.array([14, 12])                # motor, pendulum
    enc_noise = np.radians([0.02, 0.05])         # 1-sigma noise on top of quantisation. ESTIMATE
    enc_lsb = 2 * np.pi / 2.0**enc_bits
    R_n = np.diag(enc_noise**2 + enc_lsb**2 / 12)  # measurement noise covariance
    C_meas = np.eye(2, 4)                        # we measure theta_1, theta_2

    KtR = float((K_t / R_m).subs(constant_values))  # N m per volt

    def kalman_gain(Az, Cz, Qn, Rn):
        """Steady-state gain M for x_hat = x_pred + M (y - C x_pred)."""
        P = solve_discrete_are(Az.T, Cz.T, Qn, Rn)  # prior covariance
        return P @ Cz.T @ np.linalg.inv(Cz @ P @ Cz.T + Rn)

    # Plain Kalman filter. Process noise: 0.5 mN m of unmodelled torque at the motor, plus
    # some on the velocities. (Setting it equal to the full cogging amplitude puts an
    # estimator pole near z = -1 and makes the voltage chatter.)
    Q_n = B_d @ B_d.T * (0.5e-3 / KtR) ** 2 + np.diag([1e-9, 1e-9, 1e-3, 1e-3])
    return C_meas, KtR, Q_n, R_n, enc_bits, enc_lsb, enc_noise, kalman_gain


@app.cell
def _(A_d, B_d, C_meas, Q_n, R_n, kalman_gain, mo, np, np_to_latex):
    M_kf = kalman_gain(A_d, C_meas, Q_n, R_n)

    # Augmented with an arm disturbance d (volts-equivalent), modelled as a random walk.
    # q_d = 0.03 V per step was picked by sweep in MuJoCo: smaller values lost the pendulum
    # when J_1 was 4x off; larger ones (0.3) caused a large limit cycle.
    q_d = 0.03
    A_z = np.block([[A_d, B_d], [np.zeros((1, 4)), np.eye(1)]])
    B_z = np.r_[B_d, np.zeros((1, 1))]
    C_z = np.hstack([C_meas, np.zeros((2, 1))])
    Q_z = np.diag([1e-9, 1e-9, 1e-3, 1e-3, q_d**2])
    M_z = kalman_gain(A_z, C_z, Q_z, R_n)

    _p_kf = np.linalg.eigvals((np.eye(4) - M_kf @ C_meas) @ A_d)
    _p_z = np.linalg.eigvals((np.eye(5) - M_z @ C_z) @ A_z)
    mo.md(f"""
    Kalman gain $M = {np_to_latex(M_kf, dp=3)}$, estimator poles ${np_to_latex(_p_kf[None, :], dp=2)}$

    With disturbance state: estimator poles ${np_to_latex(_p_z[None, :], dp=2)}$

    All poles have $|z| < 1$, and none is near $-1$ (that would make the estimate flip sign every sample).
    """)
    return A_z, B_z, C_z, M_kf, M_z, Q_z, q_d


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Compare the estimators

        A linear simulation of the plant at 1 kHz, with the 100 Hz controller $K_d$ fed by each estimator. The simulation includes:

        - GM3506 cogging: 264 cycles per revolution, 2 mN m amplitude (*estimate*)
        - quantised, noisy encoders, as above
        - a 0.035 N push on the pendulum tip for 50 ms at $t = 1$ s

        `balance_demo.py` and `compare_controllers.py` run the same comparison in MuJoCo, with the full nonlinear dynamics.
        """
    )
    return


@app.cell
def _(A_sub, B, B_sub, K_d, KtR, M_kf, M_z, A_d, B_d, A_z, B_z, C_meas, C_z, constant_values, control, dt, enc_lsb, enc_noise, np, sympy):
    def simulate_estimators(duration=8.0, sub_dt=1e-3, seed=0):
        B_tau = sympy.matrix2numpy(B.subs(constant_values), dtype=float)  # torque columns: [arm, pendulum]
        _sys = control.c2d(control.ss(A_sub, np.hstack([B_sub, B_tau]), np.eye(4), 0), sub_dt, method="zoh")
        Ap, Bp = np.asarray(_sys.A), np.asarray(_sys.B)  # plant at 1 kHz: inputs [V, tau_arm, tau_pend]
        n_sub = int(round(dt / sub_dt))
        steps = int(round(duration / sub_dt))
        L_2 = 0.0703
        cog_amp, cog_periods = 2e-3, 264
        results = {}
        for name in ("Differentiated angles (15 Hz low-pass)", "Kalman filter", "Kalman + disturbance state"):
            rng = np.random.default_rng(seed)
            x = np.zeros(4)
            x_hat = np.zeros(4)
            z = np.zeros(5)
            y_prev = np.zeros(2)
            u = 0.0
            d_hat = 0.0
            hist = np.zeros((steps, 6))
            for i in range(steps):
                t = i * sub_dt
                if i % n_sub == 0:
                    y = np.round((x[:2] + rng.normal(0, enc_noise)) / enc_lsb) * enc_lsb
                    if name.startswith("Differentiated"):
                        alpha = 1 - np.exp(-2 * np.pi * 15 * dt)
                        x_hat = np.r_[y, x_hat[2:] + alpha * ((y - y_prev) / dt - x_hat[2:])]
                    elif name == "Kalman filter":
                        x_pred = A_d @ x_hat + B_d[:, 0] * u
                        x_hat = x_pred + M_kf @ (y - C_meas @ x_pred)
                    else:
                        z_pred = A_z @ z + B_z[:, 0] * u
                        z = z_pred + M_z @ (y - C_z @ z_pred)
                        x_hat, d_hat = z[:4], z[4]
                    y_prev = y
                    u = float(np.clip(-K_d[0] @ x_hat - d_hat, -12, 12))
                tau_cog = -cog_amp * np.sin(cog_periods * x[0])
                tau_push = 0.035 * L_2 if 1.0 <= t < 1.05 else 0.0
                hist[i] = [t, *x[:2], x[3], x_hat[3], u]
                x = Ap @ x + Bp @ np.array([u, tau_cog, tau_push])
            results[name] = hist
        return results

    estimator_runs = simulate_estimators()
    return estimator_runs, simulate_estimators


@app.cell
def _(estimator_runs, mo, np, plt):
    _fig, _ax = plt.subplots(3, 1, sharex=True, figsize=(8, 7), layout="constrained")
    _rows = []
    for (_name, _h), _c in zip(estimator_runs.items(), ("C4", "C2", "C1")):
        _t, _th1, _th2, _w2, _w2_hat, _u = _h.T
        _s = _t > 4
        _rows.append(
            f"| {_name} | {np.degrees(np.abs(_th2)).max():.1f}° | {np.degrees(np.ptp(_th1[_s])):.1f}° "
            f"| {_u[_s].std():.2f} V | {np.degrees(np.sqrt(np.mean((_w2_hat[_s] - _w2[_s])**2))):.0f}°/s |"
        )
        _ax[0].plot(_t, np.degrees(_th2), color=_c, label=_name)
        _ax[1].plot(_t, np.degrees(_th1), color=_c)
        _ax[2].plot(_t, _u, color=_c, lw=0.7)
    for _a in _ax:
        _a.axvspan(1.0, 1.05, color="red", alpha=0.15)
    _ax[0].set_ylabel(r"Pendulum $\theta_2$ (deg)")
    _ax[0].legend(fontsize=8)
    _ax[1].set_ylabel(r"Arm $\theta_1$ (deg)")
    _ax[2].set_ylabel("Motor voltage (V)")
    _ax[2].set_xlabel("Time (s)")

    mo.vstack([
        _fig,
        mo.md(
            "| Estimator | Pendulum peak | Arm wander (t > 4 s) | Voltage jitter (rms) | Pendulum velocity error (rms) |\n"
            "|---|---|---|---|---|\n" + "\n".join(_rows)
        ),
        mo.md(
            "The disturbance state stops the arm wandering between cogging detents, for slightly more voltage jitter. "
            "Differentiated angles respond fastest to the push and need no model, which makes them more "
            "robust to an error in $J_1$ (they still worked in MuJoCo with $J_1$ 6× off; both Kalman filters "
            "fail there). Once $J_1$ and the rotor inertia are measured, use the disturbance-state filter."
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## Imports and convenience functions""")
    return


@app.cell
def _():
    import marimo as mo
    import sympy
    from sympy import symbols, Eq, diff, Matrix, S, latex
    from sympy.physics.mechanics import dynamicsymbols, mechanics_printing
    from sympy.physics.vector.printing import vlatex
    from scipy.linalg import expm, solve_discrete_are
    import numpy as np
    import control
    from matplotlib import pyplot as plt
    mechanics_printing()
    return (
        Eq,
        Matrix,
        S,
        control,
        diff,
        dynamicsymbols,
        expm,
        latex,
        mechanics_printing,
        mo,
        np,
        plt,
        solve_discrete_are,
        sympy,
        symbols,
        vlatex,
    )


@app.cell
def _(np):
    def np_to_latex(a, dp=3):
        ''' Display a numpy matrix (real or complex) as latex'''
        a = np.atleast_2d(np.asarray(a))
        if a.ndim > 2:
            raise ValueError('bmatrix can at most display two dimensions')

        def fmt(v):
            v = complex(v)
            re, im = round(v.real, dp), round(v.imag, dp)
            if im == 0:
                return f"{re:g}"
            return f"{re:g} {'+' if im > 0 else '-'} {abs(im):g}j"

        rows = [' & '.join(fmt(v) for v in row) + r'\\' for row in a]
        return '\n'.join([r'\begin{bmatrix}'] + rows + [r'\end{bmatrix}'])
    return (np_to_latex,)


if __name__ == "__main__":
    app.run()
