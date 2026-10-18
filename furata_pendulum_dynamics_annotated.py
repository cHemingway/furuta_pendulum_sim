# =============================================================================
# ANNOTATED REVIEW of furata_pendulum_dynamics.py
#
# Search for "# ISSUE" to find each note in the code below. Numbers are ranked
# by how much each one breaks the controller.
#
# Why the controller failed (sim and hardware):
#   ISSUE 1  The controller is designed with TWO inputs (tau_1 motor, tau_2
#            pendulum disturbance). The tau_2 row of K does nearly all the
#            balancing. Without it the closed loop has an unstable pole at
#            +20.8 rad/s, so the pendulum falls in about 50 ms. There is no
#            motor on the pendulum joint.
#   ISSUE 2  A_42 uses L_2 (full pendulum length) instead of l_2 (pivot to
#            centre of mass). This overstates gravity by 2.1x.
#   ISSUE 3  The torque->voltage conversion is wrong: it uses 2.61 where it
#            should use R/K_t = 82.1. Hardware gains end up about 31x too
#            small, and back-EMF damping is left out.
#   ISSUE 4  J_1 = 5.3e-4 kg m^2 is about 20-50x too big for a 20 g arm of
#            ~77 mm (expected ~1e-5 + motor rotor).
#   ISSUE 5  b1/b2 = 0.85 is the ringdown decay rate q [1/s], not a damping
#            coefficient b [N m s/rad]. It's unused anyway because A_33..A_44
#            are zeroed.
#
# Bugs in the plots/sims (these hid ISSUE 1, because none of them ever showed
# the real closed loop):
#   ISSUE 6  Initial states use theta_2 = pi; the linear model is upright at 0.
#   ISSUE 7  np.linalg.eig returns (eigenvalues, eigenvectors), not (T, D).
#   ISSUE 8  Discrete sim overwrites x(0) with zeros on the first iteration.
#   ISSUE 9  Controllability is checked with both inputs, which proves nothing.
#   ISSUE 10 Pole placement has the same tau_2 problem; poles are also very slow.
#   ISSUE 11 The "closed loop" plot is actually the open-loop plant.
#   ISSUE 12 The last cell labels the tau_1 output as theta_1_dot.
#
# Related problems in mujoco_sim.py (also cause the sim to fail):
#   a) `abs(x[0]) < 0.2` tests the ARM angle theta_1, not the pendulum
#      theta_2, so control switches off once the arm moves 11 degrees.
#   b) The MuJoCo geometry doesn't match these constants:
#        - The pendulum pivot is at 93.5 mm in the XML, not L_1 = 38.4 mm.
#        - geom size[1] is the cylinder HALF-length, so setting it to L_1/L_2
#          doubles the rod lengths.
#        - The pendulum CoM is at 64.5 mm, not l_2 = 32.8 mm.
#        - body.mass is ignored: MuJoCo derives mass from geom density, so
#          the arms are actually 6.0 g and 11.0 g.
#        - Motor rotor inertia isn't modelled (use joint `armature`).
#   c) control_k = [0.066, 2.299, 0.656, 18.894] is unstable on MuJoCo's own
#      linearisation (pole at +1.87 rad/s).
#
# Checks I ran:
#   - With ISSUE 2 fixed and MuJoCo's real geometry plugged in, the paper's
#     A/B match mujoco.mjd_transitionFD within ~1%, with the same signs. So
#     the paper's equations and the sign convention are fine.
#   - A tau_1-only LQR designed on that model balances the MuJoCo pendulum
#     from theta_2 = 0.05 rad. It settles within 5 s, with peak torque
#     0.125 N m.
# =============================================================================
import marimo

__generated_with = "0.11.5"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Symbolic State Space model of Furata Pendulum

        We wish to create a state space model of our Furata pendulum, linearised around the vertical and simplified.

        We wish to do this _symbolically_, so that we can analyse sensitivities, rearrange etc, and just sanity check (by printing it out) that it matches the paper "On the Dynamics of the Furata Pendulum"[^1]


        [^1]:  Cazzolato, Benjamin Seth, Prime, Zebb, On the Dynamics of the Furuta Pendulum, Journal of Control Science and Engineering, 2011, 528341, 8 pages, 2011. https://doi.org/10.1155/2011/528341
        """
    )
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

    def futara_symbolic_system():
        ''' Return the symbolic system for the Furata pendulum, as described in the paper'''
        # First symbols, use dynamicsymbols to get implicit reference to t

        # (Cosmetic: "Furata"/"futara" -> "Furuta" throughout.)
        A31, A32, A33, A34 = symbols('A_31 A_32 A_33 A_34')
        A41, A42, A43, A44 = symbols('A_41 A_42 A_43 A_44')

        B31, B32, B41, B42 = symbols('B_31 B_32 B_41 B_42')

        # Now create the matrices from the symbols
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


        # ISSUE 1 (root): tau_2 is a *disturbance* on the pendulum joint, and
        # you can't command it. Using it here as a model input is fine, but the
        # controller must be designed with B[:, 0] only (see the LQR cell).
        u = Matrix([τ_1, τ_2])

        return A, x, x_dot, B, u


    _A, x, _x_dot, _B, u = futara_symbolic_system()

    # Printing using vlatex gives us dot notation for derivatives as we used dynamicsymbol
    mo.output.append("For our furata pendulum, these are given by: ")
    mo.output.append(
        mo.hstack([
            mo.md(f"$$\\dot x= {vlatex(_x_dot)}$$"),  # TODO why does extra bracket appear on x_dot?
            mo.md(f"$A= {vlatex(_A)}$"),
            mo.md(f"State $x= {vlatex(x)}$"),
            mo.md(f"$B= {vlatex(_B)}$"),
            mo.md(f"Input $u= {vlatex(u)}$")
        ],
        justify="center", gap=4, align="center")
    )

    mo.output.append(mo.md(f"""With scalar variables:

    - $\\theta_1$, $\\theta_2$ are the angles of the first arm (motor) and second arm (pendulum)
    - $\\tau_1$, $\\tau_2$ are the torques applied to the first arm (from motor) and second arm (disturbance torque)
    """))

    mo.output.append("Substituting in to our original equation:")

    # Now we can write out the equation in full
    mo.output.append(mo.md(f"$${vlatex(_x_dot)}= {vlatex(_A)}{vlatex(x)} + {vlatex(_B)}{vlatex(u)}$$"))
    return futara_symbolic_system, t, u, x, θ_1, θ_2, τ_1, τ_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Values of constants $A, B$""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""### Physical parameters""")
    return


@app.cell
def _(Eq, mo, symbols):
    # Define the constants
    g = symbols('g') # Gravity
    g_eq = Eq(g, 9.81, simplify=False)

    m_1, m_2 = symbols('m_1 m_2') # Mass of each arm
    l_1, l_2 = symbols('l_1 l_2') # Lengths to centre of mass for each arm

    L_1, L_2 = symbols('L_1 L_2') # Lengths to _end_ for each arm

    J_1 = symbols('J_1') # Inertia of first arm, neglibible in axis, equal in y and z
    J_2 = symbols('J_2') # Inertia of second arm, neglibible in axis, equal in y and z

    b1, b2 = symbols("b_1 b_2") # Viscous damping coefficient

    constant_values = {
        g: 9.81,
        m_1: 0.02, # Guessing 50% density of ABS for 3d prints, plus 1g * 11 screws
        m_2: 0.01, # From solidworks
        l_1: 0.02, # Bit of a guess here
        l_2: 0.03279,
        L_1: 0.0384, #  From centre to centre in solidworks
        L_2: 0.0703,  # From axis to end face in solidworks
        J_1: 0.00053, # Rough guess at 10x L2, as solidworks failed to find this OK
        # ISSUE 4: This is 100x J_2, not 10x. A 20 g rod ~77 mm long has
        # J about its CoM of m*L^2/12 ~= 1e-5 kg m^2, so 5.3e-4 is 20-50x too
        # big. The real number should include the GM3506 rotor (~1e-5) and the
        # hub. J_1 sets how hard the arm is to accelerate, so this error
        # rescales every tau_1 gain. Measure it, e.g. step the motor
        # with the pendulum removed and fit J from the acceleration.
        J_2: 0.0000053,
        # NOTE: J_2 looks right. Your ringdown in pendulum_damping.py needed
        # l = 49 mm instead of 32.8 mm. That's because the pendulum is a
        # compound pendulum, w^2 = m_2 g l_2 / J_2_hat, so its equivalent
        # length is J_2_hat/(m_2 l_2) = 48.95 mm. The ringdown confirms the
        # SolidWorks J_2 and l_2.
        b1: 0.85, # Assume same as measured b2
        b2: 0.85 # See pendulum_damping.py for how we measured this
        # ISSUE 5: 0.85 is q, the decay rate in theta'' = -q theta' - ...,
        # with units of 1/s. The damping coefficient you need here is
        # b2 = q * J_2_hat = 0.85 * 1.605e-5 = 1.36e-5 N m s/rad. That also
        # explains the FIXME in mujoco_sim.py: 1e-5 "works" because it's
        # roughly the right value.
        # b1 is not "the same as b2". With voltage drive, the arm is damped
        # mainly by motor back-EMF, K_t^2/R = 8.3e-4 N m s/rad (see ISSUE 3).
        # Also, these values are never used, because A_33..A_44 are
        # hard-coded to 0 below.
    }


    # Now print these out
    mo.hstack(
        [mo.md(f"${k}={v}$") for k, v in constant_values.items()], 
        gap=0, align="center", wrap=True
    )
    return (
        J_1,
        J_2,
        L_1,
        L_2,
        b1,
        b2,
        constant_values,
        g,
        g_eq,
        l_1,
        l_2,
        m_1,
        m_2,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""### Simplified inertia values""")
    return


@app.cell(hide_code=True)
def _(J_1, J_2, L_1, l_1, l_2, latex, m_1, m_2, mo):
    J_1_hat = J_1 + m_1*l_1**2
    mo.output.append(mo.md(f"Total moment of interia of arm 1 about pivot point: $\\hat{{J_1}} = {latex(J_1_hat)}$"))

    J_2_hat = J_2 + m_2*l_2**2
    mo.output.append(mo.md(f"Total moment of interia of arm 2 about pivot point: $\\hat{{J_2}} = {latex(J_2_hat)}$"))

    # Formula is correct (J_0_hat = J_1_hat + m_2 L_1^2). The description is
    # slightly off: this is arm 1 with the pendulum mass lumped at its pivot,
    # and it holds upright as well as hanging down.
    J_0_hat = J_1 + m_1 * l_1**2 + m_2 * L_1**2
    mo.output.append(mo.md(f"Total moment of interia of arm 1 when arm 2 is hanging down: $\\hat{{J_0}} =  {latex(J_0_hat)}$"))
    return J_0_hat, J_1_hat, J_2_hat


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Values of A and B matrices""")
    return


@app.cell
def _(J_0_hat, J_2_hat, L_1, L_2, Matrix, g, l_2, m_2):
    A_31 = 0
    A_32 = (g*m_2**2 * l_2**2 * L_1) / (J_0_hat * J_2_hat - m_2**2 * L_1**2 * l_2**2)
    # ISSUE 5: b_1, b_2 are NOT zero (constant_values says 0.85). The paper's
    # damped terms are (den = J_0_hat*J_2_hat - m_2^2 L_1^2 l_2^2):
    #   A_33 = -b_1*J_2_hat/den      A_34 = -b_2*m_2*l_2*L_1/den
    #   A_43 = -b_1*m_2*l_2*L_1/den  A_44 = -b_2*J_0_hat/den
    # For voltage drive, b_1 should include back-EMF damping K_t^2/R.
    A_33 = 0 # As friction b_1 = 0, we can simplify
    A_34 = 0 # As friction b_2 = 0, we can simplify

    A_41 = 0
    # ISSUE 2: Should be l_2 (pivot -> CoM, 32.8 mm), not L_2 (full length,
    # 70.3 mm). Paper: A_42 = g*m_2*l_2*J_0_hat/den. As written, the
    # unstable pole is +20.9 rad/s instead of +14.3 rad/s.
    #   A_42 = (g*m_2*l_2*J_0_hat) / (...)
    A_42 = (g*m_2*L_2*J_0_hat) / (J_0_hat * J_2_hat - m_2**2 * L_1**2 * l_2**2)
    A_43 = 0 # As friction b_1 = 0, we can simplify
    A_44 = 0 # As friction b_2 = 0, we can simplify

    A = Matrix([
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [A_31, A_32, A_33, A_34],
        [A_41, A_42, A_43, A_44]
    ])

    # B entries and signs are correct. With MuJoCo's geometry they match
    # MuJoCo's linearisation within ~1%: +tau_1 gives +theta_2'' in both.
    # The paper switches the order it lists these between A and B, watch out when comparing
    B_31 =J_2_hat / (J_0_hat*J_2_hat - m_2**2 * L_1**2 * l_2**2)
    B_41 = (m_2 * L_1 * l_2) / (J_0_hat*J_2_hat - m_2**2 * L_1**2 * l_2**2)
    B_32 = (m_2 * L_1 * l_2) / (J_0_hat*J_2_hat - m_2**2 * L_1**2 * l_2**2)
    B_42 = J_0_hat / (J_0_hat*J_2_hat - m_2**2 * L_1**2 * l_2**2)

    B = Matrix([
        [0,   0],
        [0,   0],
        [B_31, B_32],
        [B_41, B_42]
    ])


    A, B
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
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Simulate open loop (no control)

        Our open loop equation is:
        $$\\dot{x} = \mathbf{A}x$$

        If $\mathbf{A}$ has eignenvectors $\\mathbf{T}$ and eigenvalues $\\mathbf{D}$

        $$ x(t) = \mathbf{T}e^{\mathbf{D}t} \mathbf{T}^{-1} x(0) $$
        """
    )
    return


@app.cell(hide_code=True)
def _(A, constant_values, mo, np, np_to_latex, sympy):
    A_sub = sympy.matrix2numpy(A.subs(constant_values), dtype=float) # Note need to set dtype to float for eig to work

    # ISSUE 7: numpy returns (eigenvalues, eigenvectors), the opposite order
    # to MATLAB's [V, D] = eig(A). So "T" here is the 1-D eigenvalue vector,
    # which answers the FIXME below about why T isn't square. Should be:
    #   D, T = np.linalg.eig(A_sub)   # D: eigenvalues (1-D), T: eigenvectors
    # Note: this A has a double eigenvalue at 0 with only one eigenvector,
    # so T is singular and T e^{Dt} T^-1 can't work here anyway.
    # Use scipy expm(A*t) directly.
    # Equivalent to matlab eig(), see https://stackoverflow.com/a/51248586
    T, D = np.linalg.eig(A_sub)

    mo.md(f"""
    Substituting `constant_values` in: $$A={np_to_latex(A_sub)}$$

    Giving eigenvectors and esigenvalues `T, D = np.linalg.eig(A_sub)`:

    **FIXME: Why is T not square?**

    $$T = {np_to_latex(T)}   \\quad D={np_to_latex(D)}$$

    """)
    return A_sub, D, T


@app.cell
def _(D, T, expm, np, plt):
    # ISSUE 6: The linear model is in deviation coordinates about UPRIGHT,
    # where theta_2 = 0. A_42 > 0 shows this is the unstable equilibrium,
    # and MuJoCo's model is also upright at 0. theta_2 = pi here means
    # "180 deg off upright", which the linear model can't represent.
    # Use e.g. [0, 0.05, 0, 0].
    x_0 = [0, np.pi, 1E-6, 1E-6] # Pointing straight up, no velocity

    _t = np.linspace(0, 0.1, 1000)

    # Simulate the system
    # Fixme, why does this not broadcast across, and so we have to use a loop?
    # ISSUE 7 (cont):
    #  - enumerate() yields (index, value), so _ts is the INDEX and i is the
    #    time. expm(D*_ts) therefore uses 0,1,2... seconds, not _t.
    #  - expm needs a matrix: np.diag(D) (with the eig order fixed).
    #  - _x_sim[_ts:] (with colon) overwrites every row from _ts onward;
    #    you want _x_sim[_ts].
    #  - print(D) inside the loop prints 1000 times.
    # Vectorised alternative (also answers the broadcast FIXME):
    #   _x_sim = np.array([expm(A_sub*tt) @ x_0 for tt in _t])
    # or: control.initial_response(control.ss(A_sub, B_sub, np.eye(4), 0), _t, x_0)
    _x_sim = np.zeros((len(_t), 4))
    for _ts, i in enumerate(_t):
        print(D)
        _x_sim[_ts:] = T @ expm(D*_ts) @ np.linalg.inv(T) @ x_0

    print(_x_sim[:10])
    _fig, _ax = plt.subplots()

    _ax.plot(_t, _x_sim[:,0], label="Theta 1")
    _ax.plot(_t, _x_sim[:,1], label="Theta 2")

    plt.show()
    return i, x_0


@app.cell
def _(A_matrix, expm, np, plt):
    # Discretise the system matrix
    # ISSUE 8: A_matrix comes from the "startup behaviour" cell further down,
    # where it's built without dtype=float, so it's an object array. Use
    # A_sub instead.
    _dt = 1E-3
    A_d = expm(A_matrix*_dt)


    # Now simulate
    _t = np.arange(0, 1, _dt)

    _x_sim = np.zeros((len(_t), 4)) # Theta 1, Theta 2, Theta 1 dot, Theta 2 dot
    _x_sim[0] = [0, np.pi, 0.01, 0.01] # Pointing straight up, small velocity  (ISSUE 6: pi -> small angle)
    # ISSUE 8: On the first pass _i = 0, so this reads _x_sim[-1] (the last
    # row, all zeros) and the [_i:] slice writes zeros over the initial
    # state. The whole trajectory comes out as zero. Should be:
    #   for _i in range(1, len(_t)):
    #       _x_sim[_i] = A_d @ _x_sim[_i-1]
    for _i, _ts in enumerate(_t):
        _x_sim[_i:] = A_d @ _x_sim[_i-1]

    #Plot
    _fig, _ax = plt.subplots()
    _ax.plot(_t, _x_sim[:,0], label="Theta 1")
    _ax.plot(_t, _x_sim[:,1], label="Theta 2")
    _ax.legend()
    return (A_d,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Check for Controlability

        Using the real values of the constants into A and B, do we get a rank of 4 (size of the state space)?
        """
    )
    return


@app.cell
def _(A_sub, B, constant_values, control, np, sympy):
    B_sub = sympy.matrix2numpy(B.subs(constant_values),dtype=float)

    # ISSUE 9: With both inputs, rank 4 is trivial because tau_2 drives the
    # pendulum directly. The useful question is whether the system is
    # controllable from the motor alone:
    #   np.linalg.matrix_rank(control.ctrb(A_sub, B_sub[:, :1]))  # -> 4, it is
    f"{np.linalg.matrix_rank(control.ctrb(A_sub, B_sub))=}"
    return (B_sub,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## Make Pole Placement Controller""")
    return


@app.cell
def _(A_sub, B_sub, control, mo, np, np_to_latex):
    # ISSUE 10: Same problem as ISSUE 1. This K is 2x4 and assumes tau_2 can
    # be applied. Using only the tau_1 row, the closed-loop eigenvalues are
    # [-21.1, -1.15, -1.33, +21.0], so it's unstable. Use B_sub[:, :1].
    # Also, poles at ~-1 rad/s are about 15x slower than the open-loop
    # unstable pole (+14 rad/s with ISSUE 2 fixed). That's a fragile design:
    # it needs very precise gains and gives a large arm swing. Start with
    # poles a bit faster than ~|14| rad/s, or just use LQR.
    K = control.place(A_sub, B_sub, [-1.1, -1.2, -1.3, -1.4])


    mo.md(f"""
    Using `control.place(A,B,desired_poles)`

    Control matrix $K={np_to_latex(K.T,dp=4)}$

    Eigenvalues of system are now: {np.linalg.eig(A_sub - B_sub @ K)[0]}
    """)
    return (K,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""Plot startup behaviour. Should ideally return to zero!""")
    return


@app.cell
def _(A, B, K, constant_values, control, mo, np, plt, sympy):
    # ISSUE 11: This never simulates the closed loop.
    #  - matrix2numpy without dtype=float gives object arrays (see ISSUE 8).
    #  - C is the OUTPUT matrix (y = Cx), not a control matrix. For full
    #    state output, use np.eye(4).
    #  - StateSpace(A, B, K[1], D) is the OPEN-LOOP plant, with output = the
    #    tau_2 row of K applied to x. The closed loop is A - B K, using the
    #    tau_1-only K (ISSUE 1):
    #      cl = control.ss(A_sub - B_sub[:, :1] @ K1, B_sub[:, :1], np.eye(4), 0)
    #  - X0 has theta_2 = pi (ISSUE 6), and T=0.1 is 0.1 s, not one second.
    #  - x[2], x[3] are the VELOCITIES, but they're labelled theta_1, theta_2.
    #    The angles are x[0], x[1].
    A_matrix = sympy.matrix2numpy(A.subs(constant_values))
    B_matrix = sympy.matrix2numpy(B.subs(constant_values))

    C_matrix = [0, 0, 0, 0] # Control matrix. TODO: What is order?
    D_matrix = [0, 0] # Feed through, zero

    statespace = control.StateSpace(A_matrix, B_matrix, K[1], D_matrix)

    _resp = control.initial_response(statespace, T=0.1, X0=[0, np.pi, 0, 0]) # Simulate one second of time

    plt.plot(_resp.t, _resp.x[2], label="$\\theta_1$")
    plt.plot(_resp.t, _resp.x[3], label="$\\theta_2$")
    plt.ylabel("Angle (rad)")
    plt.xlabel("Time (s)")
    plt.legend()
    mo.mpl.interactive(plt.gcf())
    return A_matrix, B_matrix, C_matrix, D_matrix, statespace


@app.cell
def _(control, plt, statespace):
    # (ISSUE 11 cont.) This is the Bode plot of the unstable open-loop plant
    # seen through the tau_2 row of K, so it tells you little. If you want
    # stability margins, look at the loop gain K1 (sI - A)^-1 B[:, :1].
    _resp = control.frequency_response(statespace)
    _resp.plot()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Make LQR Controller""")
    return


@app.cell
def _(A_sub, B_sub, control, mo, np, np_to_latex):
    C = np.eye(4)

    def brysons_rule(z_max, u_max):
        Q = np.diag(1/np.power(z_max,2))
        R = np.diag(1/np.power(u_max,2))
        return Q,R

    # Q from "Swing-up and LQR stabilization of rotary inverted pendulum", Park,Kim,Lee
    Q = np.diag([1, 1, 50, 100])
    # Todo: Why did we choose this value of R?
    R = np.diag([0.0001] * 2)

    #Q, R = brysons_rule(
    #    z_max = [1, 0.1, 0.1, 0.01], # Penalize error in pendulum angle/velocity most
    #    u_max = [0.2, 0.001] # Max torque is 0.2Nm
    #)

    # ISSUE 1 (where it bites): with both inputs, K_lqr is roughly
    #   tau_1 row: [ 100.0,   0.65,  707.1,   6.54]
    #   tau_2 row: [  -0.65, 100.0,   -4.6, 1000.0]
    # The pendulum is balanced almost entirely by tau_2 (gains 100 and 1000
    # on theta_2, theta_2_dot), i.e. a motor on the pendulum joint that
    # doesn't exist. Keeping only the tau_1 row leaves closed-loop
    # eigenvalues [-1.3e6, -0.14, +20.75, -20.75], so it's unstable.
    # Fix: design with the real actuator only:
    #   K_lqr = control.lqr(A_sub, B_sub[:, :1], Q, R[:1, :1])[0]   # 1x4
    # On this (ISSUE 2-affected) model that gives about
    # [-100, 42247, -718, 2261]: large theta_2 gains, as expected.
    # Also:
    #  - R = 1e-4 is arbitrary. Use the Bryson's rule version below with
    #    u_max equal to the real torque limit. At 12 V stall the motor gives
    #    only ~0.15 N m (ISSUE 3). mujoco's ctrlrange of +-0.4 N m is
    #    optimistic.
    #  - Getting J_1 right (ISSUE 4) matters more than tuning Q/R.
    K_lqr = control.lqr(A_sub, B_sub, Q, R)[0]

    mo.md(f"""$K_{{LQR}} = {np_to_latex(K_lqr.T)}$

    First column is output $\\tau_1$, second is $\\tau_2$
    """)
    return C, K_lqr, Q, R, brysons_rule


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Convert gains back to real units

        Torque is in units of Nm, but our motor is not directly controlled in this.

        We use **voltage** control, so Torque $\\tau= \\frac{K_t}{R}V$

        - For our GM3506 Motor, we have $R=5.6$ and $K_v = 140$ RPM/Volt, from which we can work out $K_t$
        - Convert Kv into SI Units, $K_v = \\frac{60 * 2*\pi}{60} = 2\\pi$ (conveniently) = 14.66
        - Therefore we have scaler $\\frac{K_{v}}{R} = 6.28/5.6 = 2.61 Nm / V$

        Note, this only works as we have limited back EMF from not moving very fast!
        """
    )
    # ISSUE 3: The unit conversion above is wrong in several places:
    #  - K_v = 140 RPM/V = 140 * 2*pi/60 = 14.66 rad/s/V. The text writes
    #    "60*2*pi/60 = 2*pi" but then quotes 14.66, and later divides 6.28
    #    (not 14.66) by R.
    #  - 6.28/5.6 = 1.12, not 2.61.
    #  - The constant that matters is K_t = 1/K_v = 0.0682 N m/A, not K_v.
    #    At stall, tau = (K_t/R) * V = 0.0122 N m/V, so V = tau * R/K_t =
    #    82.1 * tau.
    #  => Voltage gains are K_V = 82.1 * K_tau. The code uses
    #     2.61 * K_tau, about 31x too small. That's enough to explain the
    #     hardware failing even with a good K.
    #  - Back-EMF is not small for a pendulum stabiliser, since the arm
    #    flicks fast. The motor torque is really
    #       tau_1 = (K_t/R) V - (K_t^2/R) theta_1_dot
    #    i.e. extra damping of 8.3e-4 N m s/rad on theta_1. Model it
    #    directly in volts:
    #       b_1 += K_t^2/R   (A_33, A_43 terms)
    #       B[:, 0] *= K_t/R (input becomes volts)
    #    Then LQR gives volt gains directly with no conversion step.
    return


@app.cell
def _(K_lqr, Matrix, latex, mo, np, np_to_latex, x):
    # ISSUE 12:
    #  - 2.61 should be 82.1 (ISSUE 3).
    #  - The "second column" of K_lqr.T is the tau_2 row. It's not "usefully
    #    very small": it's the part that balances the pendulum (ISSUE 1).
    #  - (_k_symbolic * x)[0] is tau_1 (or V), not theta_1_dot, and the
    #    control law is u = -K x (minus sign missing).
    _k_symbolic = Matrix(np.round(K_lqr*2.61, 3))
    mo.md(f"""Gains $K_{{LQR}} = {np_to_latex(K_lqr.T*2.61)}$

    Ignoring the second column as its usefully very small (we can't set $\\theta_2$ anyway)
    $$\\dot\\theta_1 = {latex((_k_symbolic * x)[0])}$$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## Imports and convenience functions""")
    return


@app.cell
def _():
    import marimo as mo
    import sympy
    from sympy import symbols, Integral, Eq, solve, diff, Function, integrate, exp, sqrt, pi, sin, cos, tan, log, diff, simplify, solve, Matrix, S, lambdify, simplify
    from sympy.physics.mechanics  import dynamicsymbols, mechanics_printing
    from sympy.solvers.ode.systems import dsolve_system
    from sympy import latex
    from sympy.physics.vector.printing import vlatex
    from scipy.linalg import expm
    import numpy as np
    import control
    from matplotlib import pyplot as plt
    mechanics_printing()
    return (
        Eq,
        Function,
        Integral,
        Matrix,
        S,
        control,
        cos,
        diff,
        dsolve_system,
        dynamicsymbols,
        exp,
        expm,
        integrate,
        lambdify,
        latex,
        log,
        mechanics_printing,
        mo,
        np,
        pi,
        plt,
        simplify,
        sin,
        solve,
        sqrt,
        symbols,
        sympy,
        tan,
        vlatex,
    )


@app.cell
def _(np):
    def np_to_latex(a, dp=3):
        ''' Display a numpy matrix as latex. Fixme, doesn't show imaginary part?'''
        # (Minor) str() of a complex array prints "1.+2.j"; .split() keeps it
        # but the "j" isn't LaTeX. Handle complex via a.real / a.imag.
        # return latex(sympy.Matrix(a))

        a = a.round(dp)
        if len(a.shape) > 2:
            raise ValueError('bmatrix can at most display two dimensions')

        with np.printoptions(precision=dp, suppress=True, threshold=5):
            lines = str(a).replace('[', '').replace(']', '').splitlines()
        rv = [r'\begin{bmatrix}']
        rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]
        rv +=  [r'\end{bmatrix}']
        return '\n'.join(rv)
    return (np_to_latex,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
