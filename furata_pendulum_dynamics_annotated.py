# Annotated copy of furata_pendulum_dynamics.py: review notes are rendered as
# red callout boxes (mo.callout) under the relevant cells. Run with `marimo edit`.
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
    mo.callout(mo.md(r"""
## ⚠ Review: why the controller didn't work

The red boxes below each cell explain the problem there. Issues are ranked by impact.

**Model and controller errors (these break the controller):**

1. **The controller balances with a motor that doesn't exist.** LQR and pole placement use both $\tau_1$ (motor) and $\tau_2$ (pendulum disturbance). The $\tau_2$ row of $K$ does almost all the balancing. With only the $\tau_1$ row, the closed loop has a pole at **+20.8 rad/s**, so the pendulum falls in ~50 ms.
2. **$A_{42}$ uses $L_2$ instead of $l_2$.** Gravity on the pendulum is overstated by 2.1×.
3. **The torque→voltage conversion is wrong.** It should be $R/K_t = 82.1$, not 2.61, so the hardware gains are ~31× too small. Back-EMF damping is also missing.
4. **$J_1 = 5.3\times10^{-4}$ kg m² is ~20–50× too big** for a 20 g arm.
5. **$b_1, b_2 = 0.85$ is the ringdown decay rate $q$ [1/s], not a damping coefficient $b$** [N m s/rad]. They're unused anyway, because $A_{33}..A_{44} = 0$.

**Plot and sim bugs** (these hid issue 1, because no plot ever showed the real closed loop):

6. Initial states use $\theta_2=\pi$, but the linear model is upright at $\theta_2 = 0$.
7. `np.linalg.eig` returns `(eigenvalues, eigenvectors)`, not `(T, D)`.
8. The discrete sim overwrites $x(0)$ with zeros on the first iteration.
9. Controllability is checked with both inputs, which proves nothing.
10. Pole placement has the same $\tau_2$ problem, and its poles are very slow.
11. The "closed loop" plot is actually the open-loop plant.
12. The last cell labels the $\tau_1$ output as $\dot\theta_1$.

**Related problems in `mujoco_sim.py`:**

- `abs(x[0]) < 0.2` tests the **arm** angle $\theta_1$, not the pendulum angle $\theta_2$, so control switches off once the arm moves 11°.
- The MuJoCo geometry doesn't match these constants:
    - The pendulum pivot is at 93.5 mm in the XML, not $L_1$ = 38.4 mm.
    - `geom.size[1]` is the cylinder **half**-length, so the rods come out 2× too long.
    - The pendulum CoM is at 64.5 mm, not $l_2$ = 32.8 mm.
    - `body.mass` is ignored, because MuJoCo uses geom density. The arms are actually 6.0 g and 11.0 g.
    - The motor rotor inertia is missing (add it as joint `armature`).
- `control_k = [0.066, 2.299, 0.656, 18.894]` is unstable on MuJoCo's own linearisation (pole at +1.87 rad/s).
"""), kind="danger")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
**What checks out:** With issue 2 fixed and MuJoCo's real geometry plugged in, this paper's $A$ and $B$ match `mujoco.mjd_transitionFD` within ~1%, with the same signs. A $\tau_1$-only LQR designed on that model balanced the MuJoCo pendulum from $\theta_2 = 0.05$ rad, with peak torque 0.125 N m.
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

    def futara_symbolic_system():
        ''' Return the symbolic system for the Furata pendulum, as described in the paper'''
        # First symbols, use dynamicsymbols to get implicit reference to t

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


        u = Matrix([τ_1, τ_2])  # ⚠ ISSUE 1 (see red box)

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
    mo.callout(mo.md(r"""
**ISSUE 1 (root cause):** $\tau_2$ is a *disturbance* torque on the pendulum joint, and you can't command it. It's fine as a model input, but the controller must be designed with `B[:, :1]` (the motor only). See the LQR section.
"""), kind="danger")
    return


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
        J_1: 0.00053, # Rough guess at 10x L2, as solidworks failed to find this OK  # ⚠ ISSUE 4 (see red box)
        J_2: 0.0000053,
        b1: 0.85, # Assume same as measured b2  # ⚠ ISSUE 5 (see red box)
        b2: 0.85 # See pendulum_damping.py for how we measured this
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
    mo.callout(mo.md(r"""
**ISSUE 4: $J_1$ is far too big.** $5.3\times10^{-4}$ is 100× $J_2$, not 10× as the comment says. A 20 g rod ~77 mm long has $J \approx mL^2/12 \approx 1\times10^{-5}$ kg m² about its CoM. The real value should add the GM3506 rotor (~$1\times10^{-5}$) and hub. $J_1$ sets how hard the arm is to accelerate, so this error rescales every $\tau_1$ gain. Measure it, e.g. step the motor with the pendulum removed and fit $J$ from the acceleration.

**ISSUE 5: these damping values have the wrong units.** 0.85 is $q$ from $\ddot\theta = -q\dot\theta - \dots$, in 1/s. The damping coefficient is $b_2 = q\hat J_2 = 0.85 \times 1.605\times10^{-5} = 1.36\times10^{-5}$ N m s/rad. That explains the FIXME in `mujoco_sim.py`: $1\times10^{-5}$ "works" because it's roughly correct. $b_1$ is not "the same as $b_2$". Under voltage drive, the arm is damped mainly by motor back-EMF, $K_t^2/R = 8.3\times10^{-4}$ N m s/rad (issue 3).
"""), kind="danger")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
**$J_2$ and $l_2$ look right.** The ringdown in `pendulum_damping.py` needed $l$ = 49 mm instead of 32.8 mm. That's because the pendulum is a *compound* pendulum: $\omega^2 = m_2 g l_2/\hat J_2$, so its equivalent length is $\hat J_2/(m_2 l_2)$ = **48.95 mm**. The ringdown confirms the SolidWorks values.
"""), kind="success")
    return


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

    J_0_hat = J_1 + m_1 * l_1**2 + m_2 * L_1**2
    mo.output.append(mo.md(f"Total moment of interia of arm 1 when arm 2 is hanging down: $\\hat{{J_0}} =  {latex(J_0_hat)}$"))
    return J_0_hat, J_1_hat, J_2_hat


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
**Minor:** The formula for $\hat J_0$ is correct, but the description isn't quite. It's arm 1 with the pendulum mass lumped at its pivot, and it holds upright as well as hanging down.
"""), kind="warn")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Values of A and B matrices""")
    return


@app.cell
def _(J_0_hat, J_2_hat, L_1, L_2, Matrix, g, l_2, m_2):
    A_31 = 0
    A_32 = (g*m_2**2 * l_2**2 * L_1) / (J_0_hat * J_2_hat - m_2**2 * L_1**2 * l_2**2)
    A_33 = 0 # As friction b_1 = 0, we can simplify  # ⚠ ISSUE 5 (see red box)
    A_34 = 0 # As friction b_2 = 0, we can simplify

    A_41 = 0
    A_42 = (g*m_2*L_2*J_0_hat) / (J_0_hat * J_2_hat - m_2**2 * L_1**2 * l_2**2)  # ⚠ ISSUE 2: L_2 -> l_2 (see red box)
    A_43 = 0 # As friction b_1 = 0, we can simplify
    A_44 = 0 # As friction b_2 = 0, we can simplify

    A = Matrix([
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [A_31, A_32, A_33, A_34],
        [A_41, A_42, A_43, A_44]
    ])

    # The paper switches the order it lists these between A and B, watch out when comparing
    B_31 = J_2_hat / (J_0_hat*J_2_hat - m_2**2 * L_1**2 * l_2**2)
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
    mo.callout(mo.md(r"""
**ISSUE 2: $A_{42}$ should use $l_2$ (pivot→CoM, 32.8 mm), not $L_2$ (full length, 70.3 mm).** The paper gives $A_{42} = g m_2 l_2 \hat J_0 / \Delta$, where $\Delta = \hat J_0 \hat J_2 - m_2^2L_1^2l_2^2$. As written, the unstable pole is +20.9 rad/s instead of +14.3 rad/s.

**ISSUE 5 (cont.):** $b_1, b_2$ are not zero, so the damped terms should be included:
$A_{33} = -b_1\hat J_2/\Delta$, $A_{34} = -b_2 m_2 l_2 L_1/\Delta$, $A_{43} = -b_1 m_2 l_2 L_1/\Delta$, $A_{44} = -b_2\hat J_0/\Delta$.
For voltage drive, $b_1$ should include the back-EMF damping $K_t^2/R$.
"""), kind="danger")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
**$B$ is correct, including signs.** With MuJoCo's geometry it matches MuJoCo's linearisation within ~1%: $+\tau_1$ gives $+\ddot\theta_2$ in both.
"""), kind="success")
    return


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

    # Equivalent to matlab eig(), see https://stackoverflow.com/a/51248586
    T, D = np.linalg.eig(A_sub)  # ⚠ ISSUE 7 (see red box)

    mo.md(f"""
    Substituting `constant_values` in: $$A={np_to_latex(A_sub)}$$

    Giving eigenvectors and esigenvalues `T, D = np.linalg.eig(A_sub)`:

    **FIXME: Why is T not square?**

    $$T = {np_to_latex(T)}   \\quad D={np_to_latex(D)}$$

    """)
    return A_sub, D, T


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
**ISSUE 7: wrong return order.** `np.linalg.eig` returns `(eigenvalues, eigenvectors)`, the opposite of MATLAB's `[V, D] = eig(A)`. So "T" here is the 1-D eigenvalue vector, which answers the FIXME. Use `D, T = np.linalg.eig(A_sub)`.

Even then, $T e^{Dt} T^{-1}$ fails for this $A$. The double eigenvalue at 0 has only one eigenvector, so $T$ is singular (rank 3). Use `scipy.linalg.expm(A_sub * t)` directly.
"""), kind="danger")
    return


@app.cell
def _(D, T, expm, np, plt):
    x_0 = [0, np.pi, 1E-6, 1E-6] # Pointing straight up, no velocity  # ⚠ ISSUE 6 (see red box)

    _t = np.linspace(0, 0.1, 1000)

    # Simulate the system
    # Fixme, why does this not broadcast across, and so we have to use a loop?
    _x_sim = np.zeros((len(_t), 4))
    for _ts, i in enumerate(_t):  # ⚠ ISSUE 7 (see red box)
        print(D)
        _x_sim[_ts:] = T @ expm(D*_ts) @ np.linalg.inv(T) @ x_0

    print(_x_sim[:10])
    _fig, _ax = plt.subplots()

    _ax.plot(_t, _x_sim[:,0], label="Theta 1")
    _ax.plot(_t, _x_sim[:,1], label="Theta 2")

    plt.show()
    return i, x_0


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
**ISSUE 6: wrong initial angle.** The linear model is in deviation coordinates about **upright**, where $\theta_2 = 0$. $A_{42}>0$ shows this is the unstable equilibrium, and MuJoCo is also upright at 0. $\theta_2 = \pi$ means "180° off upright", which the linear model can't represent. Use e.g. `[0, 0.05, 0, 0]`.

**ISSUE 7 (cont.): loop bugs.**

- `enumerate()` yields `(index, value)`, so `_ts` is the index and `expm(D*_ts)` uses 0, 1, 2… **seconds**.
- `expm` needs a matrix, `np.diag(D)`.
- `_x_sim[_ts:]` overwrites every following row; you want `_x_sim[_ts]`.
- `print(D)` runs 1000 times.

Simpler fix: `np.array([expm(A_sub*tt) @ x_0 for tt in _t])`, or `control.initial_response(...)`.
"""), kind="danger")
    return


@app.cell
def _(A_matrix, expm, np, plt):
    # Discretise the system matrix
    _dt = 1E-3
    A_d = expm(A_matrix*_dt)


    # Now simulate
    _t = np.arange(0, 1, _dt)

    _x_sim = np.zeros((len(_t), 4)) # Theta 1, Theta 2, Theta 1 dot, Theta 2 dot
    _x_sim[0] = [0, np.pi, 0.01, 0.01] # Pointing straight up, small velocity
    for _i, _ts in enumerate(_t):
        _x_sim[_i:] = A_d @ _x_sim[_i-1]  # ⚠ ISSUE 8 (see red box)

    #Plot
    _fig, _ax = plt.subplots()
    _ax.plot(_t, _x_sim[:,0], label="Theta 1")
    _ax.plot(_t, _x_sim[:,1], label="Theta 2")
    _ax.legend()
    return (A_d,)


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
**ISSUE 8: this sim is all zeros.** On the first pass `_i = 0`, so it reads `_x_sim[-1]` (the last row, zeros). The `[_i:]` slice then writes zeros over $x(0)$. Use:
```python
for _i in range(1, len(_t)):
    _x_sim[_i] = A_d @ _x_sim[_i-1]
```
Also, `A_matrix` comes from a later cell, where it's built without `dtype=float` (an object array). Use `A_sub` instead. Same $\theta_2=\pi$ problem as issue 6.
"""), kind="danger")
    return


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

    f"{np.linalg.matrix_rank(control.ctrb(A_sub, B_sub))=}"  # ⚠ ISSUE 9 (see red box)
    return (B_sub,)


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
**ISSUE 9: this check proves nothing.** With both inputs, rank 4 is trivial because $\tau_2$ drives the pendulum directly. The useful question is whether the system is controllable from the motor alone: `control.ctrb(A_sub, B_sub[:, :1])`. It is (rank 4), so a working controller exists.
"""), kind="danger")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## Make Pole Placement Controller""")
    return


@app.cell
def _(A_sub, B_sub, control, mo, np, np_to_latex):
    K = control.place(A_sub, B_sub, [-1.1, -1.2, -1.3, -1.4])  # ⚠ ISSUE 10 (see red box)


    mo.md(f"""
    Using `control.place(A,B,desired_poles)`

    Control matrix $K={np_to_latex(K.T,dp=4)}$

    Eigenvalues of system are now: {np.linalg.eig(A_sub - B_sub @ K)[0]}
    """)
    return (K,)


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
**ISSUE 10: same problem as issue 1.** This $K$ is 2×4 and assumes $\tau_2$ can be applied. With only the $\tau_1$ row, the closed-loop eigenvalues are $[-21.1, -1.15, -1.33, \mathbf{+21.0}]$, so it's unstable. Use `B_sub[:, :1]`.

Poles at ~−1 rad/s are also ~15× slower than the open-loop unstable pole (+14 rad/s once issue 2 is fixed). That's a fragile design: it needs very precise gains and gives a large arm swing. Place poles beyond ~|14| rad/s, or use LQR.
"""), kind="danger")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""Plot startup behaviour. Should ideally return to zero!""")
    return


@app.cell
def _(A, B, K, constant_values, control, mo, np, plt, sympy):
    A_matrix = sympy.matrix2numpy(A.subs(constant_values))
    B_matrix = sympy.matrix2numpy(B.subs(constant_values))

    C_matrix = [0, 0, 0, 0] # Control matrix. TODO: What is order?
    D_matrix = [0, 0] # Feed through, zero

    statespace = control.StateSpace(A_matrix, B_matrix, K[1], D_matrix)  # ⚠ ISSUE 11 (see red box)

    _resp = control.initial_response(statespace, T=0.1, X0=[0, np.pi, 0, 0]) # Simulate one second of time

    plt.plot(_resp.t, _resp.x[2], label="$\\theta_1$")
    plt.plot(_resp.t, _resp.x[3], label="$\\theta_2$")
    plt.ylabel("Angle (rad)")
    plt.xlabel("Time (s)")
    plt.legend()
    mo.mpl.interactive(plt.gcf())
    return A_matrix, B_matrix, C_matrix, D_matrix, statespace


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
**ISSUE 11: this never simulates the closed loop.**

- `StateSpace(A, B, K[1], D)` is the **open-loop** plant, whose output is the $\tau_2$ row of $K$ applied to $x$. The closed loop is $A - BK$ with the $\tau_1$-only $K$: `control.ss(A_sub - B_sub[:, :1] @ K1, B_sub[:, :1], np.eye(4), 0)`.
- `C` is the **output** matrix ($y = Cx$), not a control matrix. Use `np.eye(4)` for full state output.
- `x[2]` and `x[3]` are **velocities**, but they're labelled $\theta_1, \theta_2$. The angles are `x[0]` and `x[1]`.
- `T=0.1` is 0.1 s, not one second. `X0` has $\theta_2=\pi$ (issue 6).
- `matrix2numpy` without `dtype=float` gives object arrays.
"""), kind="danger")
    return


@app.cell
def _(control, plt, statespace):
    _resp = control.frequency_response(statespace)
    _resp.plot()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
**ISSUE 11 (cont.):** This is the Bode plot of the unstable open-loop plant, seen through the $\tau_2$ row of $K$, so it tells you little. For stability margins, use the loop gain $K_1(sI-A)^{-1}B_{:,1}$.
"""), kind="danger")
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

    K_lqr = control.lqr(A_sub, B_sub, Q, R)[0]  # ⚠ ISSUE 1 (see red box)

    mo.md(f"""$K_{{LQR}} = {np_to_latex(K_lqr.T)}$

    First column is output $\\tau_1$, second is $\\tau_2$
    """)
    return C, K_lqr, Q, R, brysons_rule


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
**ISSUE 1: this is where it bites.** With both inputs, $K_{LQR}$ is roughly:

| row | $\theta_1$ | $\theta_2$ | $\dot\theta_1$ | $\dot\theta_2$ |
|---|---|---|---|---|
| $\tau_1$ | 100 | 0.65 | 707 | 6.5 |
| $\tau_2$ | −0.65 | **100** | −4.6 | **1000** |

The pendulum is balanced almost entirely by $\tau_2$, a motor on the pendulum joint that doesn't exist. Keeping only the $\tau_1$ row gives closed-loop eigenvalues $[-1.3\times10^6, -0.14, \mathbf{+20.75}, -20.75]$, so it's unstable.

**Fix:** design with the real actuator only:
`K_lqr = control.lqr(A_sub, B_sub[:, :1], Q, R[:1, :1])[0]`. That gives large $\theta_2$ gains, as expected.

Also:

- $R = 10^{-4}$ is arbitrary. Use Bryson's rule with $u_{max}$ set to the real limit, ~0.15 N m at 12 V stall (issue 3). MuJoCo's ±0.4 N m is optimistic.
- Getting $J_1$ right (issue 4) matters more than tuning $Q$ and $R$.
"""), kind="danger")
    return


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
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
**ISSUE 3: the conversion above is wrong.**

- $K_v$ = 140 RPM/V = $140 \cdot 2\pi/60$ = 14.66 rad/s/V. The text writes "$=2\pi$", then 14.66, then divides 6.28 by $R$.
- 6.28 / 5.6 = 1.12, not 2.61.
- The constant that matters is $K_t = 1/K_v$ = 0.0682 N m/A. At stall, $\tau = (K_t/R)V$ = 0.0122 N m/V, so $V = 82.1\,\tau$.

So voltage gains are $K_V = 82.1\,K_\tau$. The code uses $2.61\,K_\tau$, which is **~31× too small**. That alone explains the hardware failing.

Back-EMF isn't small either, because the arm flicks fast: $\tau_1 = (K_t/R)V - (K_t^2/R)\dot\theta_1$. That's an extra damping of 8.3×10⁻⁴ N m s/rad on $\theta_1$. Model it directly in volts: add $K_t^2/R$ to $b_1$ and scale `B[:, 0]` by $K_t/R$. LQR then gives volt gains directly.
"""), kind="danger")
    return


@app.cell
def _(K_lqr, Matrix, latex, mo, np, np_to_latex, x):
    _k_symbolic = Matrix(np.round(K_lqr*2.61, 3))  # ⚠ ISSUE 3/12 (see red box)
    mo.md(f"""Gains $K_{{LQR}} = {np_to_latex(K_lqr.T*2.61)}$

    Ignoring the second column as its usefully very small (we can't set $\\theta_2$ anyway)
    $$\\dot\\theta_1 = {latex((_k_symbolic * x)[0])}$$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
**ISSUE 12:**

- The factor should be 82.1, not 2.61 (issue 3).
- The "second column" is the $\tau_2$ row. It isn't "usefully very small": it's the part that balances the pendulum (issue 1).
- The expression shown is $\tau_1$ (or $V$), not $\dot\theta_1$. The control law is $u = -Kx$, so the minus sign is missing.
"""), kind="danger")
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


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
**Minor:** For complex arrays, `str()` prints `1.+2.j`, which isn't valid LaTeX. Handle complex values with `a.real` and `a.imag`. (Also, "Furata" → "Furuta" throughout.)
"""), kind="warn")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
