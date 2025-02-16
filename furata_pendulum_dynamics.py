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
        J_2: 0.0000053,
        b1: 0.85, # Assume same as measured b2
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
    mo.md(r"""### Values of A and B matrices""")
    return


@app.cell
def _(J_0_hat, J_2_hat, L_1, L_2, Matrix, g, l_2, m_2):
    A_31 = 0
    A_32 = (g*m_2**2 * l_2**2 * L_1) / (J_0_hat * J_2_hat - m_2**2 * L_1**2 * l_2**2)
    A_33 = 0 # As friction b_1 = 0, we can simplify
    A_34 = 0 # As friction b_2 = 0, we can simplify

    A_41 = 0
    A_42 = (g*m_2*L_2*J_0_hat) / (J_0_hat * J_2_hat - m_2**2 * L_1**2 * l_2**2)
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
    x_0 = [0, np.pi, 1E-6, 1E-6] # Pointing straight up, no velocity

    _t = np.linspace(0, 0.1, 1000)

    # Simulate the system
    # Fixme, why does this not broadcast across, and so we have to use a loop?
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
    _dt = 1E-3
    A_d = expm(A_matrix*_dt)


    # Now simulate
    _t = np.arange(0, 1, _dt)

    _x_sim = np.zeros((len(_t), 4)) # Theta 1, Theta 2, Theta 1 dot, Theta 2 dot
    _x_sim[0] = [0, np.pi, 0.01, 0.01] # Pointing straight up, small velocity
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

    f"{np.linalg.matrix_rank(control.ctrb(A_sub, B_sub))=}"
    return (B_sub,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## Make Pole Placement Controller""")
    return


@app.cell
def _(A_sub, B_sub, control, mo, np, np_to_latex):
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
    return


@app.cell
def _(K_lqr, Matrix, latex, mo, np, np_to_latex, x):
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
