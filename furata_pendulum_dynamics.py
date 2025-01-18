# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "control==0.10.1",
#     "marimo",
#     "matplotlib==3.10.0",
#     "numpy==2.2.1",
#     "sympy==1.13.3",
# ]
# ///

import marimo

__generated_with = "0.10.14"
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
        b1: 1E-6, # Assume no friction
        b2: 1E-6 # Assume no friction
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


@app.cell
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
    mo.md("""## Simulate this""")
    return


@app.cell
def _(A, B, Eq, constant_values, diff, u, x, τ_1, τ_2):
    # For the easiest case tau=0

    eqn = Eq(diff(x), (A*x + B*u).subs(constant_values).subs({τ_1:0, τ_2:0}))
    eqn
    return (eqn,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Now check for stability

        Using the real values of the constants
        """
    )
    return


@app.cell
def _(A, constant_values, sympy):
    # First, with no input (u = 0), do we get any positive eigenvalues? If so, the system is unstable
    sympy.matrix2numpy(A.subs(constant_values))

    return


@app.cell
def _(A, B, Matrix, constant_values):
    # Now lets try a control matrix K
    # TODO: Need to check which way round this is!
    K = Matrix([
        [0, 0, 0, 0],
        [0, 0.01, 1, 0.05], # Tau 2 (disturbance torque), we can't apply so set to zero
    ])

    (A - B*K).subs(constant_values).eigenvals()

    return (K,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Make LQR controller with Python Control

        FIXME: Doesn't seem to do anything, just heads off into infinity
        """
    )
    return


@app.cell
def _():
    import numpy as np
    import control
    from matplotlib import pyplot as plt
    return control, np, plt


@app.cell
def _(A, B, constant_values, control, mo, plt, sympy):
    A_matrix = sympy.matrix2numpy(A.subs(constant_values))
    B_matrix = sympy.matrix2numpy(B.subs(constant_values))

    C_matrix = [0, 0, 0, 0] # Control matrix. TODO: What is order?
    D_matrix = [0, 0] # Feed through, zero

    statespace = control.StateSpace(A_matrix, B_matrix, C_matrix, D_matrix)

    _resp = control.initial_response(statespace, T=0.1, X0=[0, 0.1, 0, 0]) # Simulate one second of time

    plt.plot(_resp.t, _resp.x[1])
    mo.mpl.interactive(plt.gcf())
    return A_matrix, B_matrix, C_matrix, D_matrix, statespace


@app.cell
def _(control, plt, statespace):
    _resp = control.frequency_response(statespace)
    _resp.plot()
    plt.show()
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import sympy
    from sympy import symbols, Integral, Eq, solve, diff, Function, integrate, exp, sqrt, pi, sin, cos, tan, log, diff, simplify, solve, Matrix, S, lambdify
    from sympy.physics.mechanics  import dynamicsymbols, mechanics_printing
    from sympy.solvers.ode.systems import dsolve_system
    from sympy import latex
    from sympy.physics.vector.printing import vlatex
    mechanics_printing()
    return (
        Eq,
        Function,
        Integral,
        Matrix,
        S,
        cos,
        diff,
        dsolve_system,
        dynamicsymbols,
        exp,
        integrate,
        lambdify,
        latex,
        log,
        mechanics_printing,
        mo,
        pi,
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
def _():
    return


if __name__ == "__main__":
    app.run()
