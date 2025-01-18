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
def _(
    Derivative,
    Matrix,
    S,
    dynamicsymbols,
    mo,
    symbolic_system,
    symbols,
    vlatex,
):
    # Write it out in the form x_dot = Ax + Bu
    mo.output.append("The generic state space model is")

    mo.output.append(mo.md(f"$$\dot x = {vlatex(S('A*x + B*u'))}$$"))


    def futara_symbolic_system():
        ''' Return the symbolic system for the Furata pendulum, as described in the paper'''
        # First symbols, use dynamicsymbols to get implicit reference to t
        θ_1, θ_2 = dynamicsymbols('θ_1 θ_2') # Angles (x)
        τ_1, τ_2 = dynamicsymbols('τ_1 τ_2') # Torques (u)
        
        A31, A32, A33, A34 = symbols('A_31 A_32 A_33 A_34')
        A41, A42, A43, A44 = symbols('A_41 A_42 A_43 A_44')
        
        B31, B32, B41, B42 = symbols('B_31 B_32 B_41 B_42')
        
        # Now create the matrices from the symbols
        A = Matrix([[0, 0, 1, 0],
                    [0, 0, 0, 1],
                    [A31, A32, A33, A34],
                    [A41, A42, A43, A44]])
        
        x = Matrix([θ_1, θ_2, Derivative(θ_1), Derivative(θ_2)])
        
        x_dot = Derivative(x)
        
        B = Matrix([[0, 0],
                    [0, 0],
                    [B31, B32],
                    [B41, B42]])
        
        
        u = Matrix([τ_1, τ_2])

        return A, x, x_dot, B, u


    A, x, x_dot, B, u = symbolic_system()

    # Printing using vlatex gives us dot notation for derivatives as we used dynamicsymbol
    mo.output.append("For our furata pendulum, these are given by: ")
    mo.output.append(
        mo.hstack([
            mo.md(f"$$\dot x= {vlatex(x_dot)}$$"),  # TODO why does extra bracket appear on x_dot?
            mo.md(f"$A= {vlatex(A)}$"),
            mo.md(f"State $x= {vlatex(x)}$"),
            mo.md(f"$B= {vlatex(B)}$"),
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
    mo.output.append(mo.md(f"$${vlatex(x_dot)}= {vlatex(A)}{vlatex(x)} + {vlatex(B)}{vlatex(u)}$$"))
    return A, B, futara_symbolic_system, u, x, x_dot


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Values of constants $A, B$""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""### Physical parameters""")
    return


@app.cell
def _(Eq, symbols):
    # Define the constants
    g = symbols('g') # Gravity
    g_eq = Eq(g, 9.81, simplify=False)

    m_1, m_2 = symbols('m_1 m_2') # Mass of each arm
    m_1_eq = Eq(m_1, 0.5, simplify=False)
    m_2_eq = Eq(m_2, 0.5, simplify=False)
    l_1, l_2 = symbols('l_1 l_2') # Lengths to end of arms
    L_1, L_2 = symbols('L_1 L_2') # Lengths to centre of mass for each arm

    J_1 = symbols('J_1') # Inertia of first arm, neglibible in axis, equal in y and z
    J_2 = symbols('J_2') # Inertia of second arm, neglibible in axis, equal in y and z

    g_eq, m_1_eq, m_2_eq
    return J_1, J_2, L_1, L_2, g, g_eq, l_1, l_2, m_1, m_1_eq, m_2, m_2_eq


@app.cell(hide_code=True)
def _(mo):
    mo.md("""### Simplified inertia values""")
    return


@app.cell
def _(Eq, S, latex, mo, symbols):
    J_1_hat = symbols('\\hat{J_1}')
    J_1_hat_eq = Eq(J_1_hat, S('J_1 + m_1*l_1**2'), simplify=False)
    mo.output.append(mo.md(f"Total moment of interia of arm 1 about pivot point: ${latex(J_1_hat_eq)}$"))

    J_2_hat = symbols('\\hat{J_2}')
    J_2_hat_eq = Eq(J_2_hat, S('J_2 + m_2*l_2**2'), simplify=False)
    mo.output.append(mo.md(f"Total moment of interia of arm 2 about pivot point: ${latex(J_2_hat_eq)}$"))

    J_0_hat = symbols('\\hat{J_0}')
    J_0_hat_eq = Eq(J_0_hat, S('J_1 + m_1 * l_1**2 + m_2 * L_1**2 '), simplify=False)
    mo.output.append(mo.md(f"Total moment of interia of arm 1 when arm 2 is hanging down: ${latex(J_0_hat_eq)}$"))

    return J_0_hat, J_0_hat_eq, J_1_hat, J_1_hat_eq, J_2_hat, J_2_hat_eq


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import sympy
    from sympy import symbols, Integral, Eq, solve, diff, Function, Derivative, integrate, exp, sqrt, pi, sin, cos, tan, log, diff, simplify, solve, Matrix, S
    from sympy.physics.mechanics  import dynamicsymbols, mechanics_printing
    from sympy import latex
    from sympy.physics.vector.printing import vlatex
    mechanics_printing()
    return (
        Derivative,
        Eq,
        Function,
        Integral,
        Matrix,
        S,
        cos,
        diff,
        dynamicsymbols,
        exp,
        integrate,
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


if __name__ == "__main__":
    app.run()
