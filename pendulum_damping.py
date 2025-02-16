import marimo

__generated_with = "0.10.14"
app = marimo.App(width="medium", app_title="Pendulum Damping")


@app.cell(hide_code=True)
def _(mo):
    mo.md("""# Calculating pendulum arm bearing friction from recorded data""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Theory

        Small angle approximation of pendulum (valid for $\theta < 0.1$ radians)

        $$ \theta(t) = \theta_0 cos \Big(\sqrt{\frac{g}{l}}t\Big) $$

        Where l is the length (to centre of gravity, not total length), solidworks gives 0.0328 metres for us, and $g = 9.81 ms^{-2}$ 


        However we have a damping factor q (proportional to velocity, also referred to as d), so instead we need to use the [more complex form](https://www.ippp.dur.ac.uk/~krauss/Lectures/NumericalMethods/Oscillator/Lecture/os4.html) as a differential equation.

        $$ \frac{d^2 \theta}{dt^2} = -q \frac{d\theta}{dt} -\frac{g}{l}\theta $$

        Assuming underdamping (which should very much be the case if we have a half decent bearing)

        $$ \theta_t = \theta_0 exp(-qt/2) \sin \Big(\sqrt{\omega^2 - \frac{q^2}{4}}t + \phi \Big) $$

        Where $\omega^2 = g/l$

        As we know g and l pretty well, we should hence be able to find damping factor $q$ by fitting our model against recorded data. Here is some I recorded earlier by holding the rotor steady, lifting the pendulum to a small angle, and letting it go:
        """
    )
    return


@app.cell
def _(np, plt):
    # Load CSV, skip all the other columns
    raw_t, _, _, _, _, _, _, raw_angle = np.loadtxt("./data/pendulum_ringdown.csv", skiprows=1, delimiter=",", unpack=True)

    # Subtract start time from all times to get time to start at zero
    raw_t = raw_t - raw_t[0]

    # Subtract average angle to remove sensor offset
    raw_angle = raw_angle - np.mean(raw_angle)

    # Plot vs time
    _fig, _ax = plt.subplots(figsize=(8,4),layout="tight")
    _ax.plot(raw_t, raw_angle)
    _ax.set_title("Angle data from pendulum swinging down (rotor held steady)")
    _ax.set_ylabel("raw_angle (Radians)")
    _ax.set_xlabel("t (s)")
    _fig
    return raw_angle, raw_t


@app.cell(hide_code=True)
def _(mo):
    g = 9.81
    #l = 0.0328
    l_slider = mo.ui.slider(0.03, 0.055, value=0.049, step=0.001, show_value=True)
    q_slider = mo.ui.slider(0.1, 1, value=0.85, step=0.01, show_value=True)
    phase_slider = mo.ui.slider(0, 2, value=0.45, step=0.05, show_value=True)
    time_slider = mo.ui.slider(0,5, value=0, step=0.01, show_value=True)
    mo.md(f"""
          ### Run Simulation

          **Start Time:**  {time_slider} seconds, make sure to adjust to start at a peak!
          
          **Set Parameters**
          
          You can manually "curve fit" by adjusting these parameters 
          
          {g=} 
          
          L = {l_slider} (m) (should not need adjusting?)
          
          Q = {q_slider} (unitless)
          
          $\\phi=$ {phase_slider} $\\pi$ (phase, adjust to line up with recorded data)
          """)
    return g, l_slider, phase_slider, q_slider, time_slider


@app.cell
def _(
    g,
    l_slider,
    np,
    phase_slider,
    q_slider,
    raw_angle,
    raw_t,
    time_slider,
):
    def simulate(t, g, l, q, theta_0, phase=0):
        ''' Simulates a pendulum given existing time vector t and futher parameters '''
        omega_2 = g/l # Omega Squared
        envelope = theta_0 * np.exp(- (q*t)/2)
        osc = np.sin(np.sqrt(omega_2 - ((q**2)/4)) * t + phase)
        return osc*envelope, envelope

    def crop_time(start_time):
        start_idx = min(np.argwhere(raw_t>start_time))[0] 
        return raw_t[start_idx:], raw_angle[start_idx:]

    t, real_angle = crop_time(time_slider.value)

    # Get start angle
    start_angle = max(real_angle)

    sim_angle, envelope = simulate(t, g, l_slider.value, q_slider.value, start_angle, phase_slider.value * np.pi)
    return (
        crop_time,
        envelope,
        real_angle,
        sim_angle,
        simulate,
        start_angle,
        t,
    )


@app.cell(hide_code=True)
def _(envelope, plt, real_angle, sim_angle, t):
    # Simualation plotting code, hidden as its boring
    _fig, _ax = plt.subplots(figsize=(8,4),layout="tight")
    _ax.plot(t, real_angle, label="Real")
    _ax.plot(t, sim_angle, label="Sim")
    _ax.plot(t, envelope, "--", label="Envelope")
    _ax.set_title("Angle data from pendulum swinging down (rotor held steady)")
    _ax.set_ylabel("Angle (Radians)")
    _ax.set_xlabel("t (s)")
    _ax.legend()
    _fig
    return


@app.cell(hide_code=True)
def _(l_slider, mo, q_slider):
    mo.md(f"""This simplified simulation matches reality OK, but there are two issues:

    - We needed a quite different value for L than Solidworks gave, l={l_slider.value*1000}mm rather than l=32.8mm. 
        - Could this be due to not simulating air resistance or other higher order terms? Or solidworks giving the wrong value? 
        - Measuring the manufactured part (a smooth rod) I get total length = 69mm (from pivot), so ignoring the small stub above the pivot we would expect $l=L/2$ = 34.5mm, so solidworks is probably correct with its more accurate model.

    - We don't capture the "stiction" behaviour at small velocities well. This is likely harder to fix, unclear how much it upsets out model to not include this.

    The estimated value of $Q={q_slider.value:.3}$ is probably still better than an outright guess though
    """)
    return


@app.cell(hide_code=True)
def _():
    # Imports
    import marimo as mo
    import numpy as np
    from matplotlib import pyplot as plt
    return mo, np, plt


if __name__ == "__main__":
    app.run()
