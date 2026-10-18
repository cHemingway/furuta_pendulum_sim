
### Furuta Pendulum Sim
A project to simulate a furuta pendulum and design a controller for it, to teach myself modern control theory.

### Files
- [furata_pendulum_dynamics.py](furata_pendulum_dynamics.py) - General grab bag of linear model and controller. In [marimo](https://marimo.io/) format
- [pendulum_damping.py](pendulum_damping.py) - Attempt to measure damping of pendulum axis via measuring ringdown.
- [mujoco_sim.py] - Simulate our actual controller using MuCoJo.  
- [balance_demo.py](balance_demo.py) - Corrected model + motor-only discrete LQR (in volts) with a Kalman filter, balanced in MuJoCo with a push on the tip at t=1 s. Includes 100 Hz control, encoder quantisation/noise and motor cogging. Renders `balance_demo.mp4`. Run with `MUJOCO_GL=egl` when headless.


### Install / dependencies
- Python dependencies in pyproject.toml
- To render video from mujoco_sim, you need ffmpeg
    - On windows 10+ `winget install ffmpeg`
    - On linux/osx, you know what to do

### Credits
- Pendulum dynamics from [On the Dynamics of the Furuta Pendulum](https://onlinelibrary.wiley.com/doi/10.1155/2011/528341)
- Furata Pendulum MuJoCo model from [macstepien/furuta_pendulum](https://github.com/macstepien/furuta_pendulum/blob/master/furuta_pendulum_rl/model/furuta_pendulum.xml), GPL licensed

### References
- [Steve Brunton's Control Bootcamp](https://www.youtube.com/watch?v=Pi7l8mMjYVE&list=PLMrJAkhIeNNR20Mz-VpzgfQs5zrYi085m) for modern control theory
- [Controls Engineering in the FIRST Robotics Competition, Tyler Veness](https://file.tavsys.net/control/controls-engineering-in-frc.pdf) for Bryson's rule to choose **Q** and **R** matrices for LQR.