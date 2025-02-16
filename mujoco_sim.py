import time

import numpy as np
import mujoco
import mujoco.viewer

class FurataPendulumSim:
    def __init__(self, constants: dict = None, control_k = None):
        self.constants = constants
        self.control_k = control_k
        self.model, self.data = self._create_pendulum()

    def _create_pendulum(self) -> tuple[mujoco.MjModel, mujoco.MjData]:
        self.spec = mujoco.MjSpec.from_file('model/furuta_pendulum.xml')

        if self.constants is not None:
            arm1 = self.spec.find_body("arm1")
            arm2 = self.spec.find_body("arm2")

            arm1.geoms[0].size[1] = self.constants['L_1']
            arm2.geoms[0].size[1] = self.constants['L_2']
            arm1.mass = self.constants['m_1']
            arm2.mass = self.constants['m_2']

            # We use built in interial calcs for now. TODO: Why does interia not work properly?

            # FIXME: Why are these values so much smaller than expected to get a decent sim?
            arm1.joints[0].damping = 0.0002
            arm2.joints[0].damping = 0.00001

        model = self.spec.compile()
        data = mujoco.MjData(model)

        return model, data

    def get_state_vector(self, data) -> np.ndarray:
        pos = data.qpos[0:2] # [theta_1, theta_2]
        vel = data.qvel[0:2] # [theta_1_dot, theta_2_dot]
        return [pos[0], pos[1], vel[0], vel[1]] # [theta_1, theta_2, theta_1_dot, theta_2_dot]
    
    def set_state_vector(self, data, x: np.ndarray):
        data.qpos[0:2] = x[0:2]
        data.qvel[0:2] = x[2:4]

    def control(self, data: mujoco.MjData):
        # Get our current system state (using control theory terminology)
        x = self.get_state_vector(data)

        if self.control_k is not None:
            # Calculate the control input
            u = - (self.control_k @ x) # u = -Kx
            data.ctrl[0] = u

    def simulate(self, duration=5, framerate=60, initial_state=None):
        """Simulate the system for a given duration, rending at a given framerate

        Note that the physics timestep (set in XML file) is much smaller than the rendering timestep
        """
        frames = []
        sim_steps = int(duration / self.model.opt.timestep)
        x_hist = np.zeros((sim_steps, 4))
        u_hist = np.zeros((sim_steps, 1))

        mujoco.mj_resetData(self.model, self.data)

        if initial_state is not None:
            self.set_state_vector(self.data, initial_state)

        with mujoco.Renderer(self.model) as renderer:

            for i in range(sim_steps):
                self.control(self.data)
                mujoco.mj_step(self.model, self.data)

                # Save the state for later analysis
                x_hist[i, :] = self.get_state_vector(self.data)
                u_hist[i, :] = self.data.ctrl[0]

                # Render every framerate frames, not all the time!
                if len(frames) < self.data.time * framerate:
                    renderer.update_scene(self.data, camera=0) # Use 1st camera defined in XML
                    frame = renderer.render()
                    frames.append(frame)

        return frames, x_hist, u_hist


    def launch_interactive_sim(self, duration: int = 30, initial_state=None):
        mujoco.mj_resetData(self.model, self.data)
        
        if initial_state is not None:
            self.set_state_vector(self.data, initial_state)
        else:
            self.data.qvel[1] = 0.1

        with mujoco.viewer.launch_passive(self.model, self.data, show_left_ui=False, show_right_ui=False) as viewer:
            start = time.time()
            with viewer.lock():
                viewer.cam.distance = 0.75 # Zoom in with the camera
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_INERTIA] = True
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = True

            while viewer.is_running() and time.time() - start < duration:
                step_start = time.time()
                self.control(self.data)
                mujoco.mj_step(self.model, self.data)
                viewer.sync()

                time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    import mediapy as media

    constant_values = {
        "g": 9.81,
        "m_1": 0.02,
        "m_2": 0.01, 
        "l_1": 0.02, 
        "l_2": 0.03279,
        "L_1": 0.0384, 
        "L_2": 0.0703,  
        "J_1": 0.00053, 
        "J_2": 0.0000053,
        "b1": 0.85, 
        "b2": 0.85 
    }

    control_k = np.array([
        1.65, 0.14, 0.033, 0.15
    ])

    sim = FurataPendulumSim(constant_values, control_k)

    print("Testing interactive sim")
    sim.launch_interactive_sim(duration=5)

    print("Testing non interactive simulation")
    frames, x_hist, u_hist = sim.simulate(initial_state=[0, np.pi, 0, 0.1])

    try:
        media.write_video("./sim.mp4",frames, fps=60)
    except Exception as e:
        print("Failed to show video", e)

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(x_hist)
    ax[0].set_title("State vector")

    ax[1].plot(u_hist)
    ax[1].set_title("Control input")
    plt.show()
    

    print(x_hist)
