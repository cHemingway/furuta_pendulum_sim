import time
import mujoco
import mujoco.viewer

class FurataPendulumSim:
    def __init__(self, constants: dict = None):
        self.constants = constants
        self.model, self.data = self._create_pendulum()

    def _create_pendulum(self) -> tuple[mujoco.MjModel, mujoco.MjData]:
        spec = mujoco.MjSpec.from_file('model/furuta_pendulum.xml')
        
        if self.constants is not None:
            arm1 = spec.find_body("arm1")
            arm2 = spec.find_body("arm2")

            arm1.geoms[0].size[1] = self.constants['L_1']
            arm2.geoms[0].size[1] = self.constants['L_2']
            arm1.mass = self.constants['m_1']
            arm2.mass = self.constants['m_2']

            # We use built in interial calcs for now. TODO: Why does interia not work properly?

            # FIXME: Why are these values so much smaller than expected to get a decent sim?
            arm1.joints[0].damping = 0.0004
            arm2.joints[0].damping = 0.0001

            arm1.joints[0].frictionloss = 1E-4
            arm2.joints[0].frictionloss = 1E-8

        model = spec.compile()
        data = mujoco.MjData(model)

        return model, data
    

    def launch_interactive_sim(self, duration: int = 30):
        self.data.qpos[1] = 0.1

        with mujoco.viewer.launch_passive(self.model, self.data, show_left_ui=False, show_right_ui=False) as viewer:
            start = time.time()
            with viewer.lock():
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_INERTIA] = True
                

            while viewer.is_running() and time.time() - start < duration:
                step_start = time.time()
                mujoco.mj_step(self.model, self.data)
                viewer.sync()

                time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

if __name__ == '__main__':
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

    sim = FurataPendulumSim(constant_values)
    sim.launch_interactive_sim()


