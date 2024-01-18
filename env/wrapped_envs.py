from env.base import BaseEnv
from env.jax_pendulum.pendulum import JaxPendulum


class WrappedJaxPendulum:
    pendulum_params = {
        'name': 'jax_pendulum',
        'num_links': 2,
        'dt': 2e-2,
        't_max': 20,  # in seconds
        'dt_sample': 2e-2,
    }

    def __init__(self, mle):
        self._env = JaxPendulum(**self.pendulum_params)
        self.obs_size = self._env.observation_space_size
        self.action_size = self._env.action_space_size
        self.a_scale = self._env.action_space_bounds

        self.name = self._env.name
        self.n = self._env.n
        self.dt = self._env.dt

    def reset(self):
        return self._env.reset()

    def step(self, a=None, da_ds=None):
        return self._env.step(a, 1)

    def render(self):
        return self._env.render()