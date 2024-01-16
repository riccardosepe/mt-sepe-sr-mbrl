import os
from functools import partial
from typing import Union

import jax
# import cv2
import jax.numpy as jnp
import jsrm
import numpy as np
import torch
from jax import vmap
from jax.tree_util import Partial
from jsrm import ode_factory
from jsrm.systems import pendulum

from environments.jax_base_env.base_env import JaxBaseEnv
from environments.jax_pendulum.pendulum import JaxPendulum
from utils.drawing_utils import *
from utils.env_utils import step_factory
from utils.tools import wrap

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"


class VecJaxPendulum(JaxPendulum):
    name = "jax_pendulum"

    def __init__(self, batch_size, *args, **kwargs):
        self.batch_size = batch_size
        self._batched_wrap = jax.vmap(wrap)
        super().__init__(*args, **kwargs)
        self._batched_obs_to_state = jax.vmap(self.obs_to_state)
        self._batched_state_to_obs = jax.vmap(self.state_to_obs)

    def _reset_state(self, **kwargs):
        if 'obs' in kwargs:
            obs = kwargs['obs']
            if type(obs) is torch.Tensor:
                obs = obs.cpu().numpy()
            if type(obs) is not jnp.ndarray:
                state = self._batched_obs_to_state(obs)
                self._state = jnp.array(state)
        elif 'rand' in kwargs and kwargs['rand']:
            initial_pos = np.random.uniform(-1, 1, size=(self.n,)) * self._pos_limit
            initial_vel = np.random.uniform(-1, 1, size=(self.n,)) * self._vel_limit
            self._state = jnp.concatenate((initial_pos, initial_vel))
        else:
            # with 2 links it should be something like [q0, q1, q0_dot, q1_dot]
            self._state = jnp.zeros((self.batch_size, 2 * self.n,))  # initial condition
            # TODO: set initial configuration (random?)
        self._wrap_state()

    def get_obs(self):
        """
        In general this function is used to provide to the outside an observation from the inner (complete) state. In
        this environment, it just converts the state to a numpy array to enable compatibility with other (non-jax)
        environments.
        Returns: An observation of the current state

        """
        return torch.tensor(np.asarray(self._batched_state_to_obs(self._state)))

    def _wrap_state(self):
        wrapped_pos = self._batched_wrap(self._state[:, :self.n])
        clipped_vel = np.clip(self._state[:, -self.n:], -self._vel_limit, self._vel_limit)
        self._state = jnp.concatenate((wrapped_pos, clipped_vel), axis=1)

    def step(self, a: np.ndarray, action_repeat, **kwargs):
        if self._state is None:
            raise RuntimeError("You have to call reset before calling step.")
        if not isinstance(a, torch.Tensor):
            a = torch.asarray(a)
        # assert torch.all((a <= 1.0) & (-1.0 <= a))
        tau = jnp.array(a * self.action_space_bounds)

        batched_step = jax.vmap(partial(self._step_fn,
                                        t0=self._t,
                                        dt=self._dt,
                                        ips=self._instants_per_step,
                                        dynamical_matrices_fn=Partial(self._dynamical_matrices_fn),
                                        ode_factory=Partial(ode_factory),
                                        params=self.params
                                        ))

        self._state = batched_step(self._state, tau)

        self._wrap_state()

        self._t += self._instants_per_step * action_repeat

        return self.get_obs()

    def _done(self, **kwargs):
        raise NotImplementedError("Should not call done signal in this environment")

    def get_reward(self, **kwargs):
        raise NotImplementedError("Should not call reward signal in this environment")

    def cartesian_from_obs(self, obs: np.ndarray = None, numpy: bool = False) -> Union[np.ndarray, jnp.ndarray]:
        raise NotImplementedError("Should not call cartesian_from_obs method in this environment")

    def _draw(self, **kwargs):
        raise NotImplementedError("Should not call draw method in this environment")

    def render(self, **kwargs):
        raise NotImplementedError("Should not call render method in this environment")

    def close(self):
        raise NotImplementedError("Should not call close in this environment")
