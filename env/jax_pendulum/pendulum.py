import os
from typing import Union

# import cv2
import jax.numpy as jnp
import jsrm
import numpy as np
import torch
from jax import vmap
from jax.tree_util import Partial
from jsrm import ode_factory
from jsrm.systems import pendulum

from env import rewards
from utils.drawing_utils import *
from utils.env_utils import step_factory
from utils.tools import wrap

from env.jax_base_env.base_env import JaxBaseEnv

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"


class JaxPendulum(JaxBaseEnv):
    name = "jax_pendulum"

    def __init__(self, num_links=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert 0 < num_links <= 2, "Only 1 or 2 links are supported"
        self.n = num_links
        # Times 3 because in the obs i have sin(q), cos(q) and q_dot for each link
        self._obs_size = 3 * self.n
        self._action_size = self.n  # TODO: check this

        # Physical parameters for the pendulum
        self.params = {
            "m": jnp.array([0.1, 0.1])[:self.n],
            "I": jnp.array([0.1/12, 0.1/12])[:self.n],
            "l": jnp.array([1, 1])[:self.n],
            "lc": jnp.array([0.5, 0.5])[:self.n],  # center of mass of each link (distance from joint)
            "g": jnp.array([0.0, 0.0]),
            # "g": jnp.array([0.0, -9.81]),
        }
        self._total_length = jnp.sum(self.params["l"])

        # Retrieve the path of the symbolic expression file
        sym_exp_filepath = os.path.join(os.path.dirname(jsrm.__file__),
                                        "symbolic_expressions",
                                        f"pendulum_nl-{num_links}.dill")

        # Jax backbone functions
        forward_kinematics_fn, self._dynamical_matrices_fn = pendulum.factory(sym_exp_filepath)
        self._batched_forward_kinematics_fn = vmap(
            forward_kinematics_fn, in_axes=(None, None, 0), out_axes=-1
        )

        # Craft the jax-based step function
        self._step_fn = step_factory()

        # TODO: check these lines
        # Physical bounds for state and actuation
        self._pos_limit = jnp.pi * jnp.ones((self.n,))
        self._vel_limit = 20. * jnp.ones((self.n,))
        self._state_limit = jnp.concatenate((self._pos_limit, self._vel_limit))
        # b = [5.*(i+1) for i in reversed(range(self.n))]
        # b = [5.0, 5.0]
        self._act_limit = 0.1 * jnp.ones((self._action_size,))
        self._goal = jnp.array([0.0, 0.15])

        # RESET STATE
        self._reset_state()

    @property
    def observation_space_bounds(self) -> np.ndarray:
        return np.concatenate(([1, 1]*self.n, np.asarray(self._vel_limit)))

    @property
    def action_space_bounds(self) -> np.ndarray:
        return np.asarray(self._act_limit)

    @property
    def pos_tolerances(self) -> np.ndarray:
        return 1.01 * np.ones((2*self.n,))

    @property
    def observation_space_size(self) -> int:
        return self._obs_size

    @property
    def action_space_size(self) -> int:
        return self._action_size

    def __str__(self):
        # TODO: dt sample?
        return f"{self.name}_n{self.n}_@{self._dt_sample}"

    def _sample_goal(self) -> None:
        pass

    def _wrap_state(self):
        wrapped_pos = wrap(self._state[:self.n])
        # clipped_vel = np.clip(self._state[self.n:], -self._vel_limit, self._vel_limit)
        # self._state = jnp.concatenate((wrapped_pos, clipped_vel))
        self._state = jnp.concatenate((wrapped_pos, self._state[self.n:]))

    def _reset_state(self, **kwargs):
        if 'obs' in kwargs:
            self._state = kwargs['obs']
            if type(self._state) is not jnp.ndarray:
                self._state = jnp.array(self._state)
        elif 'rand' in kwargs and kwargs['rand']:
            initial_pos = np.random.uniform(-1, 1, size=(self.n,)) * self._pos_limit
            initial_vel = np.random.uniform(-1, 1, size=(self.n,)) * self._vel_limit
            self._state = jnp.concatenate((initial_pos, initial_vel))
        else:
            # with 2 links it should be something like [q0, q1, q0_dot, q1_dot]
            self._state = jnp.zeros((2 * self.n,))  # initial condition
            # TODO: set initial configuration (random?)
        self._wrap_state()

    def get_obs(self):
        return np.asarray(self.state_to_obs(self._state), dtype=np.float64)

    def state_to_obs(self, state):
        """
        Convert a state to an observation
        Args:
            state: state to convert
        """
        s = []
        for i in range(self.n):
            s.append(jnp.cos(state[i]))
            s.append(jnp.sin(state[i]))
        s.extend([*state[self.n:]])

        return jnp.array(s)

    def obs_to_state(self, obs):
        """
        Convert an observation to a state
        Args:
            obs: observation to convert
        """
        s = []
        for i in range(self.n):
            j = i * self.n
            s.append(jnp.arctan2(obs[j + 1], obs[j]))
        s.extend([*obs[-self.n:]])

        return jnp.array(s)

    def step(self, a: np.ndarray, action_repeat, **kwargs):
        if self._state is None:
            raise RuntimeError("You have to call reset before calling step.")
        if not isinstance(a, torch.Tensor):
            a = torch.asarray(a)
        else:
            a = a.detach().cpu().numpy()
        # assert torch.all((a <= 1.0) & (-1.0 <= a))
        tau = jnp.array(a * self.action_space_bounds)

        if 'check_goal' in kwargs:
            check_goal = kwargs['check_goal']
        else:
            check_goal = True

        if 'time_limit' in kwargs:
            time_limit = kwargs['time_limit']
        else:
            time_limit = True

        if 'check_feasibility' in kwargs:
            check_feasibility = kwargs['check_feasibility']
        else:
            check_feasibility = True

        self._state = self._step_fn(self._state,
                                    tau,
                                    self._t,
                                    self._dt,
                                    self._instants_per_step * action_repeat,
                                    Partial(self._dynamical_matrices_fn),
                                    Partial(ode_factory),
                                    self.params)

        self._wrap_state()

        # print("Stepped: ", end="")
        # print(self._t // (self._instants_per_step * action_repeat))

        self._t += self._instants_per_step * action_repeat

        chi_curve_points = self.cartesian_from_obs()

        # end effector position
        ee_pos = chi_curve_points[-1, :]

        done_or_truncated = self._done(end_effector_pos=ee_pos,
                                       check_goal=check_goal,
                                       time_limit=time_limit,
                                       check_feasibility=check_feasibility)

        rew = self.get_reward(end_effector_pos=ee_pos,
                              action=(tau/self.action_space_bounds))

        obs = self.get_obs()

        if done_or_truncated < 0:
            # NB: being the reward negative, multiplying times 2 actually makes the reward more negative
            # rew = max(rew * 2, -1)
            rew *= 100

        return obs, rew, bool(done_or_truncated)

    def _done(self,
              end_effector_pos: jnp.ndarray,
              check_goal=True,
              time_limit=True,
              check_feasibility=True) -> int:

        assert not jnp.any(jnp.isnan(self._state))
        if time_limit and self._t >= self._t_max:
            return -1

        # Configuration bounds
        if (check_feasibility and
                (jnp.any(jnp.isnan(self._state)) or jnp.any(jnp.abs(self.get_obs()) > self.observation_space_bounds))):
            return -1

        # Goal reached
        if check_goal and jnp.isclose(end_effector_pos, self._goal).all():
            return 1
        return 0

    def get_reward(self, **kwargs):
        upright = (np.array([np.cos(self._state[0]), np.cos(self._state[1])]) + 1) / 2

        ang_vel = self._state[self.n:]
        small_velocity = rewards.tolerance(ang_vel, margin=self._vel_limit[0]).min()
        small_velocity = (1 + small_velocity) / 2

        reward = upright.mean() * small_velocity

        return reward

    def cartesian_from_obs(self, obs: np.ndarray = None, numpy: bool = False) -> Union[np.ndarray, jnp.ndarray]:
        if obs is None:
            obs = self._state
        elif type(obs) is torch.Tensor:
            obs = jnp.array(obs.cpu().numpy())
        elif type(obs) is np.ndarray:
            obs = jnp.array(obs)
        elif isinstance(obs, jnp.ndarray):
            pass
        else:
            raise TypeError(f"Unknown type {type(obs)}")
        q = obs[:self.n]

        link_indices = jnp.arange(self.params["l"].shape[0], dtype=jnp.int32)
        chi_ls = jnp.zeros((3, link_indices.shape[0] + 1))
        chi_ls = chi_ls.at[:, 1:].set(
            self._batched_forward_kinematics_fn(self.params, q, link_indices)
        )[:2, :].T

        if numpy:
            return np.asarray(chi_ls)
        else:
            return chi_ls

    def _draw(self, **kwargs):
        # plotting in Pygame
        h, w = self._screen_height, self._screen_width  # img height and width
        ppm = h / (2.5 * jnp.sum(self.params["l"]))  # pixel per meter
        robot_color = (255, 0, 0)
        pred_r_color = (0, 153, 255)
        tip_color = (255, 215, 50)
        pred_t_color = (0, 102, 255)
        goal_color = (102, 204, 0)
        base_color = (153, 153, 153)

        # poses along the robot of shape (3, N)
        chi_ls = self.cartesian_from_obs().T

        curve_origin = np.array([w // 2, h // 2], dtype=np.int32)  # in x-y pixel coordinates

        # transform robot poses to pixel coordinates
        # should be of shape (N, 2)
        curve = np.array((curve_origin + chi_ls[:2, :].T * ppm), dtype=np.int32)
        curve[:, 1] = h - curve[:, 1]  # invert the v pixel coordinate
        # curve = np.flip(curve, axis=1)

        pygame.draw.lines(self._screen, robot_color, False, curve, 10)
        pygame.draw.circle(self._screen, tip_color, curve[-1, :], 6)
        pygame.draw.circle(self._screen, base_color, curve_origin, 10)

        if 'pred' in kwargs:
            pred = kwargs['pred'].T.squeeze()
            pred = self.obs_to_state(np.asarray(pred))
            chi_ls_pred = self.cartesian_from_obs(obs=pred).T

            curve_pred = np.array((curve_origin + chi_ls_pred[:2, :].T * ppm), dtype=np.int32)
            curve_pred[:, 1] = h - curve_pred[:, 1]
            pygame.draw.lines(self._screen, pred_r_color, False, curve_pred, 10)
            pygame.draw.circle(self._screen, pred_t_color, curve_pred[-1, :], 6)

        # draw the goal
        self._draw_goal(curve_origin, ppm, goal_color)
