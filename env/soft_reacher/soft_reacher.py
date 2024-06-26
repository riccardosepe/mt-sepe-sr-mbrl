import os

import dill as pickle
import numpy as np
import pygame
from scipy.integrate import solve_ivp

from env import rewards
from env.base import BaseEnv
from env.utils import basic_check
from utils.utils import adjust_color_brightness

pickle.settings['recurse'] = True

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"


class SoftReacher(BaseEnv):
    def __init__(self, mle):
        th0 = 0.0
        rho = 1070
        self.l = 1e-1
        self.r = 2e-2
        gy = -9.81
        E = 1e3
        mu = 5e2
        d_diag = [1e-5, 1e-2, 1e-2]
        self._eps = 1e-2
        dt = 1e-2
        self.dt_small = 2e-4

        super(SoftReacher, self).__init__(name="soft_reacher",
                                          n=3,
                                          obs_size=6,
                                          action_size=3,
                                          inertials=[th0, rho, self.l, self.r, gy, E, mu, *d_diag],
                                          dt=dt,
                                          a_scale=np.array([2.0]),
                                          mle=mle)

        with open("./env/"+self.name+"/chi.p", "rb") as inf:
            self.chi = pickle.load(inf)

        self.a_scale = np.array([0.002,   0.05,    0.1])
        self._rk_method = "RK45"

        del self.ang_vel_limit
        self.qdot_limit = np.array([200, 5, 5])
        assert self.qdot_limit.shape == (self.n,)

        self.mask = np.tile([self.l, 1, 1], 2)

        self._goal = None
        self._sample_goal()

    @property
    def eps(self):
        if np.abs(self.state[0]) < self._eps:
            eps = self._eps * np.sign(self.state[0])
        else:
            eps = 0

        return eps

    def _eps_fn(self, state):
        if np.abs(state[0]) < self._eps:
            eps = self._eps * np.sign(state[0])
        else:
            eps = 0

        return eps

    def reset(self, wide=False, sample_goal=False):
        self.reset_state(wide)

        if self.mle:
            self.phi = np.identity(self.phi_dim)

        self.t = 0
        self.w = np.array([0.0])
        if sample_goal:
            # self._sample_goal()
            self._goal = np.random.uniform(low=-self.l, high=self.l, size=2)

        return self.get_obs(), 0.0, False

    def reset_state(self, wide=False):
        if wide:
            v = 0.5
        else:
            v = 0.01
        initial_pos = np.random.uniform(low=-v, high=v, size=self.n)
        self.state = np.concatenate((initial_pos, np.zeros(self.n, )))

    def get_obs(self, bunch_of_states=None):
        if bunch_of_states is not None:
            return bunch_of_states * self.mask
        else:
            return self.state * self.mask

    def _reward_function(self, state):
        q, qdot = state[:self.n], state[self.n:]
        ee_pos = self.chi([0, self.r, self.eps, *q], self.l)[:2].T
        goal_dist = np.linalg.norm(ee_pos - self._goal)

        pos_reward = rewards.tolerance(goal_dist, margin=self.l * 0.7)
        # pos_reward_n = (1 + pos_reward) / 2

        vel_reward = rewards.tolerance(qdot, margin=self.qdot_limit).min()
        vel_reward = (1 + vel_reward) / 2

        reward = pos_reward * vel_reward
        self.reward_breakup.append([pos_reward, vel_reward])

        return reward

    def get_reward(self, bunch_of_states=None):
        if bunch_of_states is not None:
            return np.array([self._reward_function(bunch_of_states[i, :]) for i in range(bunch_of_states.shape[0])])
        else:
            return self._reward_function(self.state)

    def get_power(self, a, sdot):
        return np.array([a[0] * sdot[0]])

    def _dsdt(self, t, s_all):
        s, a, w = self.get_components(s_all)
        if np.abs(s[0]) < self._eps:
            eps = self._eps * np.sign(s[0])
        else:
            eps = 0
        sdot = self.F(self.inertials + [eps] + s.tolist()+a.tolist()).flatten()
        return np.concatenate((sdot, self.a_zeros, self.get_power(a, sdot)))

    def set_state(self, o):
        self.state = o / self.mask

    def step(self, a, da_ds=None, last=False):
        s = self.state
        a = np.clip(a, -1.0, 1.0)
        w = self.w

        t_eval = np.arange(0+self.dt_small, self.dt+self.dt_small, step=self.dt_small)

        s_all = np.concatenate((s, a*self.a_scale, w))
        y1 = solve_ivp(self._dsdt, [0, self.dt], s_all, t_eval=t_eval, method=self._rk_method)
        ns = y1.y[:, -1]  # only care about final timestep
        ns, nw = ns[:-(1+self.action_size)], ns[-1:]  # omit action

        self.w = nw
        self.state = ns
        self.t += 1

        if self.t >= self.t_max:  # infinite horizon formulation, no terminal state, similar to dm_control
            done = True
        else:
            done = False

        if last:
            return self.get_obs(), self.get_reward(), done
        else:
            states = y1.y[:-(1+self.action_size), :].T

            return self.get_obs(states), self.get_reward(states), done

    def cartesian_from_obs(self, state=None):
        s_ps = np.linspace(0, self.l, 50)
        chi_ps = []

        if state is None:
            eps = self.eps
            q = self.state[:self.n]
        else:
            eps = self._eps_fn(state)
            q = state[:self.n]
        for s_p in s_ps:
            chi_ps.append(self.chi([0, self.r, eps, *q], s_p))
        return np.array(chi_ps).squeeze()

    def _sample_goal(self):
        # for the moment hardcoded
        self._goal = np.array([self.l/2, self.l*0.8])

    def draw(self, **kwargs):
        # plotting in Pygame
        h, w = self.screen_height, self.screen_width  # img height and width
        ppm = h / (2.0 * self.l)  # pixel per meter
        base_color = (150, 150, 150)
        # robot_color = (72, 209, 204)
        robot_color = (0, 40, 75)
        tip_color = (255, 69, 0)

        # poses along the robot of shape (3, N)
        chi_ps = self.cartesian_from_obs().T

        curve_origin = np.array(
            [w // 2, int(0.1 * h)], dtype=np.int32
        )  # in x-y pixel coordinates

        # draw the base
        pygame.draw.rect(self.screen, base_color, (0, h - curve_origin[1], w, curve_origin[1]))

        if 'goal' in kwargs:
            # draw goal
            # goal color: green
            goal_color = (50, 205, 50)
            goal_pos = curve_origin + self._goal * ppm
            goal_pos[1] = h - goal_pos[1]
            pygame.draw.circle(self.screen, goal_color, goal_pos, 10)

        # transform robot poses to pixel coordinates
        # should be of shape (N, 2)
        curve = np.array((curve_origin + chi_ps[:2, :].T * ppm), dtype=np.int32)
        # invert the v pixel coordinate
        curve[:, 1] = h - curve[:, 1]
        pygame.draw.lines(self.screen, robot_color, False, curve, 10)
        pygame.draw.circle(self.screen, tip_color, curve[-1, :], 10)

        if 'other' in kwargs:
            other_robot_color = adjust_color_brightness(robot_color, -0.8)
            other_tip_color = adjust_color_brightness(tip_color, -0.8)
            state = kwargs['other']
            chi_ps = self.cartesian_from_obs(state).T

            # transform robot poses to pixel coordinates
            # should be of shape (N, 2)
            curve = np.array((curve_origin + chi_ps[:2, :].T * ppm), dtype=np.int32)
            # invert the v pixel coordinate
            curve[:, 1] = h - curve[:, 1]
            pygame.draw.lines(self.screen, other_robot_color, False, curve, 10)
            pygame.draw.circle(self.screen, other_tip_color, curve[-1, :], 10)


if __name__ == '__main__':
    while os.getcwd().split('/')[-1] != "Physics_Informed_Model_Based_RL":
        os.chdir('..')
        if os.getcwd() == '/':
            raise Exception("Could not find project directory")
    basic_check("soft_reacher", 0)
