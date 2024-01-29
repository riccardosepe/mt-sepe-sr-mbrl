from env.base import BaseEnv
from env.utils import rect_points, wrap, basic_check
from env import rewards
import numpy as np
import pygame
import os

import dill as pickle
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
        self.eps = 1e-2

        super(SoftReacher, self).__init__(name="soft_reacher",
                                          n=3,
                                          obs_size=6,
                                          action_size=3,
                                          inertials=[th0, rho, self.l, self.r, gy, E, mu, *d_diag],
                                          dt=0.02,
                                          a_scale=np.array([2.0]),
                                          mle=mle)

        with open("./env/"+self.name+"/chi.p", "rb") as inf:
            self.chi = pickle.load(inf)

        self.a_scale = np.array([0.002,   0.05,    0.1])
        self._rk_method = "RK45"

    def reset_state(self):
        initial_pos = np.random.uniform(low=-0.01, high=0.01, size=self.n)
        self.state = np.concatenate((initial_pos, np.zeros(self.n, )))

    def get_obs(self):
        return self.state

    def get_reward(self):
        # upright = (np.array([np.cos(self.state[0])]) + 1) / 2
        #
        # qdot = self.state[self.n:]
        # ang_vel = qdot
        # small_velocity = rewards.tolerance(
        #     ang_vel, margin=self.ang_vel_limit).min()
        # small_velocity = (1 + small_velocity) / 2
        #
        # reward = upright.mean() * small_velocity
        # self.reward_breakup.append([upright.mean(), small_velocity])

        return -1

    def get_power(self, a, sdot):
        return np.array([a[0] * sdot[0]])

    def _dsdt(self, t, s_all):
        s, a, w = self.get_components(s_all)
        if np.abs(s[0]) < self.eps:
            eps = self.eps*np.sign(s[0])
        else:
            eps = 0
        sdot = self.F(self.inertials + [eps] + s.tolist()+a.tolist()).flatten()
        return np.concatenate((sdot, self.a_zeros, self.get_power(a, sdot)))

    def cartesian_from_obs(self):
        s_ps = np.linspace(0, self.l, 50)
        chi_ps = []
        if np.abs(self.state[0]) < self.eps:
            eps = self.eps*np.sign(self.state[0])
        else:
            eps = 0
        q = self.state[:self.n]
        for s_p in s_ps:
            chi_ps.append(self.chi([0, self.r, eps, *q], s_p))
        return np.array(chi_ps).squeeze()

    def draw(self):
        # plotting in Pygame
        h, w = self.screen_height, self.screen_width  # img height and width
        ppm = h / (2.0 * self.l)  # pixel per meter
        base_color = (150, 150, 150)
        robot_color = (72, 209, 204)
        tip_color = (255, 69, 0)

        # poses along the robot of shape (3, N)
        chi_ps = self.cartesian_from_obs().T

        curve_origin = np.array(
            [w // 2, int(0.1 * h)], dtype=np.int32
        )  # in x-y pixel coordinates

        # draw the base
        pygame.draw.rect(self.screen, base_color, (0, h - curve_origin[1], w, curve_origin[1]))

        # transform robot poses to pixel coordinates
        # should be of shape (N, 2)
        curve = np.array((curve_origin + chi_ps[:2, :].T * ppm), dtype=np.int32)
        # invert the v pixel coordinate
        curve[:, 1] = h - curve[:, 1]
        pygame.draw.lines(self.screen, robot_color, False, curve, 10)
        pygame.draw.circle(self.screen, tip_color, curve[-1, :], 6)


if __name__ == '__main__':
    while os.getcwd().split('/')[-1] != "Physics_Informed_Model_Based_RL":
        os.chdir('..')
        if os.getcwd() == '/':
            raise Exception("Could not find project directory")
    basic_check("soft_reacher", 0)
