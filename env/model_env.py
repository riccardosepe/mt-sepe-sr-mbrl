# checkpoint = {'epoch': epoch,
#               'transition_model': transition_model.state_dict(),
#               'transition_optimizer': transition_optimizer.state_dict(),
#               'reward_model': reward_model.state_dict(),
#               'reward_optimizer': reward_optimizer.state_dict(),
#               'replay_buffer': replay_buffer
#               }
import os

import numpy as np
import pygame
import torch

from env.base import BaseEnv
from env.utils import make_env, create_background
from models.mbrl import lnn, reward_model_FC

import dill as pickle
pickle.settings['recurse'] = True

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"


class ModelEnv(BaseEnv):
    def __init__(self, path, env, batch_size=64):

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        a_zeros = None
        transition_model = lnn(
            env.name,
            env.n,
            env.obs_size,
            env.action_size,
            env.dt,
            env.dt_small,
            a_zeros).to(self.device)
        reward_model = reward_model_FC(env.obs_size).to(self.device)

        self.l = env.l
        self.r = env.r
        self.a_scale = torch.tensor(env.a_scale).to(self.device)

        self._eps = 1e-2

        checkpoint = torch.load(path, map_location=self.device)
        transition_model.load_state_dict(checkpoint['transition_model'], strict=False)
        reward_model.load_state_dict(checkpoint['reward_model'])

        self.transition_model = transition_model
        self.reward_model = reward_model

        self.name = env.name
        self.n = env.n
        self.obs_size = env.obs_size
        self.action_size = env.action_size

        self._state = None
        self.t = None
        self.t_max = 50

        self.display = False
        self.screen_width = 500
        self.screen_height = 500

        self.screen = None
        self.background = None

        self._sample_goal()
        with open("./env/"+self.name+"/chi.p", "rb") as inf:
            self.chi = pickle.load(inf)

        self._pos_bounds = np.array([np.pi, 1, 1])
        self._vel_bounds = self._pos_bounds

        self._batch_size = batch_size

    @property
    def eps(self):
        if torch.abs(self._state[0, 0]) < self._eps:
            eps = self._eps * torch.sign(self._state[0, 0])
        else:
            eps = 0

        return eps

    def reset(self):
        self.t = 0
        v = 0.1
        initial_pos = np.random.uniform(low=-2*v, high=2*v, size=(self._batch_size, self.n)) * self._pos_bounds
        initial_vel = np.random.uniform(low=-v, high=v, size=(self._batch_size, self.n)) * self._vel_bounds
        state = np.concatenate((initial_pos, initial_vel), axis=1)  # [:, None]
        # state = np.repeat(state, 64, axis=1).T

        self._state = torch.tensor(state).to(self.device)

        return self._state, 0, False

    def step(self, action, da_ds=None):
        next_state = self.transition_model(self._state, action * self.a_scale, train=False)
        reward = self.reward_model(next_state)
        self._state = next_state
        self.t += 1

        done = self.t >= self.t_max

        return self._state, reward, done

    def _sample_goal(self):
        # for the moment hardcoded
        self._goal = torch.tensor([self.l/2, self.l*0.8])

    def cartesian_from_obs(self):
        s_ps = np.linspace(0, self.l, 50)
        chi_ps = []
        eps = self.eps
        q = self._state[0, :self.n].cpu().numpy()
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

        # draw goal
        # goal color: green
        goal_color = (0, 255, 0)
        goal_pos = curve_origin + self._goal * ppm
        goal_pos[1] = h - goal_pos[1]
        pygame.draw.circle(self.screen, goal_color, goal_pos, 6)

    def render(self):
        if self.display:
            self.screen.blit(self.background, (0, 0))
            self.draw()
            pygame.display.flip()
        else:
            self.display = True
            pygame.init()
            self.screen = pygame.display.set_mode(
                (self.screen_width, self.screen_height))
            pygame.display.set_caption(self.name)
            self.background = create_background(
                self.screen_width, self.screen_height)


if __name__ == '__main__':
    env = ModelEnv(path='/Users/riccardo/PycharmProjects/TUDelft/branches/Physics_Informed_Model_Based_RL/export/model/seed_487/emergency.ckpt', env_name='soft_reacher')
    env.render()
    env.reset()
    done = False
    with torch.no_grad():
        while not done:
            action = torch.rand(env.action_size) * 2 - 1
            _, _, done = env.step(action.unsqueeze(0))
            env.render()
    pygame.quit()
