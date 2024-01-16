import datetime
from abc import ABCMeta, abstractmethod
from typing import Union, Tuple

# import cv2
import numpy as np
from gym.spaces import Box
from jax import numpy as jnp

from utils.drawing_utils import *


class JaxBaseEnv(metaclass=ABCMeta):
    name: str
    _alpha = 1 - np.log(2)

    def __init__(self, *args, dt: float = 1e-4, dt_sample: float = 1e-2, t_max: int = None, **kwargs):
        """
        Build the base environment.
        Args:
            *args: Positional arguments
            dt: The timestep of the simulation
            dt_sample: The timestep of the sampling (inverse of the sampling frequency)
            t_max: The maximum number of timesteps
            **kwargs: Keyword arguments
        """
        self._t = 0  # nr of sim steps
        self._state = None
        self._dt = float(dt)
        assert t_max is None or 0 < t_max <= 20
        self._t_max = t_max / self._dt if t_max is not None else np.inf
        self._goal = None
        self._dt_sample = float(dt_sample)

        self._total_length = 0.

        self._instants_per_step = int(self._dt_sample / self._dt)  # TODO: if this isn't integer?

        assert self._instants_per_step >= 1

        self._background = None
        self._display = False
        self._screen_width = 700
        self._screen_height = 700
        self._num_plotting_points = 50
        self._screen = None

    @property
    @abstractmethod
    def observation_space_bounds(self) -> np.ndarray:
        """
        Returns the bounds of the observation space.
        Currently, the assumption is that the observation space is symmetric, i.e. the lower bound is the opposite of
        the upper bound.
        """
        pass

    @property
    @abstractmethod
    def action_space_bounds(self) -> np.ndarray:
        """
        Returns the bounds of the action space.
        Currently, the assumption is that the action space is symmetric, i.e. the lower bound is the opposite of the
        upper bound.
        """
        pass

    @property
    @abstractmethod
    def pos_tolerances(self) -> np.ndarray:
        """
        Returns the tolerances for the position of the end effector.
        It is used in rollout simulations to detect when the predicted states are diverging a lot (according to the
        tolerances) with respect to the reference trajectory.
        """
        pass

    @property
    @abstractmethod
    def observation_space_size(self) -> int:
        """
        Returns the size of the observation space.
        """
        pass

    @property
    @abstractmethod
    def action_space_size(self) -> int:
        """
        Returns the size of the action space.
        """
        pass

    @property
    def total_length(self) -> float:
        """
        Returns the total length of the simulated robot.
        """
        return float(self._total_length)

    @property
    def observation_space(self) -> Box:
        """
        Returns the observation space.
        """
        return Box(shape=(self.observation_space_size,), low=-self.observation_space_bounds,
                   high=self.observation_space_bounds, dtype=np.float64)

    @property
    def action_space(self, *args, **kwargs) -> Box:
        """
        Returns the action space.
        """
        return Box(shape=(self.action_space_size,), low=-self.action_space_bounds,
                   high=self.action_space_bounds, dtype=np.float64)

    @property
    def unwrapped(self):
        return self

    def seed(self, seed=None):
        return [None]

    @abstractmethod
    def __str__(self):
        """
        Returns a string representation of the environment
        """
        pass

    @property
    def time(self) -> float:
        """
        Returns the time in seconds since the beginning of the episode.
        """
        return self._t * self._dt

    @property
    def _time_str(self) -> str:
        """
        Returns the time in the format mm:ss:ms
        """
        delta = datetime.timedelta(seconds=self.time)

        # Get the minutes, seconds, and milliseconds
        minutes = delta.seconds // 60
        seconds = delta.seconds % 60
        milliseconds = delta.microseconds // 1000

        return f"{minutes:02}:{seconds:02}:{milliseconds:03}"

    @property
    def sample_frequency(self) -> float:
        """
        Returns the sampling frequency, i.e. the number of samples per second.
        """
        return self._dt_sample**-1

    @property
    def instants_per_step(self) -> int:
        """
        Returns the number of instants per step, i.e. the number of `self._dt`s between two subsequent
        samples (i.e. observations).
        """
        return self._instants_per_step

    @property
    def dt(self) -> float:
        """
        Returns the timestep used for the simulation.
        """
        return self._dt

    @abstractmethod
    def _wrap_state(self) -> None:
        """
        Wrap the state which can assume redundant configurations into a fixed one.
        """
        pass

    def _reset_state(self, **kwargs):
        """
        Resets the state. For compatibility, it accepts every argument in kwargs. Based on the parameter 'state', it
        either resets the state to the initial configuration (every entry is zero or around it) or resets the state
        to a provided one.
        Args:
            **kwargs: The only useful parameter is actually 'state', used to reset the state of the environment to a
            specified one.

        """
        if type(self) is JaxBaseEnv:
            raise UserWarning("This method can only be called in subclasses.")

        if 'obs' in kwargs:
            self._state = kwargs['obs']
            if type(self._state) is not jnp.ndarray:
                self._state = jnp.array(self._state)
        elif 'rand' in kwargs:
            initial_pos = np.random.uniform(low=-1, high=1, size=self.n)
            self._state = jnp.concatenate((jnp.array(initial_pos * self._pos_limit), jnp.zeros(self.n, )))
        else:
            initial_pos = np.random.uniform(low=-0.01, high=0.01, size=self.n)
            self._state = jnp.concatenate((jnp.array(initial_pos * self._pos_limit), jnp.zeros(self.n, )))

    @abstractmethod
    def get_obs(self) -> np.ndarray:
        """
        Provide to the outside an observation from the inner (complete) state.
        Returns: An observation of the current state

        """
        pass

    @abstractmethod
    def _sample_goal(self) -> None:
        """
        This function is used to sample a goal from the task space when the environment is initialized.
        """
        pass

    def reset(self, **kwargs) -> Tuple[np.ndarray, float, bool]:
        """
        Resets the environment. This function internally resets the state and the timestep counter.
        Args:
            **kwargs: Directly passed to reset_state

        Returns: observation, reward and done signal

        """
        self._reset_state(**kwargs)

        self._t = 0

        return self.get_obs(), 0.0, False

    @abstractmethod
    def step(self, a: np.ndarray, action_repeat, **kwargs) -> Tuple[np.ndarray, float, bool]:
        """
        Step function following the gym convention. It internally calls the jax-based step function and computes the
        info needed to calculate the reward and the done signal.
        Args:
            a: the action to perform on this environment
            action_repeat: the number of times the action is repeated

        Returns: observation, reward and done signal
        """
        pass

    @abstractmethod
    def _done(self, *args) -> int:
        """
        This function computes the done conditions for the reaching task.

        * if the episode is terminated due to timestep limit or infeasible states, it returns -1
        * if it is terminated because the goal is reached, it returns 1, otherwise it returns 0

        The returned value is used as follows:

        * if it is -1, the reward will be doubled and the episode is terminated.
        * if the returned value is 0, the reward is obtained from the reward function and the episode continues.
        * if the returned value is 1, the reward is obtained from the reward function and the episode is terminated
        Args:
            end_effector_pos: The position of the end effector in cartesian coordinates

        Returns: An integer number that maps the different conditions of done, not done or truncated

        """
        pass

    @abstractmethod
    def get_reward(self, **kwargs) -> float:
        """
        This function computes and returns the reward for the reaching task.
        """
        pass

    @abstractmethod
    def cartesian_from_obs(self, obs: np.ndarray = None, numpy: bool = False) -> Union[np.ndarray, jnp.ndarray]:
        """
        This function takes the current state or the optional parameter obs and computes the cartesian coordinates for
        the robot in this state along self.num_points points. If the parameter numpy is True, the returned array is a
        numpy array, otherwise it will be a jax array.
        """
        pass

    @abstractmethod
    def _draw(self, *args, **kwargs):
        """
        This function draws the environment in the current state. It is called by the render function.
        """
        pass

    def _init_render(self):
        """
        This function initializes the attributes used for the rendering of the environment.
        """
        self._display = True
        pygame.init()
        try:
            self._screen = pygame.display.set_mode(
                (self._screen_width, self._screen_height), display=1)
        except pygame.error:
            self._screen = pygame.display.set_mode(
                (self._screen_width, self._screen_height), display=0)
        pygame.display.set_caption(self.name)
        self._background = create_background(
            self._screen_width, self._screen_height)

    def _draw_caption(self, caption: dict):
        """
        This function draws a box at the top of the screen, containing the caption passed as argument.
        """
        font = pygame.font.Font(None, 30)
        # Draw the box
        box_width, box_height = 680, 50
        box_x, box_y = self._screen_width - box_width - 10, 10
        pygame.draw.rect(self._screen, (255, 255, 255), (box_x, box_y, box_width, box_height))

        # Draw the text in the box
        string = (caption['name'] + ': ' +
                  np.array2string(np.array(caption['value']), precision=4, suppress_small=True))
        text = font.render(string, True, (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.topleft = (box_x + 10, box_y + 10)  # Position the text inside the box
        self._screen.blit(text, text_rect)

    def _add_interactive_button(self):
        """
        This function adds a button to reset the environment in the bottom right corner of the screen.
        """
        button_width, button_height = 100, 50
        button_x, button_y = self._screen_width - button_width - 10, self._screen_height - button_height - 10
        # draw button yellow
        pygame.draw.rect(self._screen, (255, 255, 0), (button_x, button_y, button_width, button_height))
        font = pygame.font.Font(None, 30)
        # add text of red color
        text = font.render('Reset', True, (255, 0, 0))
        text_rect = text.get_rect()
        text_rect.topleft = (button_x + 10, button_y + 10)  # Position the text inside the box
        self._screen.blit(text, text_rect)
        # check if the button is pressed
        mouse_pos = pygame.mouse.get_pos()
        mouse_pressed = pygame.mouse.get_pressed()
        if button_x <= mouse_pos[0] <= button_x + button_width and button_y <= mouse_pos[1] <= button_y + button_height:
            if mouse_pressed[0]:
                self._t = self._t_max

        for ev in pygame.event.get():
            if ev.type == pygame.MOUSEBUTTONDOWN:
                # if left click and position is inside the button, reset the environment
                if (ev.button == 1
                        and button_x <= ev.pos[0] <= button_x + button_width
                        and button_y <= ev.pos[1] <= button_y + button_height):
                    print("CLICKED")
                    self._t = self._t_max

    def _draw_clock(self):
        """
        This function draws a clock in the bottom left corner of the screen, showing the elapsed time since the
        beginning of the episode.
        """
        # Define the circle's parameters
        circle_radius = 50
        circle_color = (255, 255, 255)
        circle_center = (circle_radius, self._screen_height - circle_radius)

        # Create a font for the text
        font = pygame.font.Font(None, 28)
        text = str(self._time_str)

        # Get the text's dimensions
        text_width, text_height = font.size(text)

        # Calculate the position to center the text
        text_x = circle_center[0] - text_width / 2
        text_y = circle_center[1] - text_height / 2

        pygame.draw.circle(self._screen, circle_color, circle_center, circle_radius)
        pygame.draw.circle(self._screen, (0, 0, 0), circle_center, circle_radius, 2)
        text_surface = font.render(text, True, (0, 0, 0))
        self._screen.blit(text_surface, (text_x, text_y))

    def _draw_goal(self, curve_origin, ppm, goal_color):
        """
        This function draws the goal in the environment.
        Args:
            curve_origin: The origin of the curve in pixel coordinates
            ppm: The pixels per meter ratio
            goal_color: The color of the goal
        """
        goal = curve_origin + np.asarray(self._goal*ppm)
        goal[1] = self._screen_height - goal[1]
        pygame.draw.circle(self._screen, goal_color, goal, 6)

    def render(self, **kwargs) -> None:
        """
        This function renders the robot in the current state, by internally calling the method done
        """
        if len(self._state.shape) > 1:
            ns = self._state.shape[0]
        else:
            ns = 1
        for i in range(ns):
            video = False
            if 'video_writer' in kwargs:
                video = True
                video_writer = kwargs['video_writer']
            if self._display:
                self._screen.blit(self._background, (0, 0))
                self._draw(**kwargs)
                if video:
                    image = pygame.surfarray.array3d(self._screen)
                    # image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    # video_writer.write(image_bgr)
            else:
                self._init_render()
            if 'caption' in kwargs:
                self._draw_caption(kwargs['caption'])

            if 'interactive' in kwargs and kwargs['interactive']:
                self._add_interactive_button()

            self._draw_clock()

            pygame.display.flip()

    def close(self) -> None:
        """
        This function resets the variables used for rendering, so that the env can be rendered multiple times.
        """
        pygame.quit()
        self._display = False
        self._screen = None
        self._background = None
