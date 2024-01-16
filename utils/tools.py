import random

import numpy as np
import torch
import yaml

from utils import PROJECT_PATH


def inspect_tensor(tensor: torch.Tensor, name=None):
    tmin = tensor.min()
    tmax = tensor.max()
    tmean = tensor.mean()
    if name:
        name += ' '
    return f"{name}Min: {tmin}, Max: {tmax}, Mean: {tmean}"


def validate_experiment_setup(num_episodes, max_ep_length, batch_size):
    # the calculations are performed assuming that
    # - dt has a fixed value of 1e-4 s
    # - the control frequency fcont has a fixed value of 25 Hz
    # the formula is:
    # initial_episodes 8 t_max * dt * fcont >= batch_size
    if num_episodes * max_ep_length / batch_size >= 400:
        return num_episodes
    else:
        return int(400 * batch_size / max_ep_length)


def wrap(x):
    """
    The wrap function takes an angle in radians and wraps it to the interval [-pi, pi].

    :param x: The angle to wrap
    :return: A value between -pi and pi
    """
    return ((x + np.pi) % (2 * np.pi)) - np.pi


def get_conf(argv, main=True):
    assert len(argv) <= 2, "Cannot pass more than one argument"
    try:
        config_file = argv[1]
    except IndexError as e:
        if main:
            config_file = "config_fast.yaml"
        else:
            raise e

    if not config_file.endswith(".yaml"):
        config_file += '.yaml'
    try:
        with open(f"setup/{config_file}") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: file {config_file} not found. Is it in the folder `{PROJECT_PATH}/setup`?")
        exit(-1)

    # config['config_file'] = config_file
    print(f"Using config file {config_file}")

    return config


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
