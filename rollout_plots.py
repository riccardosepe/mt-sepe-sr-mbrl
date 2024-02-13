import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm


PLT_LABELS = ['bend', 'shear', 'axial', 'bend_vel', 'shear_vel', 'axial_vel']


@torch.no_grad()
def rollout_plots(env, model, render=False, save=False, save_path=None):
    """
    Rollout the model in the environment and return the trajectories.
    :param env: The environment (GT)
    :param model: The model (PRED)
    :param render: whether to render or not
    :param save: whether to save the plots or not to a file
    :param save_path: the path to save the plots
    """

    dt = env.dt
    max_time = 5  # seconds
    max_steps = int(max_time / dt)
    act_size = env.action_size
    steps_per_second = int(1 / dt)

    horizon_length = 16

    a_bounds = 1. * env.a_scale

    o_t, _, _ = env.reset()

    observations_gt = [o_t]
    observations_pred = [o_t]
    actions = []

    o_t = torch.tensor(o_t).unsqueeze(0)
    pbar = tqdm(range(max_steps))

    for _ in range(max_time):
        act_ones = np.random.choice([-1, 0, 1], size=act_size)
        act = torch.tensor(act_ones * a_bounds).unsqueeze(0)
        for _ in range(steps_per_second):
            actions.append(act_ones)
            o_t_1_gt, _, _ = env.step(act_ones)
            o_t_1_pred = model(o_t, act, train=False)

            observations_gt.append(o_t_1_gt[-1, :])
            observations_pred.append(o_t_1_pred.numpy().squeeze())

            o_t = o_t_1_pred
            pbar.update(1)
            if render:
                env.render()

    # Convert to numpy arrays
    observations_gt = np.array(observations_gt)
    observations_pred = np.array(observations_pred)
    actions = np.array(actions)

    # observations have shape (n_steps, 6)
    fig, axs = plt.subplots(3, 2, figsize=(15, 10))

    # Currently the xticks are given by the index of the observations
    # We should change this to the time in seconds
    xticks_locations = np.arange(0, max_time + dt) * steps_per_second
    xticks = np.arange(0, max_time + dt)

    for i in range(3):
        axs[i % 3, 0].plot(observations_gt[:, i], label=f"Ground truth")
        axs[i % 3, 0].plot(observations_pred[:, i], label=f"Prediction")
        axs[i % 3, 0].plot(actions[:, i], label="Actuation")
        axs[i % 3, 0].axvline(x=horizon_length, color='r', linestyle='--', label="Horizon")
        axs[i % 3, 0].set_title(PLT_LABELS[i])
        axs[i % 3, 0].set_xticks(xticks_locations, xticks)
        axs[i % 3, 0].legend()

        axs[i % 3, 1].plot(observations_gt[:, i+3], label=f"Ground truth")
        axs[i % 3, 1].plot(observations_pred[:, i+3], label=f"Prediction")
        axs[i % 3, 1].axvline(x=horizon_length, color='r', linestyle='--', label="Horizon")
        axs[i % 3, 1].set_title(PLT_LABELS[i+3])
        axs[i % 3, 1].set_xticks(xticks_locations, xticks)
        axs[i % 3, 1].legend()

    plt.tight_layout()

    if save:
        # save `fig` to `save_path`
        fig.savefig(save_path)
    else:
        plt.show()
