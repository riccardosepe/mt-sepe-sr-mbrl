import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm


PLT_LABELS = ['bend', 'shear', 'axial', 'bend_vel', 'shear_vel', 'axial_vel']


@torch.no_grad()
def rollout_plots(env, model):
    """
    Rollout the model in the environment and return the trajectories.
    :param env: The environment (GT)
    :param model: The model (PRED)
    """
    dt = env.dt
    max_time = 5  # seconds
    max_steps = int(max_time / dt)
    act_size = env.action_size

    a_bounds = 0.5 * env.a_scale

    o_t, _, _ = env.reset()

    observations_gt = [o_t]
    observations_pred = [o_t]
    actions = []

    o_t = torch.tensor(o_t).unsqueeze(0)

    for _ in tqdm(range(max_steps)):
        indices = np.random.choice([-1, 0, 1], size=act_size)
        act = indices * a_bounds
        actions.append(act)
        o_t_1_gt, _, _ = env.step(act)
        o_t_1_pred = model(o_t, torch.tensor(act).unsqueeze(0))

        observations_gt.append(o_t_1_gt)
        observations_pred.append(o_t_1_pred.numpy().squeeze())

        o_t = o_t_1_pred

    # Convert to numpy arrays
    observations_gt = np.array(observations_gt)
    observations_pred = np.array(observations_pred)
    actions = np.array(actions)

    # observations have shape (n_steps, 6)
    fig, axs = plt.subplots(3, 2, figsize=(15, 10))

    for i in range(3):
        axs[i % 3, 0].plot(observations_gt[:, i], label=f"GT_{PLT_LABELS[i]}")
        axs[i % 3, 0].plot(observations_pred[:, i], label=f"PRED_{PLT_LABELS[i]}")
        axs[i % 3, 0].set_title(PLT_LABELS[i])

        axs[i % 3, 1].plot(observations_gt[:, i+3], label=f"GT_{PLT_LABELS[i+3]}")
        axs[i % 3, 1].plot(observations_pred[:, i+3], label=f"PRED_{PLT_LABELS[i+3]}")
        axs[i % 3, 1].set_title(PLT_LABELS[i+3])

    plt.tight_layout()
    plt.legend()
    plt.show()
