import os.path
import sys

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy import stats
from tqdm import tqdm

from env.soft_reacher.soft_reacher import SoftReacher
from models.mbrl import LNN

PLT_LABELS = ['bend', 'shear', 'axial', 'bend_vel', 'shear_vel', 'axial_vel']


def compute_metrics(observations_gt, observations_pred):
    metrics = {}
    for i in range(6):
        mse = np.mean(np.square(observations_gt[:, i] - observations_pred[:, i]))
        metrics[f"mse#{PLT_LABELS[i]}"] = mse

        rmse = np.sqrt(mse)
        metrics[f"rmse#{PLT_LABELS[i]}"] = rmse

        mae = np.mean(np.abs(observations_gt[:, i] - observations_pred[:, i]))
        metrics[f"mae#{PLT_LABELS[i]}"] = mae

        mape = np.mean(np.abs(observations_gt[:, i] - observations_pred[:, i] / np.where(observations_gt[:, i] != 0, observations_gt[:, i], 1e-10)))
        metrics[f"mape#{PLT_LABELS[i]}"] = mape

        r_squared = stats.linregress(observations_gt[:, i], observations_pred[:, i]).rvalue ** 2
        metrics[f"r_squared#{PLT_LABELS[i]}"] = r_squared

        pearson_corr_coef = np.corrcoef(observations_gt[:, i], observations_pred[:, i])[0, 1]
        metrics[f"pearson_corr_coef#{PLT_LABELS[i]}"] = pearson_corr_coef

    return metrics


@torch.no_grad()
def rollout_plots(env, models, epoch, render=False, save=False, save_path=None, skip=False):
    """
    Rollout the model in the environment and return the trajectories.
    :param env: The environment (GT)
    :param models: a list of models (PRED)
    :param render: whether to render or not
    :param save: whether to save the plots or not to a file
    :param save_path: the path to save the plots
    :param epoch: the epoch of the training. Must be non-None if save is True
    :param skip: whether to skip some models
    """

    dt = env.dt
    max_time = 20  # seconds
    # max_time = 1  # seconds
    max_steps = int(max_time / dt)
    act_size = env.action_size
    steps_per_second = int(1 / dt)

    horizon_length = 16

    a_bounds = 1. * env.a_scale

    o_t, _, _ = env.reset()

    num_models = len(models)

    observations_gt = [o_t]
    observations_pred_l = [[o_t] for _ in range(num_models)]
    actions = []

    # get the device of the model
    device = next(models[0].parameters()).device

    o_t_l = [torch.tensor(o_t).unsqueeze(0).to(device) for _ in range(num_models)]
    pbar = tqdm(range(max_steps))

    for _ in range(max_time):
        act_ones = np.random.choice([-1, 0, 1], size=act_size)
        act = torch.tensor(act_ones * a_bounds).unsqueeze(0).to(device)
        for _ in range(steps_per_second):
            actions.append(act_ones)
            o_t_1_gt, _, _ = env.step(act_ones)
            observations_gt.append(o_t_1_gt[-1, :])
            for i, model in enumerate(models):
                if skip and i in [0, 2, 3]:
                    continue
                o_t_1_pred = model(o_t_l[i], act, train=False)
                observations_pred_l[i].append(o_t_1_pred.cpu().numpy().squeeze())
                o_t_l[i] = o_t_1_pred

            pbar.update(1)
            if render:
                env.render()

    # Compute metrics
    metrics = {f"seed_{i}": compute_metrics(np.array(observations_gt[1:]), np.array(observations_pred_l[i][1:])) for i in range(num_models) if not (skip and i in [0, 2, 3])}
    print(metrics)

    # Convert to numpy arrays
    observations_gt = np.array(observations_gt)
    observations_pred_l = [np.array(observations_pred_l[i]) for i in range(num_models)]
    actions = np.array(actions)

    # observations have shape (n_steps, 6)
    fig_r, axs_r = plt.subplots(3, 2, figsize=(15, 10))
    fig_e, axs_e = plt.subplots(3, 2, figsize=(15, 10))

    # Currently the xticks are given by the index of the observations
    # We should change this to the time in seconds
    xticks_locations = np.arange(0, max_time + dt) * steps_per_second
    xticks = np.arange(0, max_time + dt, dtype=int)

    for i in range(3):
        # Rollout data
        axs_r[i % 3, 0].plot(observations_gt[:, i], label=f"Ground truth", linewidth=5, color='red')
        axs_r[i % 3, 0].plot(actions[:, i], label="Actuation", linewidth=5, color='lightgreen')
        axs_r[i % 3, 1].plot(observations_gt[:, i+3], label=f"Ground truth", linewidth=5, color='red')

        for k, observations_pred in enumerate(observations_pred_l):
            if skip and k in [0, 2, 3]:
                continue
            axs_r[i % 3, 0].plot(observations_pred[:, i], label=f"Prediction seed {k}")
            axs_r[i % 3, 1].plot(observations_pred[:, i+3], label=f"Prediction seed {k}")
            axs_e[i % 3, 0].plot(np.abs(observations_gt[:, i] - observations_pred[:, i]), label=f"Error seed {k}")
            axs_e[i % 3, 1].plot(np.abs(observations_gt[:, i + 3] - observations_pred[:, i + 3]), label=f"Error seed {k}")

        axs_r[i % 3, 0].set_title(PLT_LABELS[i])
        axs_r[i % 3, 0].set_xticks(xticks_locations, xticks)
        axs_r[i % 3, 0].legend()

        axs_r[i % 3, 1].set_title(PLT_LABELS[i+3])
        axs_r[i % 3, 1].set_xticks(xticks_locations, xticks)
        axs_r[i % 3, 1].legend()

        # Error data
        axs_e[i % 3, 0].set_title(PLT_LABELS[i])
        # axs_e[i % 3, 0].axvline(x=horizon_length, color='r', linestyle='--', label="Horizon")
        axs_e[i % 3, 0].set_xticks(xticks_locations, xticks)
        axs_e[i % 3, 0].legend()

        axs_e[i % 3, 1].set_title(PLT_LABELS[i+3])
        # axs_e[i % 3, 1].axvline(x=horizon_length, color='r', linestyle='--', label="Horizon")
        axs_e[i % 3, 1].set_xticks(xticks_locations, xticks)
        axs_e[i % 3, 1].legend()

    fig_r.suptitle(f"Epoch {epoch} - Rollout plots")
    fig_r.tight_layout()

    fig_e.suptitle(f"Epoch {epoch} - Error plots")
    fig_e.tight_layout()

    if save:
        fig_r.savefig(os.path.join(save_path, f"rollout_{epoch}.png"))
        fig_e.savefig(os.path.join(save_path, f"error_{epoch}.png"))
    else:
        plt.show()


if __name__ == "__main__":
    base_path = os.path.join(os.path.dirname(__file__), "FINAL", "model_alone")
    env = SoftReacher(mle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = [LNN(
        env.name,
        env.n,
        env.obs_size,
        env.action_size,
        env.dt,
        env.dt_small,
        None).to(device) for _ in range(5)]

    for i, model in enumerate(models):
        file = os.path.join(base_path, f"seed_{i}", "emergency.ckpt")
        checkpoint = torch.load(file, map_location=device)
        r = model.load_state_dict(checkpoint['transition_model'], strict=False)
        print("Loading weights:", r, file=sys.stderr)

    rollout_plots(env, models, 0, render=False, skip=True)
