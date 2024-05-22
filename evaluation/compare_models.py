import json
import os.path
import sys

import dill as pickle
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.legend_handler import Line2D
from matplotlib.lines import Line2D
from scipy import stats
from tqdm import tqdm

from env.soft_reacher.soft_reacher import SoftReacher
from models.mbrl import LNN, MLP
from utils.utils import adjust_color_brightness

pickle.settings['recurse'] = True

# PLT_LABELS = ['bend', 'shear', 'axial', 'bend_vel', 'shear_vel', 'axial_vel']
STATE_LABELS = ['$\\theta_{bend}$',
                '$\\sigma_{shear}$',
                '$\\sigma_{axial}$',
                '$\\dot{\\theta}_{bend}$',
                '$\\dot{\\sigma}_{shear}$',
                '$\\dot{\\sigma}_{axial}$']

ACT_LABELS = ['$u_{bend}$',
              '$u_{shear}$',
              '$u_{axial}$']

with open(f"{os.path.dirname(__file__)}/utils/rcparams2.json", "r") as f:
    plt.rcParams.update(json.load(f))


def compute_metrics(observations_gt, observations_pred):
    metrics = {}
    for i in range(6):
        mse = np.mean(np.square(observations_gt[:, i] - observations_pred[:, i]))
        metrics[f"mse#{PLT_LABELS[i]}"] = mse

        rmse = np.sqrt(mse)
        metrics[f"rmse#{PLT_LABELS[i]}"] = rmse

        mae = np.mean(np.abs(observations_gt[:, i] - observations_pred[:, i]))
        metrics[f"mae#{PLT_LABELS[i]}"] = mae

        mape = np.mean(np.abs(observations_gt[:, i] - observations_pred[:, i] / np.where(observations_gt[:, i] != 0,
                                                                                         observations_gt[:, i], 1e-10)))
        metrics[f"mape#{PLT_LABELS[i]}"] = mape

        r_squared = stats.linregress(observations_gt[:, i], observations_pred[:, i]).rvalue ** 2
        metrics[f"r_squared#{PLT_LABELS[i]}"] = r_squared

        pearson_corr_coef = np.corrcoef(observations_gt[:, i], observations_pred[:, i])[0, 1]
        metrics[f"pearson_corr_coef#{PLT_LABELS[i]}"] = pearson_corr_coef

    return metrics


def plot_state(observations_gt,
               observations_pred_dict,
               actions,
               epoch=0,
               save=False,
               skip=False,
               max_time=20,
               dt=1e-2,
               steps_per_second=100):
    # observations have shape (n_steps, 6)
    fig_r, axs_r = plt.subplots(3, 2, figsize=(15, 10))
    fig_e, axs_e = plt.subplots(3, 2, figsize=(15, 10))

    # Currently the xticks are given by the index of the observations
    # We should change this to the time in seconds
    xticks_locations = np.arange(0, max_time + dt) * steps_per_second
    xticks = np.arange(0, max_time + dt, dtype=int)

    colors = [
        '#1f77b4',
        '#ff7f0e'
    ]
    line_styles = ['--', '-']
    error_line_styles = ['-', '-']
    markers = [None, None]
    factors = [0, -0.3]
    line_sizes = [3, 3]
    alphas = [1, 0.65]

    # rescale data between -1 and 1
    # stack gt and pred
    all_data = np.concatenate([observations_gt, *observations_pred_dict.values()], axis=0)
    # get abs max
    # abs_max = np.max(np.abs(all_data), axis=0)
    abs_max = np.array([np.pi, 0.2, 0.2, 20, 5, 5])

    # labels = ['lnn_seed_4', 'mlp_seed_0']

    errors = {model_name: np.abs(observations_gt - observations_pred_dict[model_name]) / abs_max for model_name in
              observations_pred_dict.keys()}

    lines = {
        'act': None,
        'gt': None,
        'lnn': None,
        'mlp': None,
        'e_lnn': None,
        'e_mlp': None
    }

    for i in range(3):
        # Rollout data
        line_r_gt, = axs_r[i % 3, 0].plot(observations_gt[:, i] / abs_max[i],
                                          label=STATE_LABELS[i],
                                          linewidth=3,
                                          color=adjust_color_brightness(colors[0], 0.3),
                                          marker='^',
                                          markevery=50,
                                          markersize=10
                                          )
        line_r_act, = axs_r[i % 3, 0].plot(actions[:, i],
                                           label=ACT_LABELS[i],
                                           linewidth=3,
                                           color=colors[1]
                                           )
        axs_r[i % 3, 1].plot(observations_gt[:, i + 3] / abs_max[i + 3],
                             label=STATE_LABELS[i + 3],
                             linewidth=3,
                             color=adjust_color_brightness(colors[0], 0.3),
                             marker='^',
                             markevery=50,
                             markersize=10
                             )
        axs_r[i % 3, 1].plot(actions[:, i],
                             label=ACT_LABELS[i],
                             linewidth=3,
                             color=colors[1]
                             )

        if lines['act'] is None:
            lines['act'] = line_r_act

        if lines['gt'] is None:
            lines['gt'] = line_r_gt

        for k, model_name in enumerate(observations_pred_dict.keys()):
            line_r_p, = axs_r[i % 3, 0].plot(observations_pred_dict[model_name][:, i] / abs_max[i],
                                             label=f"{STATE_LABELS[i]}$^{{{model_name.upper()}}}$",
                                             color=adjust_color_brightness(colors[0], factors[k]),
                                             linestyle=line_styles[k],
                                             linewidth=line_sizes[k],
                                             alpha=alphas[k]
                                             )
            axs_r[i % 3, 1].plot(observations_pred_dict[model_name][:, i + 3] / abs_max[i + 3],
                                 label=f"{STATE_LABELS[i + 3]}$^{{{model_name.upper()}}}$",
                                 color=adjust_color_brightness(colors[0], factors[k]),
                                 linestyle=line_styles[k],
                                 linewidth=line_sizes[k],
                                 alpha=alphas[k]
                                 )

            line_e_p, = axs_e[i % 3, 0].plot(errors[model_name][:, i],
                                             label=f"$\\Delta${STATE_LABELS[i]}$^{{{model_name.upper()}}}$",
                                             color=adjust_color_brightness(colors[0], factors[k]),
                                             linestyle=line_styles[k],
                                             linewidth=line_sizes[k],
                                             alpha=alphas[k]
                                             )
            axs_e[i % 3, 1].plot(errors[model_name][:, i + 3],
                                 label=f"$\\Delta${STATE_LABELS[i + 3]}$^{{{model_name.upper()}}}$",
                                 color=adjust_color_brightness(colors[0], factors[k]),
                                 linestyle=line_styles[k],
                                 linewidth=line_sizes[k],
                                 alpha=alphas[k]
                                 )

            if lines[model_name] is None:
                lines[model_name] = line_r_p

            if lines[f"e_{model_name}"] is None:
                lines[f"e_{model_name}"] = line_e_p

        axs_r[i % 3, 0].set_title(STATE_LABELS[i])
        axs_r[i % 3, 0].set_xticks(xticks_locations, xticks)
        # axs_r[i % 3, 0].legend()

        axs_r[i % 3, 1].set_title(STATE_LABELS[i + 3])
        axs_r[i % 3, 1].set_xticks(xticks_locations, xticks)
        # axs_r[i % 3, 1].legend()

        # Error data
        axs_e[i % 3, 0].set_title(STATE_LABELS[i])
        # axs_e[i % 3, 0].axvline(x=horizon_length, color='r', linestyle='--', label="Horizon")
        axs_e[i % 3, 0].set_xticks(xticks_locations, xticks)
        # axs_e[i % 3, 0].legend()

        axs_e[i % 3, 1].set_title(STATE_LABELS[i + 3])
        # axs_e[i % 3, 1].axvline(x=horizon_length, color='r', linestyle='--', label="Horizon")
        axs_e[i % 3, 1].set_xticks(xticks_locations, xticks)
        # axs_e[i % 3, 1].legend()

    # fig_r.legend()
    # fig_e.legend()
    r_legend_handles = [
        Line2D(
            [],
            [],
            color=lines['gt'].get_color(),
            linestyle=lines['gt'].get_linestyle(),
            linewidth=lines['gt'].get_linewidth(),
            marker=lines['gt'].get_marker(),
            markersize=lines['gt'].get_markersize(),
            alpha=lines['gt'].get_alpha(),
            label="reference"),
        Line2D(
            [],
            [],
            color=lines['act'].get_color(),
            linestyle=lines['act'].get_linestyle(),
            linewidth=lines['act'].get_linewidth(),
            marker=lines['act'].get_marker(),
            markersize=lines['act'].get_markersize(),
            alpha=lines['act'].get_alpha(),
            label="control"),
        Line2D(
            [],
            [],
            color=lines['lnn'].get_color(),
            linestyle=lines['lnn'].get_linestyle(),
            linewidth=lines['lnn'].get_linewidth(),
            marker=lines['lnn'].get_marker(),
            markersize=lines['lnn'].get_markersize(),
            alpha=lines['lnn'].get_alpha(),
            label="LNN"),
        Line2D(
            [],
            [],
            color=lines['mlp'].get_color(),
            linestyle=lines['mlp'].get_linestyle(),
            linewidth=lines['mlp'].get_linewidth(),
            marker=lines['mlp'].get_marker(),
            markersize=lines['mlp'].get_markersize(),
            alpha=lines['mlp'].get_alpha(),
            label="MLP")
    ]
    e_legend_handles = [
        Line2D(
            [],
            [],
            color=lines[f"e_{model}"].get_color(),
            linestyle=lines[f"e_{model}"].get_linestyle(),
            linewidth=lines[f'e_{model}'].get_linewidth(),
            marker=lines[f'e_{model}'].get_marker(),
            markersize=lines[f'e_{model}'].get_markersize(),
            alpha=lines[f'e_{model}'].get_alpha(),
            label=model.upper())
        for model in observations_pred_dict.keys()
    ]

    for ax in axs_r.flatten():
        ax.set_xlabel("$[s]$")
        ax.set_ylabel("$[adim.]$")

    fig_r.tight_layout()
    fig_e.tight_layout()
    fig_r.subplots_adjust(top=0.88)
    fig_e.subplots_adjust(top=0.9)
    fig_r.legend(handles=r_legend_handles, loc='upper center', ncol=4)
    fig_e.legend(handles=e_legend_handles, loc='upper center', ncol=2)

    if save:
        save_path = f"{os.path.dirname(__file__)}/plots"
        fig_r.savefig(os.path.join(save_path, f"rollout.png"), bbox_inches='tight')
        fig_e.savefig(os.path.join(save_path, f"error.png"), bbox_inches='tight')
    else:
        plt.show()


def plot_ees1(observations_gt,
             observations_pred_l,
             epoch=0,
             save=False,
             skip=False,
             max_time=20,
             dt=1e-2,
             steps_per_second=100,
             th0=0,
             r=2e-2,
             l=1e-1):
    with open("../env/soft_reacher/chi.p", "rb") as inf:
        chi = pickle.load(inf)

    xticks_locations = np.arange(0, max_time + dt) * steps_per_second
    xticks = np.arange(0, max_time + dt, dtype=int)

    colors = [
        '#1f77b4',
        '#ff7f0e'
    ]
    line_styles = ['--', '-']
    error_line_styles = ['-', '-']
    markers = [None, None]
    factors = [0, -0.3]
    line_sizes = [3, 3]
    alphas = [1, 0.65]

    def _eps(x):
        if np.abs(x) < 1e-2:
            return 1e-2 * np.sign(x)
        else:
            return 0

    num_samples = len(observations_gt)
    indices = np.arange(0, num_samples, step=50, dtype=int)

    ee_gt = []
    ee_pred_l = {k: [] for k in observations_pred_l.keys()}

    for i in range(num_samples):
        q = observations_gt[i, :3]
        eps = _eps(q[0])
        ee_gt.append(chi([th0, r, eps, *q], l))

        for k in observations_pred_l.keys():
            q = observations_pred_l[k][i, :3]
            eps = _eps(q[0])
            ee_pred_l[k].append(chi([th0, r, eps, *q], l))

    ee_gt = (np.array(ee_gt)[:, :2]).squeeze()
    ee_pred_l = {k: (np.array(ee_pred_l[k])[:, :2]).squeeze() for k in observations_pred_l.keys()}

    fig_p, axs_p = plt.subplots(figsize=(15, 10))
    fig_e, axs_e = plt.subplots(figsize=(15, 5))
    # labels = ['lnn_seed_4', 'mlp_seed_0']

    axs_p.plot(ee_gt[:, 0],
               label="$x_{ee}$",
               linewidth=3,
               color=adjust_color_brightness(colors[0], 0.3),
               marker='^',
               markevery=50,
               markersize=10
               )
    axs_p.plot(ee_gt[:, 1],
               label="$y_{ee}$",
               linewidth=3,
               color=adjust_color_brightness(colors[1], 0.3),
               marker='^',
               markevery=50,
               markersize=10
               )

    for i, k in enumerate(observations_pred_l.keys()):
        axs_p.plot(ee_pred_l[k][:, 0],
                   label="$\\tilde{x}_{ee}^{" + k.upper() + "}$",
                   color=adjust_color_brightness(colors[0], factors[i]),
                   linewidth=line_sizes[i],
                   linestyle=line_styles[i],
                   marker=markers[i],
                   markevery=50,
                   markersize=10,
                   alpha=alphas[i]
                   )
        axs_p.plot(ee_pred_l[k][:, 1],
                   label="$\\tilde{y}_{ee}^{" + k.upper() + "}$",
                   color=adjust_color_brightness(colors[1], factors[i]),
                   linewidth=line_sizes[i],
                   linestyle=line_styles[i],
                   marker=markers[i],
                   markevery=50,
                   markersize=10,
                   alpha=alphas[i]
                   )

    # axs_p.set_title("End effector position")
    fig_p.legend(loc='upper center', ncol=6)
    axs_p.set_xticks(xticks_locations, xticks)
    axs_p.grid()

    errors = {
        k: np.abs(ee_gt - ee_pred_l[k])
        for k in observations_pred_l.keys()
    }

    for i, k in enumerate(observations_pred_l.keys()):
        axs_e.plot(errors[k][:, 0],
                   label="$\\Delta x_{ee}^{" + k.upper() + "}$",
                   color=adjust_color_brightness(colors[0], factors[i]),
                   linewidth=line_sizes[i],
                   linestyle=error_line_styles[i],
                   marker=markers[i],
                   markevery=50,
                   markersize=10
                   )
        axs_e.plot(errors[k][:, 1],
                   label="$\\Delta y_{ee}^{" + k.upper() + "}$",
                   color=adjust_color_brightness(colors[1], factors[i]),
                   linewidth=line_sizes[i],
                   linestyle=error_line_styles[i],
                   marker=markers[i],
                   markevery=50,
                   markersize=10
                   )

    # axs_e.set_title("End effector position error")
    fig_e.legend(loc='upper center', ncol=4)
    axs_e.set_xticks(xticks_locations, xticks)
    axs_e.grid()

    axs_p.set_xlabel("$[s]$")
    axs_p.set_ylabel("$[m]$")
    axs_e.set_xlabel("$[s]$")
    axs_e.set_ylabel("$[m]$")

    fig_p.tight_layout()
    fig_e.tight_layout()
    fig_p.subplots_adjust(top=0.9)
    fig_e.subplots_adjust(top=0.8)

    return {'p': axs_p.get_ylim(), 'e': axs_e.get_ylim()}

    if save:
        save_path = f"{os.path.dirname(__file__)}/plots"
        fig_p.savefig(os.path.join(save_path, f"ee.png"), bbox_inches='tight')
        fig_e.savefig(os.path.join(save_path, f"ee_error.png"), bbox_inches='tight')
    else:
        plt.tight_layout()
        plt.show()


def plot_ees(observations_gt,
              observations_pred_l,
              epoch=0,
              save=False,
              skip=False,
              max_time=20,
              dt=1e-2,
              steps_per_second=100,
              th0=0,
              r=2e-2,
              l=1e-1):

    lims = plot_ees1(observations_gt,
                     observations_pred_l,
                     epoch,
                     save,
                     skip,
                     max_time,
                     dt,
                     steps_per_second,
                     th0,
                     r,
                     l
                     )

    with open("../env/soft_reacher/chi.p", "rb") as inf:
        chi = pickle.load(inf)

    xticks_locations = np.arange(0, max_time + dt) * steps_per_second
    xticks = np.arange(0, max_time + dt, dtype=int)

    colors = [
        '#1f77b4',
        '#ff7f0e'
    ]
    line_styles = ['--', '-']
    error_line_styles = ['-', '-']
    markers = [None, None]
    factors = [0, -0.3]
    line_sizes = [3, 3]
    alphas = [1, 0.65]

    def _eps(x):
        if np.abs(x) < 1e-2:
            return 1e-2 * np.sign(x)
        else:
            return 0

    num_samples = len(observations_gt)
    indices = np.arange(0, num_samples, step=50, dtype=int)

    ee_gt = []
    ee_pred_l = {k: [] for k in observations_pred_l.keys()}

    for i in range(num_samples):
        q = observations_gt[i, :3]
        eps = _eps(q[0])
        ee_gt.append(chi([th0, r, eps, *q], l))

        for k in observations_pred_l.keys():
            q = observations_pred_l[k][i, :3]
            eps = _eps(q[0])
            ee_pred_l[k].append(chi([th0, r, eps, *q], l))

    ee_gt = (np.array(ee_gt)[:, :2]).squeeze()
    ee_pred_l = {k: (np.array(ee_pred_l[k])[:, :2]).squeeze() for k in observations_pred_l.keys()}

    num_lines = len(ee_pred_l) + 1
    for j in range(num_lines):
        fig_p, axs_p = plt.subplots(figsize=(15, 10))
        fig_e, axs_e = plt.subplots(figsize=(15, 5))
        # labels = ['lnn_seed_4', 'mlp_seed_0']

        axs_p.plot(ee_gt[:, 0],
                   label="$x_{ee}$",
                   linewidth=3,
                   color=adjust_color_brightness(colors[0], 0.3),
                   marker='^',
                   markevery=50,
                   markersize=10
                   )
        axs_p.plot(ee_gt[:, 1],
                   label="$y_{ee}$",
                   linewidth=3,
                   color=adjust_color_brightness(colors[1], 0.3),
                   marker='^',
                   markevery=50,
                   markersize=10
                   )

        for i, k in enumerate(reversed(observations_pred_l.keys())):
            if i > j-1:
                continue
            axs_p.plot(ee_pred_l[k][:, 0],
                       label="$\\tilde{x}_{ee}^{" + k.upper() + "}$",
                       color=adjust_color_brightness(colors[0], factors[i]),
                       linewidth=line_sizes[i],
                       linestyle=line_styles[i],
                       marker=markers[i],
                       markevery=50,
                       markersize=10,
                       alpha=alphas[i]
                       )
            axs_p.plot(ee_pred_l[k][:, 1],
                       label="$\\tilde{y}_{ee}^{" + k.upper() + "}$",
                       color=adjust_color_brightness(colors[1], factors[i]),
                       linewidth=line_sizes[i],
                       linestyle=line_styles[i],
                       marker=markers[i],
                       markevery=50,
                       markersize=10,
                       alpha=alphas[i]
                       )

        axs_p.set_title("End effector position")
        fig_p.legend(loc='upper left', ncol=6)
        axs_p.set_xticks(xticks_locations, xticks)
        axs_p.grid()

        errors = {
            k: np.abs(ee_gt - ee_pred_l[k])
            for k in observations_pred_l.keys()
        }

        for i, k in enumerate(reversed(observations_pred_l.keys())):
            if i > j-1:
                continue
            axs_e.plot(errors[k][:, 0],
                       label="$\\Delta x_{ee}^{" + k.upper() + "}$",
                       color=adjust_color_brightness(colors[0], factors[i]),
                       linewidth=line_sizes[i],
                       linestyle=error_line_styles[i],
                       marker=markers[i],
                       markevery=50,
                       markersize=10
                       )
            axs_e.plot(errors[k][:, 1],
                       label="$\\Delta y_{ee}^{" + k.upper() + "}$",
                       color=adjust_color_brightness(colors[1], factors[i]),
                       linewidth=line_sizes[i],
                       linestyle=error_line_styles[i],
                       marker=markers[i],
                       markevery=50,
                       markersize=10
                       )

        axs_e.set_title("End effector position error")
        fig_e.legend(loc='upper left', ncol=4)
        axs_e.set_xticks(xticks_locations, xticks)
        axs_e.grid()

        axs_p.set_xlabel("$[s]$")
        axs_p.set_ylabel("$[m]$")
        axs_e.set_xlabel("$[s]$")
        axs_e.set_ylabel("$[m]$")

        axs_p.set_ylim(lims['p'])
        axs_e.set_ylim(lims['e'])

        fig_p.tight_layout()
        fig_e.tight_layout()
        fig_p.subplots_adjust(top=0.85)
        fig_e.subplots_adjust(top=0.7)

        if save:
            save_path = f"{os.path.dirname(__file__)}/plots"
            print("p ", axs_p.get_ylim())
            print("e ", axs_e.get_ylim())

            fig_p.savefig(os.path.join(save_path, f"ee_{j}.png"), bbox_inches='tight')
            fig_e.savefig(os.path.join(save_path, f"ee_error_{j}.png"), bbox_inches='tight')
        else:
            plt.tight_layout()
            plt.show()


@torch.no_grad()
def rollout_plots(env, models, epoch, render=False, save=False, save_path=None, skip=None):
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

    if skip is None:
        skip = []
    dt = env.dt
    max_time = 9  # seconds
    # max_time = 1  # seconds
    max_steps = int(max_time / dt)
    act_size = env.action_size
    steps_per_second = int(1 / dt)

    sched = [(0, 1, 0),
             (1, 0, -1),
             (-1, 0, 0),
             (1, -1, 0),
             (0, 0, 1),
             (1, -1, 0),
             (-1, 0, 0),
             (1, 0, -1),
             (0, 1, 0)]

    # horizon_length = 16

    a_bounds = 1. * env.a_scale

    o_t, _, _ = env.reset()

    # num_seeds = len(list(models.values())[0])
    num_seeds = 1
    observations_gt = [o_t]
    observations_pred_l = {k: [o_t] for k in models.keys()}
    actions = []

    # get the device of the model
    device = next(list(models.values())[0].parameters()).device

    o_t_l = {k: torch.tensor(o_t).unsqueeze(0).to(device) for k in models.keys()}
    pbar = tqdm(range(max_steps))

    g = 0
    for _ in range(max_time):
        # act_ones = np.random.choice([-1, 0, 1], size=act_size)
        act_ones = sched[g]
        g += 1
        act = torch.tensor(act_ones * a_bounds).unsqueeze(0).to(device)
        for _ in range(steps_per_second):
            actions.append(act_ones)
            o_t_1_gt, _, _ = env.step(act_ones, last=True)
            observations_gt.append(o_t_1_gt)
            for model_name in models.keys():
                model = models[model_name]
                o_t_1_pred = model(o_t_l[model_name], act, train=False)
                observations_pred_l[model_name].append(o_t_1_pred.cpu().numpy().squeeze())
                o_t_l[model_name] = o_t_1_pred

            pbar.update(1)
            if render:
                env.render(other=observations_pred_l['lnn'][-1])

    # Compute metrics
    # metrics = {model: compute_metrics(np.array(observations_gt[1:]), np.array(observations_pred_l[model][1:]))
    #            for model in models.keys()}
    # print(metrics)

    # Convert to numpy arrays
    observations_gt = np.array(observations_gt)
    observations_pred_l = {model: np.array(observations_pred_l[model]) for model in models.keys()}
    actions = np.array(actions)

    # Plot the states
    # plot_state(observations_gt,
    #            observations_pred_l,
    #            actions,
    #            epoch=epoch,
    #            save=save,
    #            skip=skip,
    #            max_time=max_time,
    #            dt=dt,
    #            steps_per_second=steps_per_second)

    # Plot the end effector position
    plot_ees(observations_gt,
             observations_pred_l,
             epoch=epoch,
             save=save,
             skip=skip,
             max_time=max_time,
             dt=dt,
             steps_per_second=steps_per_second)


if __name__ == "__main__":
    base_path = os.path.join(os.path.dirname(__file__), "../FINAL")
    lnn = True
    mlp = True
    env = SoftReacher(mle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if lnn and mlp:
        best_seed_lnn = 2
        best_seed_mlp = 0
        models = {
            "lnn": LNN(
                env.name,
                env.n,
                env.obs_size,
                env.action_size,
                env.dt,
                env.dt_small,
                None).to(device),
            "mlp": MLP(env.obs_size, env.action_size)
        }
        file_lnn = os.path.join(base_path, "model_lnn", f"seed_{best_seed_lnn}_lr_3e-04", "only_model.ckpt")
        checkpoint = torch.load(file_lnn, map_location=device)
        r = models["lnn"].load_state_dict(checkpoint['transition_model'], strict=False)
        print("Loading weights:", r, file=sys.stderr)

        file_mlp = os.path.join(base_path, "model_mlp", f"seed_{best_seed_mlp}_lr_3e-04", "only_model.ckpt")
        checkpoint = torch.load(file_mlp, map_location=device)
        r = models["mlp"].load_state_dict(checkpoint['transition_model'], strict=False)
        print("Loading weights:", r, file=sys.stderr)

    elif lnn:
        models_lnn = {k: [LNN(
            env.name,
            env.n,
            env.obs_size,
            env.action_size,
            env.dt,
            env.dt_small,
            None).to(device) for _ in range(5)] for k in ["3e-03", "3e-04", "3e-05"]}

        for i in range(5):
            if i != 2:
                continue
            # for lr in ["3e-03", "3e-04", "3e-05"]:
            for lr in ["3e-04"]:
                file = os.path.join(base_path, "model_lnn", f"seed_{i}_lr_{lr}", "only_model.ckpt")
                checkpoint = torch.load(file, map_location=device)
                model = models_lnn[lr][i]
                r = model.load_state_dict(checkpoint['transition_model'], strict=False)
                print("Loading weights:", r, file=sys.stderr)

    elif mlp:
        models_mlp = {k: [MLP(env.obs_size, env.action_size) for _ in range(5)] for k in ["3e-03", "3e-04", "3e-05"]}

        for i in range(5):
            if i != 0:
                continue
            for lr in ["3e-03", "3e-04", "3e-05"]:
                file = os.path.join(base_path, "model_mlp", f"seed_{i}_lr_{lr}", "only_model.ckpt")
                checkpoint = torch.load(file, map_location=device)
                model = models_mlp[lr][i]
                r = model.load_state_dict(checkpoint['transition_model'], strict=False)
                print("Loading weights:", r, file=sys.stderr)

    rollout_plots(env, models, 1, save=True, render=False)


# j i
# 0 x
# 1 0
# 2 0
# 2 1


