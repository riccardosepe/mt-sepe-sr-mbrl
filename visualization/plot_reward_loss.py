import argparse

import matplotlib.pyplot as plt
import json
import os

import numpy as np

from utils.utils import format_label

with open(f"{os.path.dirname(__file__)}/../utils/rcparams2.json", "r") as f:
    plt.rcParams.update(json.load(f))


def plot2(model, input_file, lrs=None, save=True):
    if lrs is None:
        lrs = ["3e-04"]
    with open(input_file) as f:
        data = json.load(f)

    fig, ax = plt.subplots()
    v = {lr: [] for lr in lrs}
    for run, run_data in data.items():
        lr = run.split("_")[-1]
        if lr not in lrs:
            continue
        values = run_data["transition_loss"]["values"]
        steps = run_data["transition_loss"]["steps"]
        v[lr].append(values)

    max_s = max(steps) + 1
    r = 20
    r_fill = r + 3
    min_overall = np.inf
    add_data = {lr: {'mean': None, 'color': None} for lr in lrs}
    for lr, values in v.items():
        values = np.array(values)
        means = np.mean(values, axis=0)
        stds = np.std(values, axis=0)
        mins = np.min(values, axis=0)
        maxs = np.max(values, axis=0)
        min_overall = min(min_overall, np.min(mins))
        line_above = means + stds
        line_below = means - stds
        p = ax.plot(steps[r:], means[r:], label=f"lr={lr}")
        color = p[0].get_color()
        ax.plot(steps[r:], line_below[r:], linewidth=0.5, color=color)
        ax.plot(steps[r_fill:], line_above[r_fill:], linewidth=0.5, color=color)
        ax.fill_between(steps[r_fill:], line_below[r_fill:], line_above[r_fill:], alpha=0.3)
        add_data[lr]['mean'] = means
        add_data[lr]['color'] = color

    # ax.axhline(min_overall, color='green', linestyle='--', label="min overall")
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    ymin, ymax = ylim
    ylabels_v = [0.015, 0.02, 0.025, 0.03]
    ylabels = [format_label(l) for l in ylabels_v]
    # ylabels = np.linspace(int(ymin*1000 + 0.5)/1000, int(ymax*1000 + 0.5)/1000, 3)

    inc = (ylim[1] - ylim[0]) * 0.1
    ylim = (ylim[0], ylim[1] + inc)

    for lr, values in v.items():
        means = add_data[lr]['mean']
        color = add_data[lr]['color']
        plt.plot(steps[:r+1], means[:r+1], linestyle='--', color=color)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_yscale('log')
    ax.set_yticks(ylabels_v, ylabels)

    ax.set_xlabel("epochs")
    ax.set_ylabel("L1 loss")

    assert len(lrs) == 1 or len(lrs) == 3
    if len(lrs) == 1:
        lr = lrs[0]
    else:
        lr = "all"

    if save:
        path = f"{os.path.dirname(__file__)}/../plots/{model}_{lr}_loss.png"
        plt.savefig(path, bbox_inches='tight')
    else:
        plt.tight_layout()
        plt.show()


def plot(save=True):
    lrs = ["3e-03", "3e-04", "3e-05"]
    base_path = os.path.join(os.path.dirname(__file__), "..", "FINAL")
    mlp_path = os.path.join(base_path, "model_mlp_json", "tensorboard.json")
    lnn_path = os.path.join(base_path, "model_lnn_json", "tensorboard.json")

    with open(mlp_path) as f:
        data_mlp = json.load(f)

    with open(lnn_path) as f:
        data_lnn = json.load(f)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 6), sharey="all")
    v = {'lnn': {lr: [] for lr in lrs},
         'mlp': {lr: [] for lr in lrs}}

    for run, run_data in data_lnn.items():
        lr = run.split("_")[-1]
        if lr not in lrs:
            continue
        values = run_data["reward_loss"]["values"]
        steps = run_data["reward_loss"]["steps"]
        v['lnn'][lr].append(values)

    for run, run_data in data_mlp.items():
        lr = run.split("_")[-1]
        if lr not in lrs:
            continue
        values = run_data["reward_loss"]["values"]
        steps = run_data["reward_loss"]["steps"]
        v['mlp'][lr].append(values)

    r = 0
    r_fill = r + 3
    min_overall = np.inf
    add_data = {lr: {'mean': None, 'color': None} for lr in lrs}
    colors = ['#ff7f0e', '#1f77b4', '#2ca02c']
    indices = {'lnn': 1, 'mlp': 0}
    for j, (model_name, data) in enumerate(v.items()):
        for i, (lr, values) in enumerate(data.items()):
            values = np.array(values)
            means = np.mean(values, axis=0)
            stds = np.std(values, axis=0)
            mins = np.min(values, axis=0)
            maxs = np.max(values, axis=0)
            min_overall = min(min_overall, np.min(mins))
            line_above = means + stds
            line_below = means - stds
            if j==0:
                ax[indices[model_name]].plot(steps[r:], means[r:], label=f"lr={lr}", color=colors[i])
            else:
                ax[indices[model_name]].plot(steps[r:], means[r:], color=colors[i])
            # color = p[0].get_color()
            # print(f"Color for lr={lr}: {color}")
            ax[indices[model_name]].plot(steps[r:], line_below[r:], linewidth=0.5, color=colors[i])
            ax[indices[model_name]].plot(steps[r_fill:], line_above[r_fill:], linewidth=0.5, color=colors[i])
            ax[indices[model_name]].fill_between(steps[r_fill:], line_below[r_fill:], line_above[r_fill:], alpha=0.3, color=colors[i])
            add_data[lr]['mean'] = means
            add_data[lr]['color'] = colors[i]
            ax[indices[model_name]].set_yscale('log')
            ax[indices[model_name]].set_xlabel("epochs")
            ax[indices[model_name]].set_ylabel("L1 loss")
            ax[indices[model_name]].set_title(f"{model_name.upper()} reward loss")

    # ax.axhline(min_overall, color='green', linestyle='--', label="min overall")
    # xlim = ax.get_xlim()
    ylim = ax[0].get_ylim()

    # ymin, ymax = ylim
    # ylabels = np.linspace(int(ymin*1000 + 0.5)/1000, int(ymax*1000 + 0.5)/1000, 3)

    # inc = (ylim[1] - ylim[0]) * 0.1
    # ylim = (ylim[0], ylim[1] + inc)

    ylim = (1e-3, 1e-2)
    # ax.set_ylim(ylim)

    # for lr, values in v.items():
    #     means = add_data[lr]['mean']
    #     color = add_data[lr]['color']
    #     plt.plot(steps[:r+1], means[:r+1], linestyle='--', color=color)

    ylabels_v = [0.0005, 0.001, 0.005, 0.01]
    ylabels = [format_label(l) for l in ylabels_v]

    ax[0].set_yticks(ylabels_v, ylabels)
    ax[1].set_yticks(ylabels_v, ylabels)
    ax[1].yaxis.set_tick_params(labelleft=True)

    # if len(lrs) == 3:
    #     ax.legend()

    # ax.set_xlabel("epochs")
    # ax.set_ylabel("L1 loss")

    fig.tight_layout()
    fig.subplots_adjust(top=0.8)
    fig.legend(loc='upper center', ncols=3)

    assert len(lrs) == 1 or len(lrs) == 3
    if len(lrs) == 1:
        lr = lrs[0]
    else:
        lr = "all"

    if save:
        path = f"{os.path.dirname(__file__)}/../plots/reward_{lr}_loss.png"
        plt.savefig(path, bbox_inches='tight')
    else:
        plt.show()


if __name__ == '__main__':
    plot(save=True)
