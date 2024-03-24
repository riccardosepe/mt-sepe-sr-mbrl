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


def plot(model, input_file, lrs=None, save=True):
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
        values = run_data["reward_loss"]["values"]
        steps = run_data["reward_loss"]["steps"]
        v[lr].append(values)

    max_s = max(steps) + 1
    r = 0
    r_fill = r + 3
    min_overall = np.inf
    add_data = {lr: {'mean': None, 'color': None} for lr in lrs}
    colors = ['#ff7f0e', '#1f77b4', '#2ca02c']
    for i, (lr, values) in enumerate(v.items()):
        values = np.array(values)
        means = np.mean(values, axis=0)
        stds = np.std(values, axis=0)
        mins = np.min(values, axis=0)
        maxs = np.max(values, axis=0)
        min_overall = min(min_overall, np.min(mins))
        line_above = means + stds
        line_below = means - stds
        p = ax.plot(steps[r:], means[r:], label=f"lr={lr}", color=colors[i])
        # color = p[0].get_color()
        # print(f"Color for lr={lr}: {color}")
        ax.plot(steps[r:], line_below[r:], linewidth=0.5, color=colors[i])
        ax.plot(steps[r_fill:], line_above[r_fill:], linewidth=0.5, color=colors[i])
        ax.fill_between(steps[r_fill:], line_below[r_fill:], line_above[r_fill:], alpha=0.3, color=colors[i])
        add_data[lr]['mean'] = means
        add_data[lr]['color'] = colors[i]

    # ax.axhline(min_overall, color='green', linestyle='--', label="min overall")
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    ymin, ymax = ylim
    # ylabels = np.linspace(int(ymin*1000 + 0.5)/1000, int(ymax*1000 + 0.5)/1000, 3)

    inc = (ylim[1] - ylim[0]) * 0.1
    ylim = (ylim[0], ylim[1] + inc)

    ylim = (1e-3, 1e-2)
    ax.set_ylim(ylim)

    # for lr, values in v.items():
    #     means = add_data[lr]['mean']
    #     color = add_data[lr]['color']
    #     plt.plot(steps[:r+1], means[:r+1], linestyle='--', color=color)

    # ax.set_xlim(xlim)
    # ax.set_ylim(ylim)
    ax.set_yscale('log')
    ylabels_v = [0.001, 0.002, 0.003, 0.004, 0.006, 0.01]
    ylabels = [format_label(l) for l in ylabels_v]

    ax.set_yticks(ylabels_v, ylabels)

    if len(lrs) == 3:
        ax.legend()

    ax.set_xlabel("epochs")
    ax.set_ylabel("L1 loss")

    assert len(lrs) == 1 or len(lrs) == 3
    if len(lrs) == 1:
        lr = lrs[0]
    else:
        lr = "all"

    if save:
        path = f"{os.path.dirname(__file__)}/../plots/{model}_reward_{lr}_loss.png"
        plt.savefig(path, bbox_inches='tight')
    else:
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Plot model losses")
    parser.add_argument("--input", type=str, help="Path to the JSON file", required=True)
    parser.add_argument("--model", type=str, help="Model name [lnn / mlp]", required=True)
    parser.add_argument("--lr-exp", type=str, help="Learning rate exponent", default="4")

    args = parser.parse_args()

    if args.lr_exp == 'all':
        lrs = ["3e-03", "3e-04", "3e-05"]
    else:
        lrs = [f"3e-0{args.lr_exp}"]

    plot(args.model, args.input, lrs)
