import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from utils.utils import format_label, smooth, adjust_color_brightness

with open(f"{os.path.dirname(__file__)}/../utils/rcparams2.json", "r") as f:
    plt.rcParams.update(json.load(f))


def plot(save=False):
    env = True
    base_path = os.path.join(os.path.dirname(__file__), "..", "FINAL")
    mlp_path = os.path.join(base_path, "sac_on_mlp_json", "tensorboard.json")
    lnn_path = os.path.join(base_path, "sac_on_model_json", "tensorboard.json")
    env_path = os.path.join(base_path, "sac_on_env_json", "tensorboard.json")

    with open(mlp_path) as f:
        data_mlp = json.load(f)
    with open(lnn_path) as f:
        data_lnn = json.load(f)
    if env:
        with open(env_path) as f:
            data_env = json.load(f)

    fig, ax = plt.subplots(figsize=(9, 6))
    returns = {"env": [], "mlp": [], "lnn": []}
    colors = ['#ff7f0e', '#2ca02c', '#1f77b4']

    steps = None
    for run, run_data in data_mlp.items():
        returns['mlp'].append(run_data["eval_ep_r"]['values'])
        if steps is None:
            steps = run_data["eval_ep_r"]['steps']

    for run, run_data in data_lnn.items():
        returns['lnn'].append(run_data["eval_ep_r"]['values'])

    for run, run_data in data_env.items():
        returns['env'].append(run_data["eval_ep_r"]['values'])

    for i, model_name in enumerate(returns.keys()):
        returns[model_name] = np.array(returns[model_name])

        means_mean = np.mean(returns[model_name], axis=0)
        means_std = np.std(returns[model_name], axis=0)
        means_mins = np.min(returns[model_name], axis=0)
        means_maxs = np.max(returns[model_name], axis=0)
        means_line_above = means_maxs
        means_line_below = means_mins

        # means_mean = smooth(means_mean, 0.9)
        # means_line_above = smooth(means_line_above, 0.9)
        # means_line_below = smooth(means_line_below, 0.9)
        color = colors[i]
        if model_name == 'env':
            lab = 'ref.'
        else:
            lab = model_name.upper()
        p = ax.plot(steps, means_mean, label=lab, linewidth=3, color=colors[i])
        # color = p[0].get_color()
        ax.plot(steps, means_line_below, linewidth=0.5, color=color)
        ax.plot(steps, means_line_above, linewidth=0.5, color=color)
        ax.fill_between(steps, means_line_below, means_line_above, alpha=0.3, color=color)
        ax.axhline(1000, color='green', linestyle='--', linewidth=3)
        ax.set_title("Validation returns")

    xticks = np.arange(0, max(steps)+10, 100)
    xtickslabels = [format_label(x * 1000) for x in xticks]
    ax.set_xticks(xticks, xtickslabels)

    ax.legend()
    ax.set_xlabel("timesteps")
    ax.set_ylabel("returns")

    fig.tight_layout()
    if save:
        path = f"{os.path.dirname(__file__)}/../plots/sac_val_all_returns.png"
        plt.savefig(path, bbox_inches='tight')
    else:
        plt.show()


if __name__ == "__main__":
    plot(save=True)
