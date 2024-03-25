import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from utils.utils import format_label, smooth, adjust_color_brightness

with open(f"{os.path.dirname(__file__)}/../utils/rcparams2.json", "r") as f:
    plt.rcParams.update(json.load(f))


def plot(save=False):
    base_path = os.path.join(os.path.dirname(__file__), "..", "FINAL")
    mlp_path = os.path.join(base_path, "sac_on_mlp_json", "tensorboard.json")
    lnn_path = os.path.join(base_path, "sac_on_model_json", "tensorboard.json")
    with open(mlp_path) as f:
        data_mlp = json.load(f)
    with open(lnn_path) as f:
        data_lnn = json.load(f)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))
    means = {"lnn": [], "mlp": []}
    stds = {"lnn": [], "mlp": []}
    steps = None
    for run, run_data in data_mlp.items():
        means['mlp'].append(run_data["Return/Mean"]['values'])
        stds['mlp'].append(run_data["Return/Std"]['values'])
        if steps is None:
            steps = run_data["Return/Mean"]['steps']

    for run, run_data in data_lnn.items():
        means['lnn'].append(run_data["Return/Mean"]['values'])
        stds['lnn'].append(run_data["Return/Std"]['values'])

    for model_name in means.keys():
        means[model_name] = np.array(means[model_name])
        stds[model_name] = np.array(stds[model_name])

        means_mean = np.mean(means[model_name], axis=0)
        means_std = np.std(means[model_name], axis=0)
        means_mins = np.min(means[model_name], axis=0)
        means_maxs = np.max(means[model_name], axis=0)
        means_line_above = means_mean + means_std
        means_line_below = means_mean - means_std

        stds_mean = np.mean(stds[model_name], axis=0)
        stds_std = np.std(stds[model_name], axis=0)
        stds_mins = np.min(stds[model_name], axis=0)
        stds_maxs = np.max(stds[model_name], axis=0)
        stds_line_above = stds_mean + stds_std
        stds_line_below = stds_mean - stds_std

        means_mean = smooth(means_mean, 0.9)
        means_line_above = smooth(means_line_above, 0.9)
        means_line_below = smooth(means_line_below, 0.9)

        stds_mean = smooth(stds_mean, 0.9)
        stds_line_above = smooth(stds_line_above, 0.9)
        stds_line_below = smooth(stds_line_below, 0.9)

        p = ax[0].plot(means_mean, label=model_name.upper(), linewidth=3)
        color = p[0].get_color()
        ax[0].plot(means_line_below, linewidth=0.5, color=color)
        ax[0].plot(means_line_above, linewidth=0.5, color=color)
        ax[0].fill_between(range(len(means_mean)), means_line_below, means_line_above, alpha=0.3, color=color)
        ax[0].axhline(50, color='green', linestyle='--', linewidth=3)
        ax[0].set_title("Return mean")

        p = ax[1].plot(stds_mean, linewidth=3)
        color = p[0].get_color()
        ax[1].plot(stds_line_below, linewidth=0.5, color=color)
        ax[1].plot(stds_line_above, linewidth=0.5, color=color)
        ax[1].fill_between(range(len(stds_mean)), stds_line_below, stds_line_above, alpha=0.3, color=color)
        ax[1].axhline(0, color='green', linestyle='--', linewidth=3)
        ax[1].set_title("Return standard deviation")

    fig.legend(loc='upper center', ncols=2)

    fig.tight_layout()
    fig.subplots_adjust(top=0.85)
    if save:
        path = f"{os.path.dirname(__file__)}/../plots/sac_train_returns.png"
        plt.savefig(path, bbox_inches='tight')
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Folder with tensorboard.json file")
    args = parser.parse_args()

    plot(save=True)
