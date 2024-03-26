import matplotlib.pyplot as plt
import json
import os

import numpy as np
import dill as pickle

from utils.utils import format_label, adjust_color_brightness


pickle.settings['recurse'] = True


with open(f"{os.path.dirname(__file__)}/../utils/rcparams2.json", "r") as f:
    plt.rcParams.update(json.load(f))


def _eps(x):
    if np.abs(x) < 1e-2:
        return 1e-2 * np.sign(x)
    else:
        return 0


def plot(save=False):
    data_env = np.load(f"{os.path.dirname(__file__)}/../FINAL/eval/data_env.npy", allow_pickle=True).item()
    data_mlp = np.load(f"{os.path.dirname(__file__)}/../FINAL/eval/data_mlp.npy", allow_pickle=True).item()
    data_lnn = np.load(f"{os.path.dirname(__file__)}/../FINAL/eval/data_lnn.npy", allow_pickle=True).item()

    fig, ax = plt.subplots(nrows=2,
                           ncols=1,
                           figsize=(12, 8))

    ee_env = np.array(data_env['ee'][0])
    ee_mlp = np.array(data_mlp['ee'][0])
    ee_lnn = np.array(data_lnn['ee'][0])

    goal = data_env['goal']

    colors = [
        '#1f77b4',
        '#ff7f0e'
    ]

    b = ['$x', '$y']

    num_steps = len(ee_env)
    marker_every = np.arange(0, num_steps-1, 100)
    # marker_every = list(marker_every)
    for i in range(2):
        ax[i].axhline(goal[i],
                   color=adjust_color_brightness(colors[i], 0.3),
                   linestyle='--',
                   label=b[i]+"_{goal}$",
                   linewidth=7,
                   alpha=1
                   )

        ax[i].plot(ee_env[:, i],
                label=b[i]+"^{ENV}$",
                linewidth=3,
                color=adjust_color_brightness(colors[i], -0.15),
                marker='o',
                markevery=10+marker_every,
                markersize=10,
                alpha=1
                )

        ax[i].plot(ee_lnn[:, i],
                label=b[i]+"^{LNN}$",
                color=adjust_color_brightness(colors[i], -0.5),
                linestyle='-',
                # marker='^',
                # markersize=10,
                # markevery=50+marker_every,
                linewidth=3,
                alpha=1
                )

        ax[i].plot(ee_mlp[:, i],
                label=b[i]+"^{MLP}$",
                color=adjust_color_brightness(colors[i], -0.15),
                linestyle='-',
                linewidth=3,
                alpha=1
                )
        ax[i].grid()
        ax[i].legend(ncol=4)


    # fig.legend(loc='upper center', ncol=4)
    fig.tight_layout()
    # fig.subplots_adjust(top=0.82)

    plt.show()


def plot2(save=False):
    data_env = np.load(f"{os.path.dirname(__file__)}/../FINAL/eval/data_env.npy", allow_pickle=True).item()
    data_mlp = np.load(f"{os.path.dirname(__file__)}/../FINAL/eval/data_mlp.npy", allow_pickle=True).item()
    data_lnn = np.load(f"{os.path.dirname(__file__)}/../FINAL/eval/data_lnn.npy", allow_pickle=True).item()

    fig, ax = plt.subplots(figsize=(12, 8))

    ee_env = np.array(data_env['ee'][0])
    ee_mlp = np.array(data_mlp['ee'][0])
    ee_lnn = np.array(data_lnn['ee'][0])

    goal = data_env['goal']

    colors = [
        '#1f77b4',
        '#ff7f0e'
    ]

    b = ['$x', '$y']

    num_steps = len(ee_env)
    marker_every = np.arange(0, num_steps-1, 100)
    # marker_every = list(marker_every)
    for i in range(2):
        ax.axhline(goal[i],
                   color=adjust_color_brightness(colors[i], 0.3),
                   linestyle='--',
                   label=b[i]+"_{goal}$",
                   linewidth=7,
                   alpha=1
                   )

        ax.plot(ee_env[:, i],
                label=b[i]+"^{ENV}$",
                linewidth=3,
                color=adjust_color_brightness(colors[i], -0.15),
                marker='o',
                markevery=10+marker_every,
                markersize=10,
                alpha=1
                )

        ax.plot(ee_lnn[:, i],
                label=b[i]+"^{LNN}$",
                color=adjust_color_brightness(colors[i], -0.5),
                linestyle='-',
                # marker='^',
                # markersize=10,
                # markevery=50+marker_every,
                linewidth=3,
                alpha=1
                )

        ax.plot(ee_mlp[:, i],
                label=b[i]+"^{MLP}$",
                color=adjust_color_brightness(colors[i], -0.15),
                linestyle='-',
                linewidth=3,
                alpha=1
                )


    fig.legend(loc='upper center', ncol=4)
    fig.tight_layout()
    fig.subplots_adjust(top=0.82)
    ax.grid()

    plt.show()


if __name__ == "__main__":
    plot(save=False)
