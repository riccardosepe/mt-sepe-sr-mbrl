import json
import os

import matplotlib.pyplot as plt
import numpy as np

from utils.utils import format_label, smooth, adjust_color_brightness

with open(f"{os.path.dirname(__file__)}/../utils/rcparams2.json", "r") as f:
    plt.rcParams.update(json.load(f))


def plot(save=False):
    data_path = os.path.join(os.path.dirname(__file__), "..", "FINAL", "eval", "internal_states.npy")
    data = np.load(data_path, allow_pickle=True).item()
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

    env_data = data['env'].squeeze()
    mlp_data = data['model_mlp'].squeeze()
    lnn_data = data['model_lnn'].squeeze()

    ax[0].axhline(0.05, color='green', lw=2, ls='--')
    ax[1].axhline(0.08, color='green', lw=2, ls='--')

    for i in range(2):
        ax[i].plot(env_data[:, i], label='ref.', lw=2)
        ax[i].plot(mlp_data[:, i], label='mlp', lw=2)
        ax[i].plot(lnn_data[:, i], label='lnn', lw=2)

    ax[0].legend()
    ax[1].legend()

    plt.show()


if __name__ == "__main__":
    plot()
