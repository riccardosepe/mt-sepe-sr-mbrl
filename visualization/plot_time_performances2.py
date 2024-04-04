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
    env_path = os.path.join(base_path, "sac_on_env_json", "tensorboard.json")

    with open(mlp_path) as f:
        data_mlp = json.load(f)
    with open(lnn_path) as f:
        data_lnn = json.load(f)
    with open(env_path) as f:
        data_env = json.load(f)

    returns = {"env": [], "mlp": [], "lnn": []}
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = {'env': '#ff7f0e', 'mlp': '#2ca02c', 'lnn': '#1f77b4'}

    steps = None
    for run, run_data in data_mlp.items():
        returns['mlp'].append(run_data["eval_ep_r"]['values'])
        if steps is None:
            steps = run_data["eval_ep_r"]['steps']

    for run, run_data in data_lnn.items():
        returns['lnn'].append(run_data["eval_ep_r"]['values'])

    for run, run_data in data_env.items():
        returns['env'].append(run_data["eval_ep_r"]['values'])

    lnnd = np.array(returns['lnn']).mean(0)
    mlpd = np.array(returns['mlp']).mean(0)
    envd = np.array(returns['env']).mean(0)

    ax.axhline(lnnd.max(), lw=3, linestyle='--', color=colors['lnn'])
    ax.axhline(mlpd.max(), lw=3, linestyle='--', color=colors['mlp'])
    ax.axhline(envd.max(), lw=3, linestyle='--', color=colors['env'])

    ax.axvline(4, lw=3, ls='--', color=colors['lnn'])
    # ax.axvline(mlp_x, lw=3, dashes=(6, (4, 8)), color=colors['mlp'])
    ax.axvline(450, lw=3, linestyle='--', color=colors['env'])


    x = [4] * len(steps)

    steps = np.array(steps)
    steps[-1] += 1

    ax.scatter(x, list(sorted(lnnd)), label='LNN', s=100, color=colors['lnn'], edgecolor='k', lw=2, zorder=2)
    ax.scatter(x, mlpd, label='MLP', s=100, color=colors['mlp'], edgecolor='k', lw=2, zorder=2)
    ax.scatter(steps[:], envd[:], label='ref.', s=100, color=colors['env'], edgecolor='k', lw=2, zorder=2)

    ax.set_xlim(xmin=-10)

    ax.set_xticks(steps[::2], steps[::2]*10)

    ax.set_xlabel('robot interaction time [s]')
    ax.set_ylabel('return')

    ax.annotate('\\textbf{1.0e5}', (4, lnnd.max()), xytext=(40, 10), textcoords='offset points', ha='center', color=colors['lnn'])
    ax.annotate('\\textbf{0}', (4, mlpd.max()), xytext=(20, 10), textcoords='offset points', ha='center', color=colors['mlp'])
    ax.annotate('\\textbf{4.5e5}', (450, envd.max()), xytext=(-35, -30), textcoords='offset points', ha='center', color=colors['env'])

    fig.legend(loc='center')
    fig.tight_layout()

    if save:
        path = f"{os.path.dirname(__file__)}/../plots/sac_times.png"
        plt.savefig(path, bbox_inches='tight')
    else:
        plt.show()


if __name__ == '__main__':
    plot(True)
