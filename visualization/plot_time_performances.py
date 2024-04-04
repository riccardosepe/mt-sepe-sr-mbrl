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

    lnnd = np.array(returns['lnn'])
    mlpd = np.array(returns['mlp'])
    envd = np.array(returns['env'])

    env_d = (4, 10)
    env_x = int(4.5e3)
    env_y = 50
    env_y_l = envd[env_d]

    lnn_d = (3, 2)
    lnn_x = 40
    lnn_y = 45
    lnn_y_l = lnnd[lnn_d]

    mlp_d = (0, 0)
    mlp_x = 40
    mlp_y = 10
    mlp_y_l = mlpd[mlp_d]

    ax.axhline(lnn_y, lw=3, linestyle='--', color=colors['lnn'])
    ax.axhline(mlp_y, lw=3, linestyle='--', color=colors['mlp'])
    ax.axhline(env_y, lw=3, linestyle='--', color=colors['env'])

    ax.axvline(mlp_x, lw=3, ls='--', color=colors['lnn'])
    # ax.axvline(mlp_x, lw=3, dashes=(6, (4, 8)), color=colors['mlp'])
    ax.axvline(env_x, lw=3, linestyle='--', color=colors['env'])

    ax.scatter(lnn_x, lnn_y, label='LNN', s=500, color=colors['lnn'], edgecolor='k', lw=2, zorder=2)
    ax.scatter(mlp_x, mlp_y, label='MLP', s=500, color=colors['mlp'], edgecolor='k', lw=2, zorder=2)
    ax.scatter(env_x, env_y, label='ref.', s=500, color=colors['env'], edgecolor='k', lw=2, zorder=2)

    ax.annotate('\\textbf{1.0e5}', (lnn_x, lnn_y), xytext=(40, 10), textcoords='offset points', ha='center', color=colors['lnn'])
    ax.annotate('\\textbf{0}', (mlp_x, mlp_y), xytext=(20, 10), textcoords='offset points', ha='center', color=colors['mlp'])
    ax.annotate('\\textbf{4.5e5}', (env_x, env_y), xytext=(-35, -30), textcoords='offset points', ha='center', color=colors['env'])

    ax.legend()

    ax.set_yticks([mlp_y, lnn_y, env_y], [f"{np.round(mlp_y_l, decimals=3)}",
                                          f"{np.round(lnn_y_l, decimals=3)}",
                                          f"{np.round(env_y_l, decimals=3)}"])
    ax.set_xticks([mlp_x, env_x])

    ax.set_xlabel('[s]')
    ax.set_ylabel('return')

    fig.tight_layout()

    if save:
        path = f"{os.path.dirname(__file__)}/../plots/sac_times.png"
        plt.savefig(path, bbox_inches='tight')
    else:
        plt.show()


if __name__ == '__main__':
    plot(True)
