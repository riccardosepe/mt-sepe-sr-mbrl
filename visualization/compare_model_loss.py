import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from utils.utils import format_label, smooth, adjust_color_brightness

with open(f"{os.path.dirname(__file__)}/../utils/rcparams2.json", "r") as f:
    plt.rcParams.update(json.load(f))


labels = {
    'lnn': "$\\mathcal{D}_s$",
    'mlp': "$\\mathcal{D}$"
    }

def plot(infolder, lrs=None, save=False):
    data = {"lnn": None, "mlp": None}
    for fo in data:
        path = os.path.join(infolder, f"model_{fo}_json", "tensorboard.json")
        with open(path) as f:
            data[fo] = json.load(f)

    fig, ax = plt.subplots()

    v = {'lnn': [], 'mlp': []}
    for model, model_data in data.items():
        for run, run_data in model_data.items():
            lrr = run.split("_")[-1]
            if lrr != "3e-04":
                continue
            values = run_data["transition_loss"]["values"]
            steps = run_data["transition_loss"]["steps"]
            v[model].append(values)

    r = 10
    min_overall = np.inf
    for i, (model, values) in enumerate(v.items()):
        if model == "lnn":
            r_fill = r
        else:
            r_fill = r + 5
        values = np.array(values)
        best_run = np.where(values == values.min())[0][0]
        means = np.mean(values, axis=0)
        stds = np.std(values, axis=0)
        mins = np.min(values, axis=0)
        maxs = np.max(values, axis=0)
        min_overall = min(min_overall, np.min(mins))
        line_above = means + stds
        line_below = means - stds
        p = ax.plot(steps[r:], means[r:], label=model.upper())
        color = p[0].get_color()

        ax.plot(steps[r:], line_below[r:], linewidth=0.5, color=color)
        ax.plot(steps[r_fill:], line_above[r_fill:], linewidth=0.5, color=color)
        ax.fill_between(steps[r_fill:], line_below[r_fill:], line_above[r_fill:], alpha=0.3, color=color)

        ax.plot(steps[r:], smooth(values[best_run, r:], 0.8), linestyle='--', linewidth=3, color=adjust_color_brightness(color, 0.3), label=f"best {model.upper()} run")
    # ax.axhline(min_overall, color='green', linestyle='--', label="min overall")
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    ymin, ymax = ylim
    # ylabels = np.linspace(int(ymin*1000 + 0.5)/1000, int(ymax*1000 + 0.5)/1000, 3)

    inc = (ylim[1] - ylim[0]) * 0.1
    ylim = (ylim[0], ylim[1] + inc)

    ylim = (1.3e-2, 0.8e-1)
    # ax.set_ylim(ylim)

    # for lr, values in v.items():
    #     means = add_data[lr]['mean']
    #     color = add_data[lr]['color']
    #     plt.plot(steps[:r+1], means[:r+1], linestyle='--', color=color)

    # ax.set_xlim(xlim)
    # ax.set_ylim(ylim)
    ax.set_yscale('log')
    ylabels_v = [0.01, 0.02, 0.03, 0.04]
    ylabels = [format_label(l) for l in ylabels_v]

    ax.set_yticks(ylabels_v, ylabels)

    ax.legend()

    ax.set_xlabel("epochs")
    ax.set_ylabel("L1 loss")

    if save:
        path = f"{os.path.dirname(__file__)}/../plots/compare_model_loss.png"
        plt.savefig(path, bbox_inches='tight')
    else:
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--folder", type=str, help="Folder containing the JSON files", required=True)
    #
    # args = parser.parse_args()
    folder = "../FINAL"

    plot(folder, save=True)
