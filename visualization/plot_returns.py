import matplotlib.pyplot as plt
import json
import os
import argparse

import numpy as np


def format_label(x):
    if x == 0:
        return "0"
    else:
        e = int(np.log10(x))
        n = int(x / 10 ** e)
        return f"${n}$e${e}$"


def plot_returns(input_file, save=True):
    with open(input_file) as f:
        data = json.load(f)
    fig, ax = plt.subplots()
    v = []
    for run, run_data in data.items():
        values = run_data["eval_ep_r"]["values"]
        steps = run_data["eval_ep_r"]["steps"]
        # ax.plot(steps, values, label=run)
        v.append(values)
    max_s = max(steps) + 1
    v = np.array(v)
    means = np.mean(v, axis=0)
    stds = np.std(v, axis=0)
    mins = np.min(v, axis=0)
    maxs = np.max(v, axis=0)
    line_above = maxs
    line_below = mins
    p = ax.plot(steps, means)
    color = p[0].get_color()
    ax.axhline(1000, color='green', linestyle='--')
    ax.plot(steps, line_below, linewidth=0.5, color=color)
    ax.plot(steps, line_above, linewidth=0.5, color=color)
    ax.fill_between(steps, line_below, line_above, alpha=0.3)
    locations = np.arange(0, max_s+1, 100)
    labels_v = locations * 1000
    labels = [format_label(l) for l in labels_v]
    ax.set_xticks(locations, labels)
    ax.set_xlabel("timesteps")
    ax.set_ylabel("return")

    if save:
        path = f"{os.path.dirname(__file__)}/../plots/returns.png"
        plt.savefig(path, bbox_inches='tight')
    else:
        plt.tight_layout(pad=1.0)
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Plot returns")
    parser.add_argument("--input", type=str, help="Path to the JSON file", required=True)
    rcpath = os.path.join(os.path.dirname(__file__), "../utils/rcparams2.json")
    with open(rcpath) as f:
        rcparams = json.load(f)
    plt.rcParams.update(rcparams)
    args = parser.parse_args()
    plot_returns(args.input)
