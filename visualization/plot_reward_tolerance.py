import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from env.rewards import tolerance

import dill as pickle
pickle.settings['recurse'] = True


def tolerances():
    xmax = 1.5e-1
    # x = np.linspace(-xmax, xmax, 200)
    # y = np.linspace(0, 2e-1, 100)
    #
    # X, Y = np.meshgrid(x, y)
    # Z = np.zeros(X.shape)
    #
    # for i in range(X.shape[0]):
    #     for j in range(X.shape[1]):
    #         Z[i, j] = tolerance(X[i, j], margin=Y[i, j])

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(X, Y, Z, alpha=0.7)
    #
    # # Add a line at x = 1e-1
    # x_line = 1e-1
    # y_line = np.linspace(0, 2e-1, 100)
    # z_line = np.array([tolerance(x_line, margin=yi) for yi in y_line])
    # ax.plot([x_line] * len(y_line), y_line, z_line, color='r', linewidth=4)
    #
    # ax.set_xlabel('x')
    # ax.set_ylabel('margin')
    # ax.set_zlabel('tolerance')

    fig2, ax2 = plt.subplots()
    # Add axes
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.axvline(0, color='black', linewidth=0.8)
    # Add a line at y = 1e-2
    y_line = 5e-2
    x_line = np.linspace(-xmax, xmax, 200)
    z_line = np.array([tolerance(xi, margin=y_line) for xi in x_line])
    ax2.plot(x_line, z_line, linewidth=2)

    ax2.set_xlabel('$x$')
    ax2.set_ylabel('$\\bar{t}(x,'+str(y_line)+')$')
    ax2.grid(True)

    ax2.axhline(0.1, color='black', linewidth=0.8, linestyle='--', xmin=0, xmax=y_line+0.6)
    ax2.axvline(y_line, color='black', linewidth=0.8, linestyle='--', ymin=0, ymax=0.09+0.05)
    ax2.axvline(-y_line, color='black', linewidth=0.8, linestyle='--', ymin=0, ymax=0.09+0.05)

    ax2.scatter(-y_line, 0.1, marker='o', color='red', s=50, zorder=2)
    ax2.scatter(y_line, 0.1, marker='o', color='red', s=50, zorder=2)

    yticks_pos = np.linspace(0, 1, 6).tolist()
    yticks_pos.append(0.1)
    yticks_pos = np.array(list(sorted(yticks_pos)))
    yticks = [f'{y:.2f}' for y in yticks_pos]
    ax2.set_yticks(yticks_pos, yticks)
    # ax2.set_xticks(np.linspace(0, 1.5e-1, 10))
    # plt.show()
    plt.savefig('plots/tolerances.png', bbox_inches='tight')


def pos_reward():
    goal = np.array([0.05, 0.1])

    xmin = -0.15
    xmax = 0.15
    ymin = -0.05
    ymax = 0.15

    # Create a grid of points in the xy plane
    x_array = np.linspace(xmin, xmax, 200)
    y_array = np.linspace(ymin, ymax, 200)

    x_array = np.array(sorted(list(x_array) + [goal[0]]))
    y_array = np.array(sorted(list(y_array) + [goal[1]]))

    distances = np.zeros((y_array.shape[0], x_array.shape[0]))

    margin = 0.1 * 0.7

    for i, x in enumerate(x_array):
        for j, y in enumerate(y_array):
            dist = np.linalg.norm(np.array([x, y]) - goal)
            if y < 0:
                distances[j, i] = 0.
            else:
                distances[j, i] = tolerance(dist, margin=margin)

    # Define the colors for the colormap
    colors = [(1, 1, 1), (247/255, 164/255, 52/255)]  # Red to White

    # Create the colormap
    cmap_name = 'red_white'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors)

    # Create the plot
    fig, ax = plt.subplots()
    cax = ax.imshow(distances, cmap=cm, origin='lower', extent=[xmin, xmax, ymin, ymax])
    fig.colorbar(cax, ax=ax, label='Reward', fraction=0.03)

    # Add Cartesian axes
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)

    # Plot robot at rest
    robot_color = (72/255, 209/255, 204/255)
    ax.plot([0, 0], [0, 0.1], color=robot_color, linewidth=3, label='Robot at rest')
    ax.plot(0, 0.1, 'o', color='r', markersize=5)

    with open("./env/soft_reacher/chi.p", "rb") as inf:
        chi = pickle.load(inf)

    s_ps = np.linspace(0, 0.1, 50)
    xes = []
    yes = []
    q = np.array([-0.5, 0.1, 0.1]) * np.array([35, 0.5, 0.5])
    for s_p in s_ps:
        xy = chi([0, 1e-2, 0, *q], s_p)[:2]
        xes.append(xy[0])
        yes.append(xy[1])

    ax.plot(xes, yes, color=robot_color, linewidth=3, label='Robot in motion', alpha=0.5)
    ax.plot(xes[-1], yes[-1], 'o', color='r', markersize=5, alpha=0.7)

    ax.scatter(goal[0], goal[1], marker='o', color='limegreen', s=50, label='Goal', edgecolor='k', zorder=2)

    xes = []
    yes = []
    q = np.array([-0.21, 0.2, 0.26]) * np.array([35, 0.5, 0.5])
    for s_p in s_ps:
        xy = chi([0, 1e-2, 0, *q], s_p)[:2]
        xes.append(xy[0])
        yes.append(xy[1])

    ax.plot(xes, yes, color=robot_color, linewidth=3, alpha=0.5)
    ax.plot(xes[-1], yes[-1], 'o', color='r', markersize=5, alpha=0.5)

    # Draw base
    ax.axhspan(ymin, 0, color='grey', alpha=0.3, hatch='//')

    ax.legend(loc='upper left')

    # ax.set_title(f'Position reward with goal in ${*goal,}$ and margin=${np.round(margin, decimals=3)}$', pad=20)

    ax.set_xlabel('[m]')
    ax.set_ylabel('[m]')

    # plt.show()
    plt.savefig('plots/pos_reward.png', bbox_inches='tight')


if __name__ == '__main__':
    with open("utils/rcparams.json", "r") as inf:
        params = json.load(inf)
    plt.rcParams.update(params)
    # pos_reward()
    tolerances()
