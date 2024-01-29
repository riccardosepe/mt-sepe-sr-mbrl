import matplotlib.pyplot as plt
import numpy as np
from env.rewards import tolerance
from mpl_toolkits.mplot3d import Axes3D


def main():
    x = np.linspace(0, 1.5e-1, 100)
    y = np.linspace(0, 2e-1, 100)

    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = tolerance(X[i, j], margin=Y[i, j])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, alpha=0.7)

    # Add a line at x = 1e-1
    x_line = 1e-1
    y_line = np.linspace(0, 2e-1, 100)
    z_line = np.array([tolerance(x_line, margin=yi) for yi in y_line])
    ax.plot([x_line] * len(y_line), y_line, z_line, color='r', linewidth=4)

    ax.set_xlabel('x')
    ax.set_ylabel('margin')
    ax.set_zlabel('tolerance')

    fig2, ax2 = plt.subplots()
    # Add a line at y = 1e-2
    y_line = 3.5e-2
    x_line = np.linspace(0, 1.5e-1, 100)
    z_line = np.array([tolerance(xi, margin=y_line) for xi in x_line])
    ax2.plot(x_line, z_line, color='k', linewidth=4)

    # ax2.set_xticks(np.linspace(0, 1.5e-1, 10))
    plt.show()


if __name__ == '__main__':
    main()
