import numpy as np


PLT_LABELS = ['bend', 'shear', 'axial', 'bend_vel', 'shear_vel', 'axial_vel']


def states_histograms(buffer):
    import matplotlib.pyplot as plt
    states = np.array([np.array(s[0]) for s in buffer.buffer])
    if states.shape[1] != 6:
        raise ValueError(f"States have shape {states.shape} instead of (N, 6)")
    fig, axs = plt.subplots(3, 2, figsize=(15, 10))
    for i in range(6):
        axs[i % 3, i // 3].hist(states[:, i], bins=20)
        axs[i % 3, i // 3].set_title(PLT_LABELS[i])

    plt.tight_layout()
    fig.suptitle("States histograms")
    plt.show()


def joint_positions_velocities_histograms(buffer):
    states = np.array([np.array(s[0]) for s in buffer.buffer])
    if states.shape[1] != 6:
        raise ValueError(f"States have shape {states.shape} instead of (N, 6)")
    fig, axs = plt.subplots(3, 1, figsize=(10, 30))
    for i in range(3):
        position = states[:, i]
        velocity = states[:, i + 3]
        # Create a 2D histogram
        hist, xedges, yedges = np.histogram2d(position, velocity, bins=10)

        # Plot the 2D histogram as an image on the axis
        cax = axs[i].imshow(hist, interpolation='nearest', origin='lower',
                            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                            aspect='auto'
                            )

        # Add colorbar and labels
        cbar = fig.colorbar(cax, ax=axs[i], fraction=0.05)
        cbar.set_label('Frequency', fontsize=14)
        axs[i].set_xlabel('q', fontsize=14)
        axs[i].set_ylabel('qdot', fontsize=14)

        axs[i].set_title(PLT_LABELS[i], fontsize=16)
        axs[i].set_aspect(abs(xedges[-1] - xedges[0]) / abs(yedges[-1] - yedges[0]))
        axs[i].tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()
    fig.suptitle("States joint histograms", fontsize=20)
    plt.show()


def actions_histograms(buffer):
    import matplotlib.pyplot as plt
    actions = np.array([np.array(sample[1]) for sample in buffer.buffer])
    # if actions.shape[1] != 3:
    #     raise ValueError(f"Actions have shape {actions.shape} instead of (N, 3)")
    act_dim = actions.shape[1]
    fig, axs = plt.subplots(act_dim, 1, figsize=(8, 10))
    for i in range(act_dim):
        axs[i].hist(actions[:, i], bins=20)
        axs[i].set_title(PLT_LABELS[i])

    plt.tight_layout()
    fig.subplots_adjust(top=0.92)
    fig.suptitle("Actions histograms")
    plt.show()


def rewards_histograms(buffer):
    import matplotlib.pyplot as plt
    rewards = np.array([np.array(s[2]) for s in buffer.buffer])
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.hist(rewards, bins=10)
    ax.set_title("Rewards")

    plt.tight_layout()
    fig.suptitle("Rewards histograms")
    fig.subplots_adjust(top=0.88)
    plt.show()
