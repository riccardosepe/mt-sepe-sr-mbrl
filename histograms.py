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
