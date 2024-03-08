import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


def main():
    x1 = np.linspace(-3, 3, 400)
    x2 = np.linspace(-3.5, 5.5, 400)
    y2 = np.cos(x2) - np.sin(2 * x2)
    y2 = np.where(((y2 > 1) & (x2 < 2)), 1, y2)

    # y1 = np.cosh(x1/3)
    y1 = -norm(0, 1).pdf(x1)

    # Create masks for the regions where you want dashed and solid lines
    x1_mask_dashed_1 = (x1 < -2.25)
    x1_mask_dashed_2 = (x1 > 2.25)
    x1_mask_solid = ~(x1_mask_dashed_1 | x1_mask_dashed_2)

    x2_mask_dashed_1 = (x2 < -3)
    x2_mask_dashed_2 = (x2 > 5)
    x2_mask_solid = ~(x2_mask_dashed_1 | x2_mask_dashed_2)

    fig, axes = plt.subplots(1, 2, figsize=(28, 12))

    # Plot dashed lines for the tails
    axes[0].plot(x1[x1_mask_dashed_1], y1[x1_mask_dashed_1], linestyle='dashed', c='#1f77b4', linewidth=5)
    axes[0].plot(x1[x1_mask_dashed_2], y1[x1_mask_dashed_2], linestyle='dashed', c='#1f77b4', linewidth=5)

    axes[1].plot(x2[x2_mask_dashed_1], y2[x2_mask_dashed_1], linestyle='dashed', c='#1f77b4', linewidth=5)
    axes[1].plot(x2[x2_mask_dashed_2], y2[x2_mask_dashed_2], linestyle='dashed', c='#1f77b4', linewidth=5)

    # Plot solid line for the middle part
    axes[0].plot(x1[x1_mask_solid], y1[x1_mask_solid], linestyle='solid', c='#1f77b4', linewidth=5)
    axes[1].plot(x2[x2_mask_solid], y2[x2_mask_solid], linestyle='solid', c='#1f77b4', linewidth=5)

    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    for a in axes:
        a.spines['top'].set_linewidth(5)
        a.spines['right'].set_linewidth(5)
        a.spines['bottom'].set_linewidth(5)
        a.spines['left'].set_linewidth(5)

    axes[0].set_title('Convex landscape (a)', fontsize=60)
    axes[1].set_title('Non-convex landscape (b)', fontsize=60)

    plt.tight_layout()

    plt.subplots_adjust(wspace=0.05)

    plt.savefig('../plots/convex_non_convex_landscape.png', bbox_inches='tight', pad_inches=0.1)


if __name__ == "__main__":
    main()
