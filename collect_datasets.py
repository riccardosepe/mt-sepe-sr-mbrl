import os

import numpy as np
import torch
from tqdm import tqdm

from env.soft_reacher.soft_reacher import SoftReacher
from models.mbrl import ReplayBuffer
from utils.utils import seed_all


def main(seed):
    base_dir = f"data/seed_{seed}"
    os.makedirs(base_dir, exist_ok=True)

    # Set the seed
    seed_all(seed)

    # Initialize the environment
    env = SoftReacher(mle=False)

    # NB: n is most likely between 50 and 100
    n = int(env.dt / env.dt_small)

    K = 10
    large_data_size = 200000
    small_data_size = int(large_data_size / n)
    action_repeat = 1
    ep_length = int(large_data_size / (K * n * action_repeat))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_large = ReplayBuffer(large_data_size, device)
    data_small = ReplayBuffer(large_data_size, device)
    pbar = tqdm(range(large_data_size))
    # Initialize replay buffer with K random episodes
    for episode in range(K):
        pbar.set_postfix_str(f"Episode {episode+1}/{K}")
        o, _, _ = env.reset()
        o_tensor = torch.tensor(
            o, dtype=torch.float64, device=device)
        for _ in range(ep_length):
            a = np.random.uniform(-1.0, 1.0, size=env.action_size)
            for _ in range(action_repeat):
                o_1, r, done = env.step(a)
                a_tensor = torch.tensor(
                    a, dtype=torch.float64, device=device)
                o_1_tensor = torch.tensor(
                    o_1, dtype=torch.float64, device=device)
                r_tensor = torch.tensor(
                    r, dtype=torch.float64, device=device)
                data_small.push(o_tensor, a_tensor, r_tensor[-1], o_1_tensor[-1, :].squeeze())
                for i in range(n):
                    o_1_small = o_1_tensor[i, :].squeeze()
                    r_small = r_tensor[i].squeeze()
                    data_large.push(o_tensor, a_tensor, r_small, o_1_small)
                    pbar.update(1)

                    o_tensor = o_1_small

    data = {"large": data_large, "small": data_small}
    torch.save(data, os.path.join(base_dir, "data.pt"))
    print("Done initialization ...")


if __name__ == '__main__':
    # main(27)
    from visualization.histograms import *
    data = torch.load("data/seed_27/data.pt")
    data_large = data["large"]
    data_small = data["small"]
    states_histograms(data_large)
    states_histograms(data_small)
    actions_histograms(data_large)
    actions_histograms(data_small)
    rewards_histograms(data_large)
    rewards_histograms(data_small)
    joint_positions_velocities_histograms(data_large)
    joint_positions_velocities_histograms(data_small)
