import argparse
import os
import random

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from env.soft_reacher.soft_reacher import SoftReacher
from models.mbrl import ReplayBuffer, lnn, reward_model_FC
from rollout_plots import rollout_plots


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def train_model(resume=False, preprocess=False, seed=None):
    base_dir = f"log/model/seed_{seed}"
    if os.path.isdir(base_dir) and not resume:
        raise FileExistsError(f"Folder {base_dir} already exists.")
    plots_dir = os.path.join(base_dir, "plots")
    if not os.path.isdir(plots_dir):
        os.makedirs(plots_dir, exist_ok=True)

    # Set the seed
    seed_all(seed)

    # Initialize the environment
    env = SoftReacher(mle=False)

    tensorboard_dir = os.path.join(base_dir, "tensorboard")
    K = 10
    lr = 3e-4

    replay_size = 100000

    clip_term = 100

    batch_size = 64

    # preprocess threshold
    pth = 0.5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    writer = SummaryWriter(log_dir=tensorboard_dir)

    a_zeros = None
    transition_model = lnn(
        env.name,
        env.n,
        env.obs_size,
        env.action_size,
        env.dt,
        env.dt_small,
        a_zeros).to(device)

    transition_loss_fn = torch.nn.L1Loss()

    reward_model = reward_model_FC(env.obs_size).to(device)
    reward_loss_fn = torch.nn.L1Loss()

    transition_optimizer = torch.optim.AdamW(
        transition_model.parameters(), lr=lr)
    reward_optimizer = torch.optim.AdamW(
        reward_model.parameters(), lr=lr)

    a_scale = torch.tensor(env.a_scale, dtype=torch.float64, device=device)
    if not resume:
        replay_buffer = ReplayBuffer(replay_size, device)
        pbar = tqdm(range(replay_size))
        # Initialize replay buffer with K random episodes
        for episode in tqdm(range(K)):
            o, _, _ = env.reset()
            o_tensor = torch.tensor(
                o, dtype=torch.float64, device=device)
            while True:
                a = np.random.uniform(-1.0, 1.0, size=env.action_size)
                o_1, r, done = env.step(a)
                a_tensor = torch.tensor(
                    a, dtype=torch.float64, device=device)
                o_1_tensor = torch.tensor(
                    o_1, dtype=torch.float64, device=device)
                r_tensor = torch.tensor(
                    r, dtype=torch.float64, device=device)
                # Here there is the first big change:
                #  - o_tensor is a tensor of shape (D,)
                #  - a_tensor is a tensor of shape (A,)
                #  - r_tensor is a tensor of shape (N, 1)
                #  - o_1 is a tensor of shape (N, D), where:
                #  - B is batch size, for example 64
                #  - D is the observation size, in this case 6
                #  - A is the action size, in this case 3
                # - N is the number of intermediate observations returned by the environment, i.e. dt_large/dt_small
                # NB: N is most likely between 50 and 100
                n = int(env.dt / env.dt_small)
                for i in range(n):
                    o_1_small = o_1_tensor[i, :].squeeze()
                    r_small = r_tensor[i].squeeze()
                    if preprocess:
                        # remove small values for bending
                        if not (-pth < o_tensor[0] < pth) and not (-pth < o_1_small[0] < pth):
                            replay_buffer.push(o_tensor, a_tensor, r_small, o_1_small)
                            pbar.update(1)
                    else:
                        replay_buffer.push(o_tensor, a_tensor, r_small, o_1_small)
                        pbar.update(1)

                    o_tensor = o_1_small

                if done:
                    break

        epoch0 = 0
        print("Done initialization ...")
    else:
        ckpt = torch.load(os.path.join(f"log/model/seed_{seed}", "emergency.ckpt"))
        epoch0 = ckpt['epoch']
        transition_model.load_state_dict(ckpt['transition_model'])
        reward_model.load_state_dict(ckpt['reward_model'])

        transition_optimizer.load_state_dict(ckpt['transition_optimizer'])
        reward_optimizer.load_state_dict(ckpt['reward_optimizer'])

        replay_buffer = ckpt['replay_buffer']

        print("Loaded checkpoint")

    num_epochs = 500

    for epoch in range(epoch0, num_epochs):
        # Model learning
        transition_loss_list, reward_loss_list = [], []
        transition_grad_list, reward_grad_list = [], []
        for model_batches in tqdm(range(10000)):
            O, A, R, O_1 = replay_buffer.sample_transitions(batch_size)

            # Dynamics learning
            O_1_pred = transition_model(O, A * a_scale, train=True)
            transition_loss = transition_loss_fn(O_1_pred, O_1)
            # if torch.isnan(transition_loss).any().item():
            #     print("NAN LOSS", file=sys.stderr)
            #     exit(1)
            transition_optimizer.zero_grad()
            transition_loss.backward()
            torch.nn.utils.clip_grad_norm_(transition_model.parameters(), clip_term)
            transition_optimizer.step()
            transition_loss_list.append(transition_loss.item())
            transition_grad = []
            for param in transition_model.parameters():
                if param.grad is not None:
                    transition_grad.append(param.grad.flatten())
            transition_grad_list.append(torch.norm(
                torch.cat(transition_grad)).item())

            # Reward learning
            R_pred = reward_model(O_1)
            reward_loss = reward_loss_fn(R_pred, R)
            reward_optimizer.zero_grad()
            reward_loss.backward()
            torch.nn.utils.clip_grad_norm_(reward_model.parameters(), clip_term)
            reward_optimizer.step()
            reward_loss_list.append(reward_loss.item())
            reward_grad_list.append(torch.norm(torch.cat(
                [param.grad.flatten() for param in reward_model.parameters()])).item())

        transition_loss_mean = np.mean(transition_loss_list)
        reward_loss_mean = np.mean(reward_loss_list)
        transition_grad_mean = np.mean(transition_grad_list)
        reward_grad_mean = np.mean(reward_grad_list)

        writer.add_scalar('transition_loss', transition_loss_mean, epoch)
        writer.add_scalar('reward_loss', reward_loss_mean, epoch)
        writer.add_scalar('transition_grad', transition_grad_mean, epoch)
        writer.add_scalar('reward_grad', reward_grad_mean, epoch)

        if epoch % 25 == 0 or epoch == 0:
            checkpoint = {'epoch': epoch,
                          'transition_model': transition_model.state_dict(),
                          'transition_optimizer': transition_optimizer.state_dict(),
                          'reward_model': reward_model.state_dict(),
                          'reward_optimizer': reward_optimizer.state_dict(),
                          'replay_buffer': replay_buffer
                          }
            torch.save(checkpoint, os.path.join(base_dir, "emergency.ckpt"))
            rollout_plots(env,
                          transition_model,
                          render=False,
                          save=True,
                          save_path=os.path.join(plots_dir, f"rollout_{epoch}.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train the model")
    parser.add_argument("--resume", action="store_true", default=False, help="Resume training")
    parser.add_argument("--seed", type=int, default=None, help="seed")
    parser.add_argument("--preprocess", action="store_true", default=False, help="Preprocess the data")
    args = parser.parse_args()
    train_model(seed=args.seed, resume=args.resume)
