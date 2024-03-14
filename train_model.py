import argparse
import os

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from env.soft_reacher.soft_reacher import SoftReacher
from models.mbrl import ReplayBuffer, LNN, RewardMLP
from utils.utils import seed_all
from visualization.rollout_plots import rollout_plots


def train_model(resume=False, seed=None):
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
    lr = 3e-4
    clip_term = 100

    batch_size = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    writer = SummaryWriter(log_dir=tensorboard_dir)

    a_zeros = None
    transition_model = LNN(
        env.name,
        env.n,
        env.obs_size,
        env.action_size,
        env.dt,
        env.dt_small,
        a_zeros).to(device)

    transition_loss_fn = torch.nn.L1Loss()

    reward_model = RewardMLP(env.obs_size).to(device)
    reward_loss_fn = torch.nn.L1Loss()

    transition_optimizer = torch.optim.AdamW(
        transition_model.parameters(), lr=lr)
    reward_optimizer = torch.optim.AdamW(
        reward_model.parameters(), lr=lr)

    a_scale = torch.tensor(env.a_scale, dtype=torch.float64, device=device)

    data = torch.load("data/seed_27/data.pt")
    replay_buffer = data["large"]

    if not resume:
        epoch0 = 0
    else:
        ckpt = torch.load(os.path.join(f"log/model/seed_{seed}", "emergency.ckpt"))
        epoch0 = ckpt['epoch']
        transition_model.load_state_dict(ckpt['transition_model'])
        reward_model.load_state_dict(ckpt['reward_model'])

        transition_optimizer.load_state_dict(ckpt['transition_optimizer'])
        reward_optimizer.load_state_dict(ckpt['reward_optimizer'])

        print("Loaded checkpoint")

    num_epochs = 500

    for epoch in range(epoch0, num_epochs):
        # Model learning
        transition_loss_list, reward_loss_list = [], []
        transition_grad_list, reward_grad_list = [], []
        pbar = tqdm(range(10000))
        for model_batches in pbar:
            pbar.set_postfix_str(f"Epoch {epoch+1}/{num_epochs}")
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

        if epoch % 25 == 0 or epoch == num_epochs - 1:
            checkpoint = {'epoch': epoch,
                          'transition_model': transition_model.state_dict(),
                          'transition_optimizer': transition_optimizer.state_dict(),
                          'reward_model': reward_model.state_dict(),
                          'reward_optimizer': reward_optimizer.state_dict(),
                          }
            torch.save(checkpoint, os.path.join(base_dir, "emergency.ckpt"))
            rollout_plots(env,
                          transition_model,
                          epoch,
                          render=False,
                          save=True,
                          save_path=plots_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train the model")
    parser.add_argument("--resume", action="store_true", default=False, help="Resume training")
    parser.add_argument("--seed", type=int, default=None, help="seed")
    parser.add_argument("--preprocess", action="store_true", default=False, help="Preprocess the data")
    args = parser.parse_args()
    train_model(seed=args.seed, resume=args.resume)
