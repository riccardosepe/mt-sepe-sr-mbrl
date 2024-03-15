import argparse
import os
from functools import partial

import numpy as np
import optuna
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from env.soft_reacher.soft_reacher import SoftReacher
from models.mbrl import ReplayBuffer, LNN, RewardMLP, MLP
from utils.utils import seed_all
from visualization.rollout_plots import rollout_plots


def train_model(trial, seed=None):
    base_dir = f"log/model_mlp/seed_{seed}"
    if os.path.isdir(base_dir):
        raise FileExistsError(f"Folder {base_dir} already exists.")

    # Set the seed
    seed_all(seed)

    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)

    # Initialize the environment
    env = SoftReacher(mle=False)

    tensorboard_dir = os.path.join(base_dir, "tensorboard")
    clip_term = 100

    batch_size = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    writer = SummaryWriter(log_dir=tensorboard_dir)

    transition_model = MLP(env.obs_size, env.action_size).to(device)

    transition_loss_fn = torch.nn.L1Loss()

    reward_model = RewardMLP(env.obs_size).to(device)
    reward_loss_fn = torch.nn.L1Loss()

    transition_optimizer = torch.optim.AdamW(
        transition_model.parameters(), lr=lr)
    reward_optimizer = torch.optim.AdamW(
        reward_model.parameters(), lr=lr)

    a_scale = torch.tensor(env.a_scale, dtype=torch.float64, device=device)

    data = torch.load("data/seed_27/data.pt")
    replay_buffer = data["small"]
    replay_buffer.device = device

    valid_data = torch.load("data/seed_27/data.pt")
    valid_replay_buffer = valid_data["small"]

    epoch0 = 0

    num_epochs = 500
    val_losses = []
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
        writer.add_scalar('lr', lr, epoch)

        if epoch % 25 == 0 or epoch == num_epochs - 1:
            # Validate
            val_loss = 0
            for _ in range(50):
                O, A, R, O_1 = valid_replay_buffer.sample_transitions(batch_size)
                O_1_pred = transition_model(O, A * a_scale, train=False)
                val_loss += transition_loss_fn(O_1_pred, O_1).item()
            val_loss /= 50
            writer.add_scalar('validation_loss', val_loss, epoch)
            val_losses.append(val_loss)
            trial.report(np.mean(val_losses), epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

            checkpoint = {'epoch': epoch,
                          'transition_model': transition_model.state_dict(),
                          'transition_optimizer': transition_optimizer.state_dict(),
                          'reward_model': reward_model.state_dict(),
                          'reward_optimizer': reward_optimizer.state_dict(),
                          }
            torch.save(checkpoint, os.path.join(base_dir, "emergency.ckpt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train the model")
    parser.add_argument("--seed", type=int, default=None, help="seed")

    args = parser.parse_args()

    study = optuna.create_study(study_name='model_mlp',
                                direction='minimize',
                                sampler=optuna.samplers.TPESampler(),
                                pruner=optuna.pruners.HyperbandPruner()
                                )
    study.optimize(partial(train_model, seed=args.seed), n_trials=10)

