import argparse
import os
import random
from copy import deepcopy

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from env.soft_reacher.soft_reacher import SoftReacher
from models.mbrl import ReplayBuffer, lnn, reward_model_FC, Pi_FC, V_FC
from rollout_plots import rollout_plots
from utils import seed_all


def hard_update(target, source):
    with torch.no_grad():
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)


def train_policy(resume=False, preprocess=False, seed=None):
    base_dir = f"log/policy/seed_{seed}"
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
    T = 16
    gamma = 0.99
    Lambda = 0.95

    replay_size = 200000

    clip_term = 100

    batch_size = 64

    # preprocess threshold
    pth = 0.5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    writer = SummaryWriter(log_dir=tensorboard_dir)

    a_zeros = None

    actor = Pi_FC(env.obs_size,
                  env.action_size).to(device)
    critic = V_FC(env.obs_size).to(device)
    critic_target = deepcopy(critic)

    actor_optimizer = torch.optim.AdamW(
        actor.parameters(), lr=lr)
    critic_optimizer = torch.optim.AdamW(
        critic.parameters(), lr=lr)

    a_scale = torch.tensor(env.a_scale, dtype=torch.float64, device=device)
    if not resume:
        replay_buffer = ReplayBuffer(replay_size, device)
        pbar = tqdm(range(replay_size))
        # Initialize replay buffer with K random episodes
        for episode in range(K):
            pbar.set_postfix_str(f"Episode {episode+1}/{K}")
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
            if pbar.n >= replay_size:
                break

        episode0 = 0
        print("Done initialization ...")
    else:
        ckpt = torch.load(os.path.join(f"log/model/seed_{seed}", "emergency.ckpt"))
        episode0 = ckpt['episode']

        replay_buffer = ckpt['replay_buffer']

        print("Loaded checkpoint")

    episodes = 500

    # Behaviour learning
    actor_loss_list, critic_loss_list = [], []
    actor_grad_list, critic_grad_list = [], []

    nan_count = 0

    for episode in range(episode0, episodes):
        for behaviour_batches in tqdm(range(behaviour_batches)):
            O = replay_buffer.sample_states(batch_size)
            t = 0
            values, values_target, values_lambda, R = [], [], [], []
            log_probs = []
            try:
                while True:
                    A, log_prob = actor(O, False, True)
                    log_probs.append(log_prob)
                    O_1 = self.transition_model(O, A * self.a_scale, train=False)
                    R.append(self.reward_model(O_1))
                    values.append(critic(O))
                    values_target.append(critic_target(O))
                    t += 1
                    O = O_1
                    if t % T == 0:
                        values_target.append(critic_target(O_1))
                        break

                # lambda-return calculation
                gae = torch.zeros_like(R[0])
                for t_ in reversed(range(T)):
                    delta = R[t_] + gamma * \
                            values_target[t_ + 1] - values_target[t_]
                    gae = delta + gamma * Lambda * gae
                    values_lambda.append(gae + values_target[t_])
                values_lambda = torch.stack(values_lambda)
                values_lambda = values_lambda.flip(0)

                values = torch.stack(values)
                critic_loss = 0.5 * \
                              torch.pow(values - values_lambda.detach(),
                                        2).sum(0).mean()

                log_probs = torch.stack(log_probs)
                actor_loss = - (values_lambda - 0.0001 *
                                log_probs).sum(0).mean()

                critic_optimizer.zero_grad()
                critic_loss.backward(
                    inputs=[param for param in critic.parameters()])
                torch.nn.utils.clip_grad_norm_(
                    critic.parameters(), clip_term)

                actor_optimizer.zero_grad()
                actor_loss.backward(
                    inputs=[param for param in actor.parameters()])
                torch.nn.utils.clip_grad_norm_(
                    actor.parameters(), clip_term)

                critic_grad = torch.norm(
                    torch.cat([param.grad.flatten() for param in critic.parameters()]))
                actor_grad = torch.norm(
                    torch.cat([param.grad.flatten() for param in actor.parameters()]))

                if torch.isnan(critic_grad).any().item() or torch.isnan(actor_grad).any().item():
                    nan_count += 1
                else:
                    critic_optimizer.step()
                    actor_optimizer.step()

                    critic_target_updates = (critic_target_updates + 1) % 100
                    if critic_target_updates == 0 or not first_update:
                        first_update = True
                        hard_update(critic_target, critic)

                    actor_loss_list.append(actor_loss.item())
                    critic_loss_list.append(critic_loss.item())
                    actor_grad_list.append(actor_grad.item())
                    critic_grad_list.append(critic_grad.item())

            except BaseException as e:
                nan_count += 1

        if nan_count > 0:
            print("episode", episode,
                  "got nan during behaviour learning", "nan count", nan_count)
        writer.add_scalar('critic_loss', np.mean(
            critic_loss_list), episode)
        writer.add_scalar('actor_loss', np.mean(actor_loss_list), episode)
        writer.add_scalar('critic_grad', np.mean(
            critic_grad_list), episode)
        writer.add_scalar('actor_grad', np.mean(actor_grad_list), episode)

        # Environment Interaction
        print("Starting Environment Interaction")
        o, _, _ = env.reset()
        o_tensor = torch.tensor(
            o, dtype=torch.float64, device=device)
        ep_r = 0
        while True:
            with torch.no_grad():
                try:
                    a_tensor, _ = actor(o_tensor[None])
                except:
                    print("episode", episode,
                          "got nan during environment interaction")
                    break
            o_1, r, done = env.step(a_tensor.cpu().numpy()[0])
            o_1_tensor = torch.tensor(
                o_1, dtype=torch.float64, device=device)
            r_tensor = torch.tensor(
                r, dtype=torch.float64, device=device)
            a_tensor = a_tensor.squeeze()

            n = int(env.dt / env.dt_small)
            assert n > 0
            for i in range(n):
                o_1_small = o_1_tensor[i, :].squeeze()
                r_small = r_tensor[i].squeeze()
                replay_buffer.push(o_tensor, a_tensor, r_small, o_1_small)

                o_tensor = o_1_small

            ep_r += r_small.cpu().item()
            if done:
                writer.add_scalar('ep_r', ep_r, episode)
                if episode % 25 == 0 or episode == episodes - 1:
                    try:
                        # Evaluate agent performance
                        eval_ep_r_list = evaluate(env, actor, 50, False)
                        writer.add_scalar('eval_ep_r', np.mean(
                            eval_ep_r_list), episode)
                    except:
                        print("episode", episode, "got nan during eval")
                    checkpoint = {'episode': episode,
                                  'actor': actor.state_dict(),
                                  'actor_optimizer': actor_optimizer.state_dict(),
                                  'critic': critic.state_dict(),
                                  'critic_optimizer': critic_optimizer.state_dict(),
                                  'critic_target': critic_target.state_dict(),
                                  'replay_buffer': replay_buffer
                                  }
                    torch.save(checkpoint, os.path.join(base_dir, "emergency.ckpt"))

                break
        print("Done Environment Interaction")


def evaluate(env, actor, episodes, render=False):
    device = next(actor.parameters()).device
    # Evaluate agent performance over several episodes
    ep_r_list = []
    for episode in tqdm(range(episodes)):
        o, _, _ = env.reset()
        ep_r = 0
        while True:
            with torch.no_grad():
                a, _ = actor(torch.tensor(
                    o, dtype=torch.float64, device=device).unsqueeze(0), True)
            a = a.cpu().numpy()[0]
            o_1, r, done = env.step(a)
            if render:
                env.render()
            ep_r += r[-1]
            o = o_1[-1, :]
            if done:
                ep_r_list.append(ep_r)
                if render:
                    print("Episode finished with total reward ", ep_r)
                break

    return ep_r_list