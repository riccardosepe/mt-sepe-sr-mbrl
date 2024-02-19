import argparse
import os
from copy import deepcopy
from functools import partial

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from env.soft_reacher.soft_reacher import SoftReacher
from env.utils import make_env
from env.vec_env import VecEnv
from models.mbrl import ReplayBuffer, Pi_FC, V_FC
from utils import seed_all


def hard_update(target, source):
    with torch.no_grad():
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)


class FunctionEnvironment(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_act, vec_env):
        act = input_act.detach().cpu().numpy()
        obss, rews, _ = vec_env.step(act)

        w, _, _, _ = np.linalg.lstsq(act, obss, rcond=None)
        ctx.save_for_backward(torch.tensor(w))

        return (torch.tensor(obss, dtype=torch.float64, device=input_act.device),
                torch.tensor(rews, dtype=torch.float64, device=input_act.device))

    @staticmethod
    def backward(ctx, grad_output_obs, grad_output_rew):
        w, = ctx.saved_tensors
        return grad_output_obs @ w.T, None


def train_policy(resume=False, preprocess=False, seed=None, num_parallel_envs=2):
    base_dir = f"log/policy/seed_{seed}"
    if os.path.isdir(base_dir) and not resume:
        raise FileExistsError(f"Folder {base_dir} already exists.")

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

    behaviour_batches = 1000

    replay_size = 100000
    clip_term = 100
    batch_size = num_parallel_envs

    critic_target_updates = 0
    first_update = False

    # preprocess threshold
    pth = 0.5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    writer = SummaryWriter(log_dir=tensorboard_dir)

    # Model substitute

    actor = Pi_FC(env.obs_size,
                  env.action_size).to(device)
    critic = V_FC(env.obs_size).to(device)
    critic_target = deepcopy(critic)

    actor_optimizer = torch.optim.AdamW(
        actor.parameters(), lr=lr)
    critic_optimizer = torch.optim.AdamW(
        critic.parameters(), lr=lr)

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
        ckpt = torch.load(os.path.join(base_dir, "emergency.ckpt"))
        episode0 = ckpt['episode']
        actor.load_state_dict(ckpt['actor'])
        critic.load_state_dict(ckpt['critic'])
        actor_optimizer.load_state_dict(ckpt['actor_optimizer'])
        critic_optimizer.load_state_dict(ckpt['critic_optimizer'])
        critic_target.load_state_dict(ckpt['critic_target'])

        replay_buffer = ckpt['replay_buffer']

        print("Loaded checkpoint")

    episodes = 500

    # Behaviour learning
    actor_loss_list, critic_loss_list = [], []
    actor_grad_list, critic_grad_list = [], []

    nan_count = 0

    vec_env = VecEnv(partial(make_env, name="soft_reacher", mle=False), batch_size)

    for episode in range(episode0, episodes):
        for behaviour_batch in tqdm(range(behaviour_batches)):
            O = replay_buffer.sample_states(batch_size)
            vec_env.set_state(O)
            t = 0
            values, values_target, values_lambda, R = [], [], [], []
            log_probs = []
            try:
                while True:
                    A, log_prob = actor(O, False, True)
                    log_probs.append(log_prob)

                    O_1, Rs = FunctionEnvironment.apply(A, vec_env)

                    R.append(Rs)
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
        rews = [0.]
        cumrews = [0.]
        while True:
            with torch.no_grad():
                a, _ = actor(torch.tensor(
                    o, dtype=torch.float64, device=device).unsqueeze(0), True)
            a = a.cpu().numpy()[0]
            o_1, r, done = env.step(a)
            if render:
                env.render()
            ep_r += r[-1]
            rews.append(r[-1])
            cumrews.append(ep_r)
            o = o_1[-1, :]
            if done:
                ep_r_list.append(ep_r)
                if render:
                    print("Episode finished with total reward ", ep_r)
                # plt.plot(rews, label="Reward")
                # plt.plot(cumrews, label="Cumulative reward")
                # plt.show()
                break

    return ep_r_list


def _test_code():
    # Create a partial function that will create a "pendulum" environment
    make_pendulum_env = partial(make_env, name="soft_reacher", mle=False)

    BS = 64

    # Create a vectorized environment with BS instances of the "pendulum" environment
    vec_env = VecEnv(make_pendulum_env, num_envs=BS)
    observations, _, _ = vec_env.reset()

    with torch.autograd.detect_anomaly():
        act = torch.ones((BS, 3), requires_grad=True)
        obs, rew = FunctionEnvironment.apply(act, vec_env)
        print(obs, rew)
        # rew.sum().backward()
        obs.mean().backward()

    vec_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--preprocess", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_parallel_envs", type=int, default=2)
    args = parser.parse_args()

    print(f"Running with seed {args.seed} and {args.num_parallel_envs} parallel environments.")

    train_policy(args.resume, args.preprocess, args.seed, args.num_parallel_envs)
    # _test_code()

