from tqdm import tqdm

from env.model_env import ModelEnv
from models.sac_with_model import ReplayBuffer, Q_FC, Pi_FC
from env.utils import make_env
from torch.utils.tensorboard import SummaryWriter
import os
import argparse
from copy import deepcopy
import numpy as np
import torch

from utils.utils import seed_all, soft_update

torch.set_default_dtype(torch.float64)


class SAC:
    """
    Soft Actor-Critic agent with model for data augmentation
    """

    def __init__(self, arglist):
        self.arglist = arglist
        seed_all(self.arglist.seed)

        self.device = torch.device(
            "cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        self.env = make_env(self.arglist.env)
        self.model = None
        self.obs_size = self.env.obs_size
        self.action_size = self.env.action_size

        self.actor = Pi_FC(self.obs_size, self.action_size).to(self.device)

        if self.arglist.mode == "train":
            if self.arglist.model_path is None:
                raise ValueError("Model path is needed for training")
            self.model = ModelEnv(self.arglist.model_path, self.env)

            self.critic_1 = Q_FC(
                self.obs_size, self.action_size).to(self.device)
            self.critic_target_1 = deepcopy(self.critic_1)
            self.critic_loss_fn_1 = torch.nn.MSELoss()

            self.critic_2 = Q_FC(
                self.obs_size, self.action_size).to(self.device)
            self.critic_target_2 = deepcopy(self.critic_2)
            self.critic_loss_fn_2 = torch.nn.MSELoss()

            # set target entropy to -|A|
            self.target_entropy = - self.action_size

            path = "./log/"+self.env.name+"/sac_on_model"
            self.exp_dir = os.path.join(path, "seed_"+str(self.arglist.seed))
            self.model_dir = os.path.join(self.exp_dir, "models")
            self.tensorboard_dir = os.path.join(self.exp_dir, "tensorboard")

            if self.arglist.resume:
                checkpoint = torch.load(os.path.join(
                    self.model_dir, "backup.ckpt"))
                self.start_episode = checkpoint['episode'] + 1

                self.actor.load_state_dict(checkpoint['actor'])
                self.critic_1.load_state_dict(checkpoint['critic_1'])
                self.critic_target_1.load_state_dict(
                    checkpoint['critic_target_1'])
                self.critic_2.load_state_dict(checkpoint['critic_2'])
                self.critic_target_2.load_state_dict(
                    checkpoint['critic_target_2'])
                self.log_alpha = torch.tensor(checkpoint['log_alpha'].item(
                ), dtype=torch.float64, device=self.device, requires_grad=True)

                self.replay_buffer = checkpoint['replay_buffer']

            else:
                self.start_episode = 0

                self.log_alpha = torch.tensor(
                    np.log(0.2), dtype=torch.float64, device=self.device, requires_grad=True)

                self.replay_buffer = ReplayBuffer(
                    self.arglist.replay_size, self.device)

                if not os.path.exists(path):
                    os.makedirs(path)
                os.mkdir(self.exp_dir)
                os.mkdir(self.tensorboard_dir)
                os.mkdir(self.model_dir)

            for param in self.critic_target_1.parameters():
                param.requires_grad = False

            for param in self.critic_target_2.parameters():
                param.requires_grad = False

            self.actor_optimizer = torch.optim.Adam(
                self.actor.parameters(), lr=self.arglist.lr)
            self.critic_optimizer_1 = torch.optim.Adam(
                self.critic_1.parameters(), lr=self.arglist.lr)
            self.critic_optimizer_2 = torch.optim.Adam(
                self.critic_2.parameters(), lr=self.arglist.lr)
            self.log_alpha_optimizer = torch.optim.Adam(
                [self.log_alpha], lr=self.arglist.lr)

            if self.arglist.resume:
                self.actor_optimizer.load_state_dict(
                    checkpoint['actor_optimizer'])
                self.critic_optimizer_1.load_state_dict(
                    checkpoint['critic_optimizer_1'])
                self.critic_optimizer_2.load_state_dict(
                    checkpoint['critic_optimizer_2'])
                self.log_alpha_optimizer.load_state_dict(
                    checkpoint['log_alpha_optimizer'])

                print("Done loading checkpoint ...")

            self.train()

        elif self.arglist.mode == "eval":
            checkpoint = torch.load(
                self.arglist.checkpoint, map_location=self.device)
            self.actor.load_state_dict(checkpoint['actor'])
            ep_r_list = self.eval(self.arglist.episodes, self.arglist.render)

    def save_checkpoint(self, name):
        checkpoint = {'actor': self.actor.state_dict()}
        torch.save(checkpoint, os.path.join(self.model_dir, name))

    def save_backup(self, episode):
        checkpoint = {'episode': episode,
                      'actor': self.actor.state_dict(),
                      'actor_optimizer': self.actor_optimizer.state_dict(),
                      'critic_1': self.critic_1.state_dict(),
                      'critic_optimizer_1': self.critic_optimizer_1.state_dict(),
                      'critic_2': self.critic_2.state_dict(),
                      'critic_optimizer_2': self.critic_optimizer_2.state_dict(),
                      'critic_target_1': self.critic_target_1.state_dict(),
                      'critic_target_2': self.critic_target_2.state_dict(),
                      'log_alpha': self.log_alpha.detach(),
                      'log_alpha_optimizer': self.log_alpha_optimizer.state_dict(),
                      'replay_buffer': self.replay_buffer
                      }
        torch.save(checkpoint, os.path.join(self.model_dir, "backup.ckpt"))

    def train(self):
        writer = SummaryWriter(log_dir=self.tensorboard_dir)

        for episode in tqdm(range(self.start_episode, self.arglist.episodes)):
            o_tensor, _, _ = self.model.reset()
            bs = o_tensor.shape[0]
            ep_r = torch.zeros((bs, ), device=self.device, dtype=torch.float64)
            while True:
                if len(self.replay_buffer) >= self.arglist.start_steps:
                    with torch.no_grad():
                        a_tensor, _ = self.actor(o_tensor)
                else:
                    a_tensor = torch.rand((bs, self.action_size), device=self.device) * 2 - 1

                with torch.no_grad():
                    o_1_tensor, r_tensor, done = self.model.step(a_tensor)
                    self.replay_buffer.push(o_tensor, a_tensor, r_tensor, o_1_tensor)
                    ep_r += r_tensor

                o_tensor = o_1_tensor

                if len(self.replay_buffer) >= self.arglist.replay_fill:
                    O, A, R, O_1 = self.replay_buffer.sample(
                        self.arglist.batch_size)

                    q_value_1 = self.critic_1(O, A)
                    q_value_2 = self.critic_2(O, A)

                    with torch.no_grad():
                        # Target actions come from *current* policy
                        A_1, logp_A_1 = self.actor(O_1, False, True)

                        next_q_value_1 = self.critic_target_1(O_1, A_1)
                        next_q_value_2 = self.critic_target_2(O_1, A_1)
                        next_q_value = torch.min(
                            next_q_value_1, next_q_value_2)
                        expected_q_value = R + self.arglist.gamma * \
                            (next_q_value - torch.exp(self.log_alpha) * logp_A_1)

                    critic_loss_1 = self.critic_loss_fn_1(
                        q_value_1, expected_q_value)
                    self.critic_optimizer_1.zero_grad()
                    critic_loss_1.backward()
                    self.critic_optimizer_1.step()

                    critic_loss_2 = self.critic_loss_fn_2(
                        q_value_2, expected_q_value)
                    self.critic_optimizer_2.zero_grad()
                    critic_loss_2.backward()
                    self.critic_optimizer_2.step()

                    for param_1, param_2 in zip(self.critic_1.parameters(), self.critic_2.parameters()):
                        param_1.requires_grad = False
                        param_2.requires_grad = False

                    A_pi, logp_A_pi = self.actor(O, False, True)
                    q_value_pi_1 = self.critic_1(O, A_pi)
                    q_value_pi_2 = self.critic_2(O, A_pi)
                    q_value_pi = torch.min(q_value_pi_1, q_value_pi_2)

                    actor_loss = - \
                        torch.mean(
                            q_value_pi - torch.exp(self.log_alpha).detach() * logp_A_pi)
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()

                    self.log_alpha_optimizer.zero_grad()
                    alpha_loss = (torch.exp(self.log_alpha) *
                                  (-logp_A_pi - self.target_entropy).detach()).mean()
                    alpha_loss.backward()
                    self.log_alpha_optimizer.step()

                    for param_1, param_2 in zip(self.critic_1.parameters(), self.critic_2.parameters()):
                        param_1.requires_grad = True
                        param_2.requires_grad = True

                    soft_update(self.critic_target_1, self.critic_1, self.arglist.tau)
                    soft_update(self.critic_target_2, self.critic_2, self.arglist.tau)

                if done:
                    returns = ep_r[~torch.isnan(ep_r)]
                    # Compute some statistics
                    mean_return = torch.mean(returns)
                    std_return = torch.std(returns)
                    min_return = torch.min(returns)
                    max_return = torch.max(returns)
                    percentile_25 = torch.quantile(returns, 0.25)
                    median_return = torch.quantile(returns, 0.50)
                    percentile_75 = torch.quantile(returns, 0.75)

                    writer.add_scalar('Return/Mean', mean_return, episode)
                    writer.add_scalar('Return/Std', std_return, episode)
                    writer.add_scalar('Return/Min', min_return, episode)
                    writer.add_scalar('Return/Max', max_return, episode)
                    writer.add_scalar('Return/25th Percentile', percentile_25, episode)
                    writer.add_scalar('Return/Median', median_return, episode)
                    writer.add_scalar('Return/75th Percentile', percentile_75, episode)

                    # writer.add_scalar('ep_r', ep_r.mean().item(), episode)
                    with torch.no_grad():
                        writer.add_scalar('alpha', torch.exp(
                            self.log_alpha).item(), episode)
                    if episode % self.arglist.eval_every == 0 or episode == self.arglist.episodes-1:
                        # Evaluate agent performance
                        eval_ep_r_list = self.eval(self.arglist.eval_over)
                        writer.add_scalar('eval_ep_r', np.mean(
                            eval_ep_r_list), episode)
                        self.save_checkpoint(str(episode)+".ckpt")
                    if (episode % 250 == 0 or episode == self.arglist.episodes-1) and episode > self.start_episode:
                        self.save_backup(episode)
                    break

    def eval(self, episodes, render=False):
        # Evaluate agent performance over several episodes
        ep_r_list = []
        for episode in range(episodes):
            o, _, _ = self.env.reset(wide=True)
            ep_r = 0
            while True:
                with torch.no_grad():
                    a, _ = self.actor(torch.tensor(
                        o, dtype=torch.float64, device=self.device).unsqueeze(0), True)
                a = a.cpu().numpy()[0]
                o_1, r, done = self.env.step(a, last=True)
                if render:
                    self.env.render()
                ep_r += r
                o = o_1
                if done:
                    ep_r_list.append(ep_r)
                    if render:
                        print("Episode finished with total reward ", ep_r)
                    break

        if self.arglist.mode == "eval":
            print("Average return :", np.mean(ep_r_list))

        return ep_r_list


def parse_args():
    parser = argparse.ArgumentParser("SAC")
    # Common settings
    parser.add_argument("--env", type=str, default="cart3pole",
                        help="pendulum / reacher / cartpole / acrobot / cart2pole / acro3bot / cart3pole")
    parser.add_argument("--model-path", type=str, default=None,
                        help="path to trained model")
    parser.add_argument("--mode", type=str,
                        default="train", help="train or eval")
    parser.add_argument("--episodes", type=int,
                        default=10000, help="number of episodes")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    # Core training parameters
    parser.add_argument("--resume", action="store_true",
                        default=False, help="resume training")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="actor, critic learning rate")
    parser.add_argument("--gamma", type=float,
                        default=0.99, help="discount factor")
    parser.add_argument("--batch-size", type=int,
                        default=256, help="batch size")
    parser.add_argument("--tau", type=float, default=0.005,
                        help="soft target update parameter")
    parser.add_argument("--start-steps", type=int,
                        default=int(1e4), help="start steps")
    parser.add_argument("--replay-size", type=int,
                        default=int(1e6), help="replay buffer size")
    parser.add_argument("--replay-fill", type=int, default=int(1e4),
                        help="elements in replay buffer before training starts")
    parser.add_argument("--eval-every", type=int, default=50,
                        help="eval every _ episodes during training")
    parser.add_argument("--eval-over", type=int, default=50,
                        help="each time eval over _ episodes")
    # Eval settings
    parser.add_argument("--checkpoint", type=str,
                        default="", help="path to checkpoint")
    parser.add_argument("--render", action="store_true",
                        default=False, help="render")

    return parser.parse_args()


if __name__ == '__main__':
    arglist = parse_args()
    sac = SAC(arglist)
