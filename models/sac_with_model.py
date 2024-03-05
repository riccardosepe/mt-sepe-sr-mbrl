import random
from collections import deque
from torch.distributions.normal import Normal
import torch.nn.functional as F
import numpy as np
import torch
torch.set_default_dtype(torch.float64)

# Experience replay buffer


class ReplayBuffer:
    def __init__(self, capacity, device):
        self.buffer = deque(maxlen=capacity)
        self.device = device

    @torch.no_grad()
    def push(self, o_tensor, a_tensor, r_tensor, o_1_tensor):
        bs = o_tensor.shape[0]
        for i in range(bs):
            data = torch.cat((o_tensor[i], a_tensor[i], r_tensor[i, None], o_1_tensor[i]))
            if torch.isnan(data).any():
                continue
            self.buffer.append((o_tensor[i], a_tensor[i], r_tensor[i], o_1_tensor[i]))

    def sample(self, batch_size):
        O, A, R, O_1 = zip(*random.sample(self.buffer, batch_size))
        return (torch.stack(O).to(self.device),
                torch.stack(A).to(self.device),
                torch.stack(R).to(self.device),
                torch.stack(O_1).to(self.device))

    def __len__(self):
        return len(self.buffer)

# Critic network


class Q_FC(torch.nn.Module):
    def __init__(self, obs_size, action_size):
        super(Q_FC, self).__init__()
        self.fc1 = torch.nn.Linear(obs_size+action_size, 256)
        self.fc2 = torch.nn.Linear(256, 256)
        self.fc3 = torch.nn.Linear(256, 1)

    def forward(self, x, a):
        y1 = F.relu(self.fc1(torch.cat((x, a), 1)))
        y2 = F.relu(self.fc2(y1))
        y = self.fc3(y2).view(-1)
        return y

# Actor network


class Pi_FC(torch.nn.Module):
    def __init__(self, obs_size, action_size):
        super(Pi_FC, self).__init__()
        self.fc1 = torch.nn.Linear(obs_size, 256)
        self.fc2 = torch.nn.Linear(256, 256)
        self.mu = torch.nn.Linear(256, action_size)
        self.log_sigma = torch.nn.Linear(256, action_size)

    def forward(self, x, deterministic=False, with_logprob=False):
        y1 = F.relu(self.fc1(x))
        y2 = F.relu(self.fc2(y1))
        mu = self.mu(y2)

        if deterministic:
            # used for evaluating policy
            action = torch.tanh(mu)
            log_prob = None
        else:
            log_sigma = self.log_sigma(y2)
            log_sigma = torch.clamp(log_sigma, min=-20.0, max=2.0)
            sigma = torch.exp(log_sigma)
            dist = Normal(mu, sigma)
            x_t = dist.rsample()
            if with_logprob:
                log_prob = dist.log_prob(x_t).sum(1)
                log_prob -= (2*(np.log(2) - x_t - F.softplus(-2*x_t))).sum(1)
            else:
                log_prob = None
            action = torch.tanh(x_t)

        return action, log_prob
