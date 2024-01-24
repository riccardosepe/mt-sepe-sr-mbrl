import os.path
import random
from collections import deque

import numpy as np
import torch
import wandb
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import safe_mean
from torch.nn import Module, Parameter
from torch.nn.functional import l1_loss, mse_loss
from wandb.integration.sb3 import WandbCallback

from utils import PROJECT_PATH
from utils.model_utils import get_strain_indices

PLT_LABELS = ['bend', 'shear', 'axial', 'bend_vel', 'shear_vel', 'axial_vel']


# Experience replay buffer
class ReplayBuffer:
    def __init__(self, capacity, device):
        self.buffer = deque(maxlen=capacity)
        self.device = device
        self._maxlen = capacity

    @property
    def full(self) -> bool:
        if len(self.buffer) == self._maxlen:
            return True
        elif len(self.buffer) < self._maxlen:
            return False
        else:
            raise ValueError("Buffer length is greater than its capacity")

    def __len__(self):
        return len(self.buffer)

    @property
    def maxlen(self):
        return self._maxlen

    def push(self, o, a, r, o_1):
        # Ensure that the data are all on the CPU (should it be done here?) (should data be entirely on GPU instead?)
        data = torch.cat((o.detach().cpu().flatten(),
                          a.detach().cpu().flatten(),
                          r.detach().cpu().flatten(),
                          o_1.detach().cpu().flatten())).cpu()
        if torch.isnan(data).any():
            raise NanError("You're pushing nans to the buffer")
        self.buffer.append((o, a, r, o_1))

    def sample_transitions(self, batch_size):
        O, A, R, O_1 = zip(*random.sample(self.buffer, batch_size))
        return torch.stack(O), torch.stack(A), torch.stack(R), torch.stack(O_1)

    def sample_states(self, batch_size):
        O, A, R, O_1 = zip(*random.sample(self.buffer, batch_size))
        return torch.stack(O)

    @property
    def states_histograms(self):
        import matplotlib.pyplot as plt
        states = np.array([np.array(s[0]) for s in self.buffer])
        if states.shape[1] != 6:
            raise ValueError(
                f"States have shape {states.shape} instead of (N, 6)")
        fig, axs = plt.subplots(3, 2, figsize=(15, 10))
        for i in range(6):
            axs[i % 3, i // 3].hist(states[:, i], bins=20)
            axs[i % 3, i // 3].set_title(PLT_LABELS[i])

        plt.tight_layout()
        fig.suptitle("States histograms")
        image = wandb.Image(fig)
        plt.close(fig)
        return image

    @property
    def actions_histograms(self):
        import matplotlib.pyplot as plt
        actions = np.array([np.array(sample[1]) for sample in self.buffer])
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
        image = wandb.Image(fig)
        plt.close(fig)
        return image

    @property
    def rewards_histograms(self):
        import matplotlib.pyplot as plt
        rewards = np.array([np.array(s[2]) for s in self.buffer])
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.hist(rewards, bins=10)
        ax.set_title("Rewards")

        plt.tight_layout()
        fig.suptitle("Rewards histograms")
        image = wandb.Image(fig)
        plt.close(fig)
        return image

    def __getitem__(self, item):
        return self.buffer[item]


class NormalizerLoss(Module):
    def __init__(self, env, loss_type='l1', *args, **kwargs):
        super().__init__(*args, **kwargs)

        obs_space_bounds = env.observation_space_bounds

        if type(obs_space_bounds) is not torch.Tensor:
            obs_space_bounds = torch.tensor(obs_space_bounds)
        assert torch.all(obs_space_bounds > 0)
        self.divider = 2*obs_space_bounds
        if loss_type == 'l1':
            self.loss_fn = l1_loss
        elif loss_type == 'l2':
            self.loss_fn = mse_loss
        else:
            raise NotImplementedError(f"Loss {loss_type} not implemented yet.")

        self.env_name = env.name
        if self.env_name != 'jax_pendulum':
            assert 0 < env.num_segments <= 2
            assert len(env.strains) == 3

            self.num_segments = env.num_segments
            self.strains = env.strains

    def forward(self, predictions: torch.Tensor, ground_truth: torch.Tensor, strain: str = None):
        assert predictions.shape == ground_truth.shape, "Predictions and ground truth tensors must have the same shape"

        n = predictions.shape[1]
        losses = torch.empty((n,))

        for i in range(n):
            losses[i] = self.loss_fn(predictions[:, i], ground_truth[:, i])

        # now losses[i] is sum(|pred[:,i] - gt[:, i]|)/BATCH_SIZE

        if strain is None:
            idx = slice(None)
        else:
            try:
                idx = get_strain_indices(
                    self.num_segments, strain, self.strains)
            except AttributeError:
                raise RuntimeError(
                    f"Environment {self.env_name} does not support strains")

        return torch.sum(losses / self.divider[idx])


class NanError(ValueError):
    def __init__(self, *args, cause=None):
        super().__init__(*args)
        self.cause = cause


class BatchNorm(Module):
    def __init__(self, num_features, epsilon=1e-5):
        super(BatchNorm, self).__init__()
        self.num_features = num_features
        self.epsilon = epsilon

        self.gamma = Parameter(torch.ones(num_features))
        self.beta = Parameter(torch.zeros(num_features))

    def forward(self, x):
        mean = x.mean(dim=0)
        variance = (x-mean).pow(2)
        x_normalized = (x - mean) / torch.sqrt(variance + self.epsilon)
        out = self.gamma * x_normalized + self.beta
        return out


class SB3WandBCallback(WandbCallback):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _on_step(self) -> bool:
        super()._on_step()
        # rew = self.locals["rewards"]
        # wandb.log({"reward": rew})
        lr = self.model.ent_coef_optimizer.param_groups[0]['lr']
        wandb.log({"learning rate": lr})
        return True


class WandbRecorderCallback(BaseCallback):
    """
    A custom callback that allows to print stuff on wandb after every evaluation

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """

    def __init__(self, eval_freq=None, wandb_loss_suffix="", verbose=0, save_checkpoint=False):
        super(WandbRecorderCallback, self).__init__(verbose)

        self.wandb_loss_suffix = wandb_loss_suffix
        # self.child_eval_freq = eval_freq
        # self.n_eval_calls = 0
        self.save_checkpoint = save_checkpoint
        self._path = os.path.join(PROJECT_PATH, 'checkpoints', wandb.run.name)
        if save_checkpoint:
            os.makedirs(self._path, exist_ok=True)

    def _on_step(self) -> bool:
        """
        This method is called as a child callback of the `EventCallback`),
        when the event is triggered.
        Optionally, it saves a checkpoint model.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        last_mean_reward = self.parent.last_mean_reward

        # self.n_eval_calls += 1
        # current_timestep = self.n_eval_calls*self.child_eval_freq
        # current_timestep = self.parent.n_calls
        # this number is multiplied by the number of parallel envs
        current_timestep = self.num_timesteps
        wandb.log({"train_mean_reward"+self.wandb_loss_suffix: last_mean_reward,
                  "timestep": current_timestep})

        rollout_rewards = [ep_info['r']
                           for ep_info in self.model.ep_info_buffer]
        wandb.log({"rollout_rewards"+self.wandb_loss_suffix: safe_mean(
            rollout_rewards), "timestep": current_timestep})

        if self.save_checkpoint:
            print(f"Saving model to {self._path}")
            save_path = os.path.join(self._path, "ckpt.pt")
            self.model.save(save_path)
            wandb.save(save_path)
            print("Model saved")

        return True


class VerboseCallback(BaseCallback):
    def _on_step(self) -> bool:
        print(f"Step {self.num_timesteps}")
        return True
