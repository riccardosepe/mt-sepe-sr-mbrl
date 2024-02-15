import os
from enum import Enum
from multiprocessing import Process, Pipe
import numpy as np

from env.utils import make_env


class AsyncState(Enum):
    DEFAULT = "default"
    WAITING_RESET = "reset"
    WAITING_STEP = "step"
    WAITING_CALL = "call"


class CustomAsyncVectorEnv:
    def __init__(self, make_env_fn, num_envs):
        self.num_envs = num_envs
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(num_envs)])
        self.ps = [Process(target=self.worker, args=(work_remote, remote, make_env_fn))
                   for (work_remote, remote) in zip(self.work_remotes, self.remotes)]
        for p in self.ps:
            p.start()
        self.states = [AsyncState.DEFAULT for _ in range(num_envs)]

    @staticmethod
    def worker(remote, parent_remote, make_env_fn):
        parent_remote.close()
        env = make_env_fn()
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                ob, reward, done = env.step(data)
                if done:
                    ob = env.reset()
                remote.send((ob, reward, done))
            elif cmd == 'reset':
                ob = env.reset()
                remote.send(ob)
            elif cmd == 'render':
                env.render()
            elif cmd == 'close':
                remote.close()
                break
            else:
                raise NotImplementedError

    def _step_async(self, actions):
        for i, (remote, action) in enumerate(zip(self.remotes, actions)):
            remote.send(('step', action))
            self.states[i] = AsyncState.WAITING_STEP

    def _step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        obs, rewards, dones = zip(*results)
        return np.stack(obs), np.stack(rewards), np.stack(dones)

    def step(self, actions):
        self._step_async(actions)
        return self._step_wait()

    def reset(self):
        for i, remote in enumerate(self.remotes):
            remote.send(('reset', None))
            self.states[i] = AsyncState.WAITING_RESET
        results = [remote.recv() for remote in self.remotes]
        obs, rewards, dones = zip(*results)
        return np.stack(obs), np.stack(rewards), np.stack(dones)

    def render(self):
        for remote in self.remotes:
            remote.send(('render', None))

    def close(self):
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()


if __name__ == "__main__":
    os.chdir("..")
    print(os.getcwd())

    from functools import partial

    # Create a partial function that will create a "pendulum" environment
    make_pendulum_env = partial(make_env, name="soft_reacher", mle=False)

    # Create a vectorized environment with 16 instances of the "pendulum" environment
    vec_env = CustomAsyncVectorEnv(make_pendulum_env, num_envs=2)

    # Reset the environment
    observations, _, _ = vec_env.reset()

    # vec_env.render()
    # Now you can use vec_env just like a normal gym environment, but it will step through 16 environments in parallel.
    print("Observations shape:", observations.shape)

    # Step the environment
    for _ in range(100):
        actions = np.random.uniform(-1, 1, (2, 3))
        observations, rewards, dones = vec_env.step(actions)
        # vec_env.render()

    vec_env.close()
