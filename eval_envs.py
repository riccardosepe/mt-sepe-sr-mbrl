import time

import numpy as np
from tqdm import trange

from env.soft_reacher.soft_reacher import SoftReacher as Env
# from env.cartpole.cartpole import cartpole as Env


def main3():
    env = Env(mle=False)

    env.reset()
    env.render()

    while True:
        states = []
        velocities = []
        v = float(input("Enter action: "))
        a = v * np.ones(env.action_size)
        a[-1] = np.abs(a[-1])
        for i in range(100):
            env.step(a)
            env.render()
            states.append(env.get_obs()[:env.n])
            velocities.append(env.get_obs()[-env.n:])

        states = np.array(states)
        states = np.abs(states)
        print("Max state: ", np.max(states, axis=0))
        print("Min state: ", np.min(states, axis=0))
        print("Mean state: ", np.mean(states, axis=0))
        print("Std state: ", np.std(states, axis=0))

        velocities = np.array(velocities)
        velocities = np.abs(velocities)
        print("Max abs velocity: ", np.max(velocities, axis=0))
        print("Min abs velocity: ", np.min(velocities, axis=0))
        print("Mean abs velocity: ", np.mean(velocities, axis=0))
        print("Std abs velocity: ", np.std(velocities, axis=0))


def main():
    env = Env(mle=False)

    env.reset()

    t = time.time()

    for _ in range(100000):
        env.step(np.random.uniform(-1, 1, env.action_size))
        # env.render()

    print(time.time() - t)


if __name__ == '__main__':
    main()
