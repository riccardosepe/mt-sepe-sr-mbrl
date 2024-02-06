import numpy as np

from env.soft_reacher.soft_reacher import SoftReacher as Env
# from env.cartpole.cartpole import cartpole as Env


def main():
    env = Env(mle=False)

    env.reset()
    env.render()

    while True:
        velocities = []
        v = float(input("Enter action: "))
        a = v * np.ones(env.action_size)
        for _ in range(100):
            env.step(a)
            env.render()
            velocities.append(env.state[-env.n:])
        velocities = np.array(velocities)
        velocities = np.abs(velocities)
        print("Max abs velocity: ", np.max(velocities, axis=0))
        print("Min abs velocity: ", np.min(velocities, axis=0))
        print("Mean abs velocity: ", np.mean(velocities, axis=0))
        print("Std abs velocity: ", np.std(velocities, axis=0))


if __name__ == '__main__':
    main()
