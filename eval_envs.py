import numpy as np

from env.soft_reacher.soft_reacher import SoftReacher as Env
# from env.cartpole.cartpole import cartpole as Env


def main():
    env = Env(mle=False)

    env.reset()
    env.render()

    while True:
        states = []
        velocities = []
        v = float(input("Enter action: "))
        a = v * np.ones(env.action_size)
        a[-1] = np.abs(a[-1])
        for i in range(5000):
            env.step(a)
            if i % 100 == 0:
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


if __name__ == '__main__':
    main()
