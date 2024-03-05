import jax.numpy as jnp
import jax
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
import pickle
import haiku as hk
import torch
import wandb
from jax import config
from tqdm import tqdm

import DeLaN_model_v4 as delan
from env.utils import make_env
from rollout_plots import rollout_plots
from utils import seed_all

seed_all(0)
config.update("jax_enable_x64", True)


def rk4_step(f, x, y, t, h):
    # one step of runge-kutta integration
    k1 = h * f(x, y, t)
    # print(k1.shape)
    # print(x.shape)
    # print(x + k1/2)
    k2 = h * f(x + k1 / 2, y, t + h / 2)
    k3 = h * f(x + k2 / 2, y, t + h / 2)
    k4 = h * f(x + k3, y, t + h)
    return x + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


total_time = 5

env = make_env(name="soft_reacher")

with open(f"one_segment_spatial_soft_robot_delan_3D.jax", 'rb') as f:
    data = pickle.load(f)


hyper = data["hyper"]
params = data["params"]

activations = {
    'tanh': jnp.tanh,
    'softplus': jax.nn.softplus,
    'sigmoid': jax.nn.sigmoid,
}

lagrangian_fn = hk.transform(partial(
    delan.structured_lagrangian_fn,
    n_dof=hyper['n_dof'],
    shape=(hyper['n_width'],) * hyper['n_depth'],
    activation=activations[hyper['activation1']],
    epsilon=hyper['diagonal_epsilon'],
    shift=hyper['diagonal_shift'],
))

#dissipative_matrix(qd, n_dof, shape, activation)
dissipative_fn = hk.transform(partial(
    delan.dissipative_matrix,
    n_dof=hyper['n_dof'],
    shape=(5,) * 3,
    activation=activations[hyper['activation2']]
))

#input_transform_matrix(q, n_dof, actuator_dof, shape, activation)
input_mat_fn = hk.transform(partial(
    delan.input_transform_matrix,
    n_dof=hyper['n_dof'],
    actuator_dof=hyper['actuator_dof'],
    shape=(hyper['n_width']//2,) * (hyper['n_depth']-1),
    activation=activations[hyper['activation1']]
))



lagrangian = lagrangian_fn.apply
dissipative_mat = dissipative_fn.apply
input_mat = input_mat_fn.apply

forward_model = jax.jit(delan.forward_model(params=params, key=None, lagrangian=lagrangian, dissipative_mat=dissipative_mat,
                                    input_mat=input_mat, n_dof=hyper['n_dof']))


def rollout_error(model,
                  env,
                  max_time: int,
                  control_frequency,
                  step,
                  render: bool = False,
                  return_plots: bool = False):
    action_scale = env.action_space_bounds
    obs_space_bounds = env.observation_space_bounds

    x_pred, y_pred, x_gt, y_gt, e_x, e_y = [], [], [], [], [], []
    time = []
    control = []
    obs_gt_list, obs_pred_list = [], []

    obs, _, _ = env.reset()
    done = False
    fcont = control_frequency
    max_i = int(max_time * fcont)
    action_repeat = int(env.sample_frequency / fcont)
    assert action_repeat > 0

    const_action = 0.5 * np.ones_like(action_scale)

    n = env.n

    pos_tolerances = env.pos_tolerances
    if env.name != 'jax_pendulum':
        mask = np.array(env.strains, dtype=bool)
        pos_tolerances = pos_tolerances[mask]
    # if env.name == 'planar_pcs':
    #     assert env.num_strains == 3

    if render:
        env.render()
    pbar = tqdm(range(int(max_i * env.sample_frequency)))
    pbar.set_description("Testing")
    for _ in range(max_i):
        if done:
            print("Breaking for done")
            break

        const_action = -const_action

        act = const_action * action_scale
        for _ in range(action_repeat):
            # Here keeping action repeat as an explicit loop in order to collect more samples and have smoother plots
            obs_gt, _, done = env.step(const_action, 1, only_trunc=True, time_limit=False)
            pbar.update(1)
            try:
                # Also here keeping action repeat as an explicit loop
                obs_pred = model(obs, act)
            except ValueError:
                done = True
                print("Breaking for nan in prediction")
                break
            except BaseException as e:
                raise e

            # if (jnp.abs(obs_pred[:, :-n]) > pos_tolerances*obs_space_bounds[:-n]).any():
            #     done = True
            #     print("Breaking for infeasible state")
            #     break
            # if(jnp.abs(obs_pred[:, -n:]) > 10*obs_space_bounds[-n:]).any():
            #     done = True
            #     print("Breaking for infeasible velocity")
            #     break
            if render:
                env.render(pred=np.asarray(obs_pred))
            cartesian_gt = env.cartesian_from_obs(obs_gt, numpy=True)
            ee_gt = cartesian_gt[-1, :]

            cartesian_pred = env.cartesian_from_obs(np.asarray(obs_pred), numpy=True)
            ee_pred = cartesian_pred[-1, :]

            time.append(env.time)
            x_gt.append(ee_gt[0])
            y_gt.append(ee_gt[1])
            x_pred.append(ee_pred[0])
            y_pred.append(ee_pred[1])
            e_x.append(abs(ee_pred[0] - ee_gt[0]))
            e_y.append(abs(ee_pred[1] - ee_gt[1]))
            control.append(const_action * action_scale)
            obs_gt_list.append(obs_gt)
            obs_pred_list.append(np.asarray(obs_pred))

            if len(obs_pred.shape) < len(obs.shape):
                obs_pred = obs_pred[None]
            obs = obs_pred

    print(f"Rollout simulation stopped at {env.time:.2f} seconds.")

    data = {
        'time': time,
        'x_pred': x_pred,
        'y_pred': y_pred,
        'x_gt': x_gt,
        'y_gt': y_gt,
        'e_x': e_x,
        'e_y': e_y,
        'action': np.array(control),
        'step': step,
    }

    if return_plots:
        fig1, fig2 = rollout_plots(env, data, None, obs_gt_list, obs_pred_list, torch.tensor(obs_space_bounds))
        # return {
        #     'step': step,
        #     'rollout/ee': wandb.Image(fig1),
        #     'rollout/state': wandb.Image(fig2),
        # }

        return fig1, fig2

    else:
        return data


# f1, f2 = rollout_error(
#     model=partial(rk4_step, forward_model, t=0.0, h=2e-4),  # TODO
#     env=env,
#     max_time=total_time,
#     control_frequency=1,
#     step=0,
#     render=True,
#     return_plots=True
# )


if __name__ == '__main__':
    model = partial(rk4_step, forward_model, t=0.0, h=1e-4)

    rollout_plots(env, model, epoch=0)


# f1.savefig("f1.png")
# f2.savefig("f2.png")
