import logging

import distinctipy
import numpy as np
import torch
import wandb
from matplotlib import pyplot as plt
from tqdm import tqdm

from environments.jax_base_env.base_env import JaxBaseEnv
from utils.utility_classes import NanError

__all__ = ['rollout_error']


def _end_effector_plot(data, total_length):
    time = data['time']
    fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    ax[0].plot(time, data['x_pred'], label='x_pred',
               color='red', linestyle='--')
    ax[0].plot(time, data['x_gt'], label='x_gt',
               color='red', linestyle='solid')

    ax[1].plot(time, data['y_pred'], label='y_pred',
               color='blue', linestyle='--')
    ax[1].plot(time, data['y_gt'], label='y_gt',
               color='blue', linestyle='solid')

    ax[2].plot(time, np.array(data['e_x']) / total_length,
               label='e_x', color='red', linestyle='solid')
    ax[2].plot(time, np.array(data['e_y']) / total_length,
               label='e_y', color='blue', linestyle='solid')
    ax[2].get_xaxis().set_visible(True)

    ax[0].set_title('x')
    ax[1].set_title('y')
    ax[2].set_title('error')

    ax[0].legend()
    ax[1].legend()
    ax[2].legend()

    plt.close(fig)
    return fig


def rollout_plots(env, data, logger, obs_gt_list, obs_pred_list, state_space_bounds):
    action = data['action']
    time = data['time']

    assert len(action) == len(time)
    if len(action) == 0:
        # No data to plot: return empty images
        logger.log("No data to plot", level=logging.WARNING)
        fig1, ax1 = plt.subplots(figsize=(10, 10))
        ax1.axis('off')

        # Add centered text to the axis
        text = "No data to plot"
        ax1.text(0.5, 0.5, text, ha='center', va='center', fontsize=20)

        fig2, ax2 = plt.subplots(figsize=(10, 10))
        ax2.axis('off')

        # Add centered text to the axis
        ax2.text(0.5, 0.5, text, ha='center', va='center', fontsize=20)

    else:
        fig1 = _end_effector_plot(data, env.total_length)
        if env.name == 'jax_pendulum':
            fig2 = _pendulum_obs_plots(
                data, obs_gt_list, obs_pred_list, state_space_bounds)
        else:
            fig2 = _sr_obs_plots(data, obs_gt_list, obs_pred_list,
                                 state_space_bounds, env.strains, env.num_segments)

        fig1.suptitle('End effector plots')
        fig2.suptitle('Observation plots')
        fig1.tight_layout()
        fig2.tight_layout()

        return fig1, fig2


def _sr_obs_plots(data, obs_gt_list, obs_pred_list, state_space_bounds, strains, num_segments):
    time = data['time']
    num_strains = strains.count(1)

    assert num_segments == 1
    # Observation plots
    fig, ax = plt.subplots(num_strains+1, 2, figsize=(10, 5*(num_strains+1)))

    obs_gt_list = np.array(obs_gt_list) / state_space_bounds.cpu().numpy()
    obs_pred_list = np.array(obs_pred_list) / state_space_bounds.cpu().numpy()

    colors = ['red', 'blue', 'green']
    strain_names = ['bend', 'shear', 'axial']
    j = 0
    for i in range(3):
        if not bool(strains[i]):
            continue
        ax[j, 0].plot(time, obs_gt_list[:, j],
                      label=f'{strain_names[i]}_gt', color=colors[i], linestyle='solid')
        ax[j, 0].plot(time, obs_pred_list[:, j],
                      label=f'{strain_names[i]}_pred', color=colors[i], linestyle='--')
        ax[j, 0].legend()
        ax[j, 0].set_title(f'{strain_names[i]}')

        ax[j, 1].plot(time, obs_gt_list[:, j+num_strains],
                      label=f'{strain_names[i]}_vel_gt', color=colors[i], linestyle='solid')
        ax[j, 1].plot(time, obs_pred_list[:, j+num_strains],
                      label=f'{strain_names[i]}_vel_pred', color=colors[i], linestyle='--')
        ax[j, 1].legend()
        ax[j, 1].set_title(f'{strain_names[i]}_vel')

        err = np.abs(obs_gt_list[:, j]-obs_pred_list[:, j])
        err_vel = np.abs(
            obs_gt_list[:, j+num_strains]-obs_pred_list[:, j+num_strains])

        ax[num_strains, 0].plot(
            time, err, label=f'abs_err_{strain_names[i]}', color=colors[i], linestyle='solid')
        ax[num_strains, 1].plot(
            time, err_vel, label=f'abs_err_{strain_names[i]}_vel', color=colors[i], linestyle='solid')

        j += 1

    ax[num_strains, 0].legend()
    ax[num_strains, 1].legend()
    ax[num_strains, 0].set_title('errors pos')
    ax[num_strains, 1].set_title('errors vel')

    plt.close(fig)
    return fig


def _pendulum_obs_plots(data, obs_gt_list, obs_pred_list, state_space_bounds):
    time = data['time']
    # Observation plots
    obs_size = len(obs_gt_list[0])
    num_links = obs_size // 3
    fig, ax = plt.subplots(num_links+1, 3, figsize=(15, 5*(num_links+1)))

    obs_gt_list = np.array(obs_gt_list) / state_space_bounds.cpu().numpy()
    obs_pred_list = np.array(obs_pred_list) / state_space_bounds.cpu().numpy()

    colors = distinctipy.get_colors(num_links, pastel_factor=0.7)

    for i in range(num_links):
        ax[i, 0].plot(time, obs_gt_list[:, i*num_links],
                      label=f'cos(q_{i})_gt [rad]', color=colors[i], linestyle='solid')
        ax[i, 0].plot(time, obs_pred_list[:, i*num_links],
                      label=f'cos(q_{i})_pred  [rad]', color=colors[i], linestyle='--')
        ax[i, 0].legend()
        ax[i, 0].set_title(f'cos(q_{i})')

        ax[i, 1].plot(time, obs_gt_list[:, i*num_links+1],
                      label=f'sin(q_{i})_gt [rad]', color=colors[i], linestyle='solid')
        ax[i, 1].plot(time, obs_pred_list[:, i*num_links+1],
                      label=f'sin(q_{i})_pred [rad]', color=colors[i], linestyle='--')
        ax[i, 1].legend()
        ax[i, 1].set_title(f'sin(q_{i})')

        ax[i, 2].plot(time, obs_gt_list[:, 2*num_links+i],
                      label=f'q_dot_{i}_gt [rad/s]', color=colors[i], linestyle='solid')
        ax[i, 2].plot(time, obs_pred_list[:, 2*num_links+i],
                      label=f'q_dot_{i}_pred [rad/s]', color=colors[i], linestyle='--')
        ax[i, 2].legend()
        ax[i, 2].set_title(f'q_dot_{i}')

        err_cos = np.abs(obs_gt_list[:, i*num_links] -
                         obs_pred_list[:, i*num_links])
        err_sin = np.abs(
            obs_gt_list[:, i*num_links+1]-obs_pred_list[:, i*num_links+1])
        err_vel = np.abs(
            obs_gt_list[:, 2*num_links+i]-obs_pred_list[:, 2*num_links+i])

        ax[num_links, 0].plot(
            time, err_cos, label=f'abs_err_cos(q_{i})', color=colors[i], linestyle='solid')
        ax[num_links, 1].plot(
            time, err_sin, label=f'abs_err_sin(q_{i})', color=colors[i], linestyle='solid')
        ax[num_links, 2].plot(
            time, err_vel, label=f'abs_err_q_dot_{i}', color=colors[i], linestyle='solid')

    ax[num_links, 0].legend()
    ax[num_links, 1].legend()
    ax[num_links, 2].legend()
    ax[num_links, 0].set_title('absolute errors cosine of positions')
    ax[num_links, 1].set_title('absolute errors sine of positions')
    ax[num_links, 2].set_title('absolute errors velocities')

    plt.close(fig)
    return fig


@torch.no_grad()
def rollout_error(model: torch.nn.Module,
                  env: JaxBaseEnv,
                  max_time: int,
                  control_frequency,
                  step,
                  render: bool = False,
                  return_plots: bool = False):
    action_scale = env.action_space_bounds
    device = model.device
    obs_space_bounds = torch.tensor(env.observation_space_bounds).to(device)

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
    obs = torch.tensor(obs)[None].to(device)

    const_action = 0.5 * np.ones_like(action_scale)

    n = env.n

    pos_tolerances = torch.tensor(env.pos_tolerances).to(device)
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

        act = torch.tensor(const_action * action_scale)[None].to(device)
        for _ in range(action_repeat):
            # Here keeping action repeat as an explicit loop in order to collect more samples and have smoother plots
            obs_gt, _, done = env.step(
                const_action, 1, only_trunc=True, time_limit=False)
            pbar.update(1)
            try:
                # Also here keeping action repeat as an explicit loop
                obs_pred = model(obs, act, 1).to(device)
            except NanError:
                done = True
                print("Breaking for nan in prediction")
                break
            except BaseException as e:
                raise e
            finally:
                if (torch.abs(obs_pred[:, :-n]) > pos_tolerances*obs_space_bounds[:-n]).any():
                    done = True
                    print("Breaking for infeasible state")
                    break
                if (torch.abs(obs_pred[:, -n:]) > 10*obs_space_bounds[-n:]).any():
                    done = True
                    print("Breaking for infeasible velocity")
                    break
            if render:
                env.render(pred=obs_pred.detach().cpu().numpy())
            cartesian_gt = env.cartesian_from_obs(obs_gt, numpy=True)
            ee_gt = cartesian_gt[-1, :]

            cartesian_pred = env.cartesian_from_obs(
                obs_pred.squeeze(), numpy=True)
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
            obs_pred_list.append(obs_pred.cpu().numpy().squeeze())

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
        fig1, fig2 = rollout_plots(
            env, data, model.logger, obs_gt_list, obs_pred_list, obs_space_bounds)
        return {
            'step': step,
            'rollout/ee': wandb.Image(fig1),
            'rollout/state': wandb.Image(fig2),
        }

    else:
        return data
