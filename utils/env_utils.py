from functools import partial

from diffrax import Dopri5, ODETerm, diffeqsolve
from jax import Array, jit
import jax.numpy as jnp


def make_env(**kwargs):
    try:
        name = kwargs['name']
    except KeyError:
        raise ValueError("Missing required argument: Environment name")
    if name == "pendulum":
        from environments.old.pendulum.pendulum import Pendulum as Env

    elif name == "reacher":
        from environments.old.reacher.reacher import Reacher as Env

    elif name == "cartpole":
        from environments.old.cartpole.cartpole import Cartpole as Env

    elif name == "acrobot":
        from environments.old.acrobot.acrobot import Acrobot as Env

    elif name == "cart2pole":
        from environments.old.cart2pole.cart2pole import Cart2pole as Env

    elif name == "acro3bot":
        from environments.old.acro3bot.acro3bot import Acro3bot as Env

    elif name == "cart3pole":
        from environments.old.cart3pole.cart3pole import Cart3pole as Env

    elif name == "jax_pendulum":
        from environments.jax_pendulum.pendulum import JaxPendulum as Env
        if 'num_links' not in kwargs:
            raise ValueError(
                f"Missing required parameter num_links in environment {name}")

    elif name == "planar_pcs":
        from environments.jax_planar_pcs.planar_pcs import PlanarPCS as Env
        if 'num_segments' not in kwargs:
            raise ValueError(
                f"Missing required parameter num_segments in environment {name}")
        if 'strains' not in kwargs:
            raise ValueError(
                f"Missing required parameter strains in environment {name}")

    elif name == "planar_hsa":
        from environments.jax_planar_hsa.planar_hsa import PlanarHSA as Env

    else:
        raise ValueError(f"Environment {name} not available.")

    env = Env(**kwargs)

    return env


def step_factory(solver_cls=Dopri5, hsa: bool = False):
    @jit
    def jitted_step(state: Array,
                    tau: Array,
                    t0: int,
                    dt: float,
                    ips: int,
                    dynamical_matrices_fn,
                    ode_factory,
                    params: dict) -> Array:
        ode_fn = ode_factory(dynamical_matrices_fn, params, tau)
        term = ODETerm(ode_fn)

        sol = diffeqsolve(
            term,
            solver=solver_cls(),
            t0=t0 * dt,
            t1=(t0 + ips) * dt,
            dt0=dt,
            y0=state,
            max_steps=None,
        )

        return jnp.asarray(sol.ys).squeeze()

    @jit
    def hsa_jitted_step(t0: int, dt: float, ips: int, tau: Array, dynamical_matrices_fn, ode_factory, params: dict, state: Array) -> Array:
        ode_fn = ode_factory(dynamical_matrices_fn, params)
        term = ODETerm(partial(ode_fn, u=tau))
        sol = diffeqsolve(
            term,
            solver=solver_cls(),
            t0=t0 * dt,
            t1=(t0 + ips) * dt,
            dt0=dt,
            y0=state,
            max_steps=None,
        )
        return jnp.asarray(sol.ys).squeeze()

    if hsa:
        return hsa_jitted_step
    else:
        return jitted_step
