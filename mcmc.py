import jax.numpy as jnp
import jax

from jax import jit
from functools import partial

def init_mcmc(wf, step_size, n_equi, n_iter):

    pdf = lambda p, c: 2*wf(p, c)

    @partial(jit, static_argnums=1,)
    def step_mcmc(key, wf_params, config, config_prob, idx):
        key, subkey = jax.random.split(key)
        move_proposal = jax.random.normal(subkey, shape=(config.shape[1],))*step_size
        proposal = jax.ops.index_add(config, 
                                     idx.astype(jnp.uint32)%config.shape[0], 
                                     move_proposal)
        proposal_prob = pdf(wf_params, proposal)

        key, subkey = jax.random.split(key)
        uniform = jnp.log(jax.random.uniform(subkey))
        accept = uniform < proposal_prob - config_prob

        new_config = jnp.where(accept, proposal, config)
        new_prob = jnp.where(accept, proposal_prob, config_prob)

        return new_config, new_prob, accept

    def mh_update(i, state):
        key, config, config_prob, wf_params = state
        key, subkey = jax.random.split(key)
        new_config, new_prob, _ = step_mcmc(
            subkey,
            wf_params,
            config,
            config_prob,
            i
        )
        return (key, new_config, new_prob, wf_params)

    def mh_update_and_store(i, state):
        key, config, config_prob, wf_params, config_buffer, accept_buffer = state
        key, subkey = jax.random.split(key)
        new_config, new_prob, accept = step_mcmc(
            subkey,
            wf_params,
            config,
            config_prob,
            i
        )
        accept_buffer = jax.ops.index_update(accept_buffer, i, accept)
        config_buffer = jax.ops.index_update(config_buffer, i, new_config)
        return (key, new_config, new_prob, wf_params, config_buffer, accept_buffer)

    def run_burnin(key, wf_params, initial_config):
        config = initial_config
        config_prob = pdf(wf_params, initial_config)
        key, subkey = jax.random.split(key)
        key, config, config_prob, _ =  jax.lax.fori_loop(
            jnp.uint32(0),
            jnp.uint32(n_equi),
            mh_update,
            (subkey, config, config_prob, wf_params)
        )

        return config

    def run_mcmc(key, wf_params, initial_config):
        accept_buffer = jnp.zeros((n_iter,), dtype=bool)
        config_buffer = jnp.zeros((n_iter, *initial_config.shape))
        config = initial_config
        config_prob = pdf(wf_params, initial_config)
        key, subkey = jax.random.split(key)
        key, config, config_prob, _, config_buffer, accept_buffer =  jax.lax.fori_loop(
            jnp.uint32(0),
            jnp.uint32(n_iter),
            mh_update_and_store,
            (subkey, config, config_prob, wf_params, config_buffer, accept_buffer)
        )

        return config_buffer, accept_buffer

    return run_mcmc, run_burnin