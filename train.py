import jax.numpy as jnp
import numpy as np
import jax

from jax.scipy.sparse.linalg import cg
#from scipy.sparse.linalg import cg, LinearOperator

from mcmc import init_mcmc
from ops import *

import pyblock
import pickle

def train(
    key,
    n_equi,
    n_iter,
    n_chains,
    n_steps,
    n_elec,
    step_size,
    eps,
    dt,
    wf,
    wf_params,
    log_fn = "nn_wf.log"
):
    run_mcmc, run_burnin = init_mcmc(
        lambda p, c: jnp.log(jnp.abs(wf(p, c))), 
        step_size, 
        n_equi, 
        n_iter
    )

    batch_run_burnin = jax.vmap(run_burnin, in_axes=(0, None, 0))
    batch_run_mcmc = jax.jit(jax.vmap(run_mcmc, in_axes=(0, None, 0), out_axes=(0, 0)))

    local_energy = gen_local_energy(wf)
    grad_op, lg_op, ovp, rewrap = gen_grad_operators(wf)
    rewrap = rewrap(wf_params)

    batch_local_energy = jax.jit(
        jax.vmap(
            local_energy, 
            in_axes=(None, 0)
        )
    )

    batch_grad_op = jax.jit(
        jax.vmap(
            grad_op, 
            in_axes=(None, 0, None, None)
        ),
        static_argnums=(2,)
    )

    batch_lg_op = jax.jit(
        jax.vmap(
            lg_op, 
            in_axes=(None, 0)
        )
    )

    batch_ovp = jax.jit(
        jax.vmap(
            ovp, 
            in_axes=(None, 0, None, None, None)
        )
    )

    key, subkey = jax.random.split(key)
    batch_configs = jax.random.normal(subkey, (n_chains, n_elec, 3))

    keys = jax.random.split(subkey, n_chains+1)
    batch_configs = batch_run_burnin(keys[:-1], wf_params, batch_configs)

    log = open(log_fn, "a")

    for i in range(n_steps):
        keys = jax.random.split(keys[-1], n_chains+1)
        batch_configs, batch_accepts = batch_run_mcmc(
              keys[:-1], 
              wf_params, 
              batch_configs[:, :, :]
        )

        energies = batch_local_energy(wf_params, batch_configs)
        stats = pyblock.blocking.reblock(energies)
        optimal_block = pyblock.blocking.find_optimal_block(batch_configs.shape[0], stats)[0]
        accept_rate = jnp.sum(batch_accepts) / batch_configs.shape[0]
        
        print("Energy at step {} : {} pm {}  acceptance rate {}".format(
          i, 
          stats[optimal_block].mean, 
          stats[optimal_block].std_err, 
          accept_rate
        ))
        log.write(
          "{} {} {} {}\n".format(i, stats[optimal_block].mean, stats[optimal_block].std_err, accept_rate)
        )

        sr_E = jnp.mean(batch_grad_op(wf_params, batch_configs, local_energy, stats[optimal_block].mean), axis=0)
        lg_E = jnp.mean(batch_lg_op(wf_params, batch_configs), axis=0)
        reduced_batch_ovp = lambda x: jnp.mean(batch_ovp(wf_params, batch_configs, x, eps, lg_E), axis=0)
        
        #A = LinearOperator((sr_E.shape[0], sr_E.shape[0]), matvec=reduced_batch_ovp)

        dp, _ = cg(reduced_batch_ovp, sr_E, maxiter=500)
        dp = dt(i) * dp
        dp = rewrap(dp)

        wf_params = jax.tree_util.tree_multimap(lambda x, *r: jnp.add(x, *r), wf_params, dp)

        if i % 200 == 0:
          log.flush()
          with open("nn_wf.par", "wb") as f:
            pickle.dump(wf_params, f)

    return wf_params, batch_configs
      
      