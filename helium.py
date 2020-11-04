import jax.numpy as jnp
import numpy as np
import jax
from jax import random, vmap, jit
from jax.interpreters import xla
#from jax.scipy.sparse.linalg import cg

from scipy.sparse.linalg import cg, LinearOperator

from mcmc import init_mcmc
from ops import *
from wavefunction import init_network_params, nn_hylleraas, hylleraas

import pyblock
import pickle


if __name__ == '__main__':
    key = random.PRNGKey(0)

    # Initialize wavefunction

    layer_sizes = [3, 32, 32, 32, 1]
    key, subkey = random.split(key)
    wf_params = (jnp.array([1.0]), init_network_params(layer_sizes, subkey))
    #wf_params = pickle.load(open('nn_wf.par', 'rb'))

    # Initialize MCMC
    
    n_equi = 2048
    n_iter = 16
    n_chains = 4096
    step_size = 0.5
    eps = 1e-3 # Overlap matrix regularization factor
    dt = lambda i: 1 / (1 + i) # Regularization factor for parameter changes

    run_mcmc, run_burnin = init_mcmc(lambda p, c: jnp.log(jnp.abs(nn_hylleraas(p, c))), step_size, n_equi, n_iter)

    # Create vmapped MCMC funcs for n_chains

    batch_run_burnin = vmap(run_burnin, in_axes=(0, None, 0))
    batch_run_mcmc = vmap(run_mcmc, in_axes=(0, None, 0), out_axes=(0, 0))

    # Create local ops and vmapped versions

    local_energy = gen_local_energy(nn_hylleraas)
    grad_op, lg_op, ovp, rewrap = gen_grad_operators(nn_hylleraas)
    rewrap = rewrap(wf_params)
    batch_local_energy = jit(vmap(jit(local_energy, static_argnums=(0,)), in_axes=(None, 0)))
    batch_grad_op = jit(vmap(jit(grad_op, static_argnums=(0, 2,)), in_axes=(None, 0, None, None), out_axes=0), static_argnums=(2,))
    batch_lg_op = jit(vmap(jit(lg_op, static_argnums=(0,)), in_axes=(None, 0)))
    batch_ovp = jit(vmap(jit(ovp, static_argnums=(0,3,)), in_axes=(None, 0, None, None, None), out_axes=0))

    # Initialize configs and run burn-in

    batch_configs = random.normal(subkey, (n_chains, 2, 3))
    key, subkey = random.split(key)

    keys = random.split(subkey, n_chains+1)
    batch_configs = batch_run_burnin(keys[:-1], wf_params, batch_configs)
    batch_configs = jnp.expand_dims(batch_configs, 1)

    log = open("nn_wf.log", "a")

    for i in range(1000):
      keys = random.split(keys[-1], n_chains+1)
      batch_configs, batch_accepts = batch_run_mcmc(
            keys[:-1], 
            wf_params, 
            batch_configs[:, -1, :, :]
        )

      batch_configs_flat = jnp.concatenate(tuple(batch_configs))

      energies = batch_local_energy(wf_params, batch_configs_flat)
      stats = pyblock.blocking.reblock(energies)
      optimal_block = pyblock.blocking.find_optimal_block(batch_configs_flat.shape[0], stats)[0]
      accept_rate = jnp.sum(batch_accepts) / batch_configs_flat.shape[0]
      print("Energy at step {} : {} pm {}  acceptance rate {}".format(
        i, 
        stats[optimal_block].mean, 
        stats[optimal_block].std_err, 
        accept_rate
      ))
      log.write(
        "{} {} {} {}\n".format(i, stats[optimal_block].mean, stats[optimal_block].std_err, accept_rate)
      )
 
      sr_E = jnp.mean(batch_grad_op(wf_params, batch_configs_flat, local_energy, stats[optimal_block].mean), axis=0)
      lg_E = jnp.mean(batch_lg_op(wf_params, batch_configs_flat), axis=0)
      reduced_batch_ovp = lambda x: jnp.mean(batch_ovp(wf_params, batch_configs_flat, x, eps, lg_E), axis=0)

      A = LinearOperator((sr_E.shape[0], sr_E.shape[0]), matvec=reduced_batch_ovp)

      dp, _ = cg(A, sr_E, maxiter=500)
      dp = dt(i) * dp#[1:] / dp[0]
      dp = rewrap(dp)

      wf_params = jax.tree_util.tree_multimap(lambda x, *r: jnp.add(x, *r), wf_params, dp)

      if i % 200 == 0:
        log.flush()
        with open("nn_wf.par", "wb") as f:
          pickle.dump(wf_params, f)

        xla._xla_callable.cache_clear()
      
      
