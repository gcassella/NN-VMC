import jax.numpy as jnp
import jax
from jax import random, vmap, jit
from jax.scipy.sparse.linalg import cg

from functools import partial
from jax.experimental import optimizers

from mcmc import init_mcmc
from ops import *

import pyblock

from wavefunction import init_network_params, nn_hylleraas

if __name__ == '__main__':
    key = random.PRNGKey(0)

    # Initialize wavefunction

    layer_sizes = [3, 32, 32, 32, 1]
    key, subkey = random.split(key)
    wf_params = init_network_params(layer_sizes, subkey)

    # Initialize MCMC
    
    n_equi = 2048
    n_iter = 16
    n_chains = 1024
    step_size = 0.3

    run_mcmc, run_burnin = init_mcmc(lambda p, c: jnp.log(jnp.abs(nn_hylleraas(p, c))), step_size, n_equi, n_iter)

    # Create vmapped MCMC funcs for n_chains

    batch_run_burnin = vmap(run_burnin, in_axes=(0, None, 0))
    batch_run_mcmc = vmap(run_mcmc, in_axes=(0, None, 0), out_axes=(0, 0))

    # Create local ops and vmapped versions

    local_energy = gen_local_energy(nn_hylleraas)
    energy_grad = gen_energy_gradient(nn_hylleraas)
    sr_op, ovp, rewrap = gen_sr_operators(nn_hylleraas)
    batch_local_energy = vmap(local_energy, in_axes=(None, 0))
    batch_energy_grad = vmap(energy_grad, in_axes=(None, 0, None, None), out_axes=0)
    batch_sr_op = vmap(sr_op, in_axes=(None, 0, None, None), out_axes=0)
    batch_ovp = vmap(ovp, in_axes=(None, 0, None), out_axes=0)

    # Initialize configs and run burn-in

    batch_configs = random.normal(subkey, (n_chains, 2, 3))
    key, subkey = random.split(key)

    keys = random.split(subkey, n_chains+1)
    batch_configs = batch_run_burnin(keys[:-1], wf_params, batch_configs)
    batch_configs = jnp.expand_dims(batch_configs, 1)

    # Initialize optimizer

    lr = lambda t: 1.0 / (1.0e2 + t)
    opt_init, opt_update, opt_get_params = optimizers.adam(lr)
    opt_state = opt_init(wf_params)

    # Optimization step
    def step(i, key, prev_configs, opt_state):
        keys = random.split(key, n_chains+1)

        batch_configs, batch_accepts = batch_run_mcmc(
            keys[:-1], 
            opt_get_params(opt_state), 
            prev_configs[:, -1, :, :]
        )
        batch_configs_flat = jnp.concatenate(tuple(batch_configs))

        energies = batch_local_energy(opt_get_params(opt_state), batch_configs_flat)
        stats = pyblock.blocking.reblock(energies)
        optimal_block = pyblock.blocking.find_optimal_block(batch_configs_flat.shape[0], stats)[0]
        batch_grad = batch_energy_grad(wf_params, batch_configs_flat, stats[optimal_block].mean, local_energy)
        grad = jax.tree_util.tree_map(partial(jnp.mean, axis=0), batch_grad)

        opt_state = opt_update(i, grad, opt_state)

        accept_rate = jnp.sum(batch_accepts) / batch_configs_flat.shape[0]

        return opt_state, stats[optimal_block], batch_configs, accept_rate

    for i in range(4000):

      ### ADAM

      #opt_state, stats, batch_configs, accept_rate = step(i, subkey, batch_configs, opt_state)
      #print("Energy at step {} : {} pm {}  acceptance rate {}".format(i, stats.mean, stats.std_err, accept_rate))

      #key, subkey = jax.random.split(key)
      ###################

      ### STOCHASTIC RECONFIGURATION
      
      keys = random.split(keys[-1], n_chains+1)
      batch_configs, batch_accepts = batch_run_mcmc(
            keys[:-1], 
            wf_params, 
            batch_configs[:, -1, :, :]
        )
      batch_configs_flat = jnp.concatenate(tuple(batch_configs))
      sr_E = jnp.mean(batch_sr_op(wf_params, batch_configs_flat, local_energy, lr(i)), axis=0)
      reduced_batch_ovp = lambda x: jnp.mean(batch_ovp(wf_params, batch_configs_flat, x), axis=0)

      if i % 10 == 0:
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
      
      dp, _ = cg(reduced_batch_ovp, sr_E, maxiter=100)
      dp = dp[1:] / dp[0]
      dp = rewrap(dp, layer_sizes)

      wf_params = jax.tree_util.tree_multimap(lambda x, *r: jnp.add(x, *r), wf_params, dp)
      
      
