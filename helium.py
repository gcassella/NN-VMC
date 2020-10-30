import jax.numpy as jnp
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
    n_chains = 512
    step_size = 0.3
    tau = lambda i: 1e-3 # Stochastic reconfiguration imaginary 'time step'
    eps = 1e-3 # Overlap matrix regularization factor
    dt = lambda i: 1 / (1e4 + i) # Regularization factor for parameter changes

    run_mcmc, run_burnin = init_mcmc(lambda p, c: jnp.log(jnp.abs(nn_hylleraas(p, c))), step_size, n_equi, n_iter)

    # Create vmapped MCMC funcs for n_chains

    batch_run_burnin = vmap(run_burnin, in_axes=(0, None, 0))
    batch_run_mcmc = vmap(run_mcmc, in_axes=(0, None, 0), out_axes=(0, 0))

    # Create local ops and vmapped versions

    local_energy = gen_local_energy(nn_hylleraas)
    energy_grad = gen_energy_gradient(nn_hylleraas)
    sr_op, ovp, rewrap = gen_sr_operators(nn_hylleraas)
    rewrap = rewrap(wf_params)
    batch_local_energy = jit(vmap(jit(local_energy, static_argnums=(0,)), in_axes=(None, 0)))
    batch_energy_grad = jit(vmap(jit(energy_grad, static_argnums=(0, 2, 3,)), in_axes=(None, 0, None, None), out_axes=0))
    batch_sr_op = jit(vmap(jit(sr_op, static_argnums=(0, 2, 3,)), in_axes=(None, 0, None, None), out_axes=0), static_argnums=(2,))
    batch_ovp = jit(vmap(jit(ovp, static_argnums=(0,3,)), in_axes=(None, 0, None, None), out_axes=0))

    # Initialize configs and run burn-in

    batch_configs = random.normal(subkey, (n_chains, 2, 3))
    key, subkey = random.split(key)

    keys = random.split(subkey, n_chains+1)
    batch_configs = batch_run_burnin(keys[:-1], wf_params, batch_configs)
    batch_configs = jnp.expand_dims(batch_configs, 1)

    for i in range(10000):
      keys = random.split(keys[-1], n_chains+1)
      batch_configs, batch_accepts = batch_run_mcmc(
            keys[:-1], 
            wf_params, 
            batch_configs[:, -1, :, :]
        )
      batch_configs_flat = jnp.concatenate(tuple(batch_configs))
      sr_E = jnp.mean(batch_sr_op(wf_params, batch_configs_flat, local_energy, tau(i)), axis=0)
      reduced_batch_ovp = lambda x: jnp.mean(batch_ovp(wf_params, batch_configs_flat, x, eps), axis=0)

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
      
      A = LinearOperator((sr_E.shape[0], sr_E.shape[0]), matvec=reduced_batch_ovp)

      dp, _ = cg(A, sr_E)
      dp = dt(i) * dp[1:] / dp[0]
      dp = rewrap(dp)

      wf_params = jax.tree_util.tree_multimap(lambda x, *r: jnp.add(x, *r), wf_params, dp)

      if i % 200 == 0:
        with open("nn_wf.par", "wb") as f:
          pickle.dump(wf_params, f)

        xla._xla_callable.cache_clear()
      
      
