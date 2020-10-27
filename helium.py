import jax.numpy as np
import jax
from jax import random, vmap, jit
from jax.scipy.sparse.linalg import cg

from functools import partial
from jax.experimental import optimizers

from mcmc import init_mcmc
from ops import gen_local_energy, gen_energy_gradient

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

    run_mcmc, run_burnin = init_mcmc(lambda p, c: np.log(np.abs(nn_hylleraas(p, c))), step_size, n_equi, n_iter)

    # Create vmapped MCMC funcs for n_chains

    batch_run_burnin = vmap(run_burnin, in_axes=(0, None, 0))
    batch_run_mcmc = vmap(run_mcmc, in_axes=(0, None, 0), out_axes=(0, 0))

    # Create local ops and vmapped versions

    local_energy = gen_local_energy(nn_hylleraas)
    energy_grad = gen_energy_gradient(nn_hylleraas)
    batch_local_energy = vmap(local_energy, in_axes=(None, 0))
    batch_energy_grad = vmap(energy_grad, in_axes=(None, 0, None, None), out_axes=0)

    # Initialize configs and run burn-in

    batch_configs = random.normal(subkey, (n_chains, 2, 3))
    key, subkey = random.split(key)

    keys = random.split(subkey, n_chains)
    batch_configs = batch_run_burnin(keys, wf_params, batch_configs)
    batch_configs = np.expand_dims(batch_configs, 1)

    # Initialize optimizer

    lr = lambda t: 1.0 / (1.0e4 + t)
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
        batch_configs_flat = np.concatenate(tuple(batch_configs))

        energies = batch_local_energy(opt_get_params(opt_state), batch_configs_flat)
        stats = pyblock.blocking.reblock(energies)
        optimal_block = pyblock.blocking.find_optimal_block(batch_configs_flat.shape[0], stats)[0]
        batch_grad = batch_energy_grad(wf_params, batch_configs_flat, stats[optimal_block].mean, local_energy)
        grad = jax.tree_util.tree_map(partial(np.mean, axis=0), batch_grad)

        opt_state = opt_update(i, grad, opt_state)

        accept_rate = np.sum(batch_accepts) / batch_configs_flat.shape[0]

        return opt_state, stats[optimal_block], batch_configs, accept_rate

    for i in range(4000):
      opt_state, stats, batch_configs, accept_rate = step(i, subkey, batch_configs, opt_state)
      print("Energy at step {} : {} pm {}  acceptance rate {}".format(i, stats.mean, stats.std_err, accept_rate))

      key, subkey = jax.random.split(key)
      
      #reduce_mc_outs(run_int_batch(batch_configs, local_energy, ml_wf))

      #def odotx(x):
      #    """
      #    Calculates the value of $S_{ij}\cdot x_j$ stochastically. 

      #    This is VERY inefficient, because we don't store the stochastic evaluations 
      #    of the gradlogs inbetween evaluations. However, with the interface that JAX
      #    uses for conjugate gradient, I can't currently think of cute way of
      #    doing this (I can think of some very messy ways).
      #    """

      #    @partial(jit, static_argnums=(1,))
      #    def op(c, w):
      #      gradlog = w.p_gradlog_eval(c)
      #      gradlog = np.concatenate((np.array([1]), np.concatenate(tuple(np.concatenate((glw.flatten(), gb.flatten())) for (glw, gb) in gradlog))))

      #      return np.multiply(gradlog, np.dot(gradlog, x))

      #    E, V = reduce_mc_outs(run_int_batch(batch_configs, op, ml_wf))
      #    return E

      #sr_E, sr_V = reduce_mc_outs(run_int_batch(batch_configs, sr_op_ml, ml_wf))

      #dps, _ = cg(odotx, sr_E)

      ## Bit of a rigmarole to flatten / unflatten the NN parameters
      #p_flat = np.concatenate(tuple(np.concatenate((w.flatten(), b.flatten())) for (w, b) in p_wrapped))
      #dps = dps[1:] / dps[0]
      #p_flat = np.add(p_flat, dps)

      #sizes = layer_sizes
      #idx = 0
      #p_wrapped = []
      #for m, n in zip(sizes[:-1], sizes[1:]):
      #  p_wrapped.append(
      #      [p_flat[idx:idx + m*n].reshape((n, m)), p_flat[idx + m*n:idx + (m+1)*(n)]]
      #  )
      #  idx += (m+1)*(n)

      #ml_wf = Wavefunction(nn_hylleraas, p_wrapped)
      #print("{} pm {}".format(E_E, np.sqrt(E_V)))
