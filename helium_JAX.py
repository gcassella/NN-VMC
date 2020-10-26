import jax.numpy as np
import jax
from jax import random, vmap, jit
from jax.scipy.sparse.linalg import cg

from jax.experimental import optimizers

from mcmc import init_mcmc
from ops import gen_local_energy, gen_energy_gradient

import pyblock

# A helper function to randomly initialize weights and biases
# for a dense neural network layer
def random_layer_params(m, n, key, scale=1):
  w_key, b_key = random.split(key)
  return [scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))]

# Initialize all layers for a fully-connected neural network with sizes "sizes"
def init_network_params(sizes, key):
  keys = random.split(key, len(sizes))
  return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

def tanh(x):
    return np.tanh(x)

def predict(p, c):
  # per-example predictions
  r = np.linalg.norm(c, axis=1)
  r1 = r[0]
  r2 = r[1]
  u = np.linalg.norm(np.subtract(c[1], c[0]))

  activations = np.array([r1, r2, u])
  for w, b in p[:-1]:
    outputs = np.dot(w, activations) + b
    activations = tanh(outputs)
  
  final_w, final_b = p[-1]
  outputs = np.dot(final_w, activations) + final_b
  return outputs[0]
 
def nn_hylleraas(p, c):
    r = np.linalg.norm(c, axis=-1)
    r1 = r[0]
    r2 = r[1]

    s = r1 + r2
    t = r1 - r2
    u = np.linalg.norm(np.subtract(c[1], c[0]))
    return np.exp(-2*s)*(1 + 0.5*u*np.exp(-u))*predict(p, c)

if __name__ == '__main__':
    key = random.PRNGKey(0)

    # Initialize wavefunction

    layer_sizes = [3, 32, 32, 32, 1]
    key, subkey = random.split(key)
    wf_params = init_network_params(layer_sizes, subkey)

    # Initialize MCMC
    
    n_equi = 2048
    n_iter = 32
    n_chains = 512
    step_size = 0.3

    run_mcmc, mcmc_params = init_mcmc(lambda p, c: np.log(np.abs(nn_hylleraas(p, c))), step_size, n_equi, n_iter)

    # Create vmapped MCMC funcs for n_chains

    batch_run_mcmc = vmap(run_mcmc, in_axes=(0, None, 0, 0), out_axes=(0, 0, 0))

    # Create local ops and vmapped versions

    local_energy = gen_local_energy(nn_hylleraas)
    energy_grad = gen_energy_gradient(nn_hylleraas)
    batch_local_energy = vmap(local_energy, in_axes=(None, 0))
    batch_energy_grad = vmap(energy_grad, in_axes=(None, 0, None, None))

    # Initialize configs

    batch_configs = np.expand_dims(random.normal(subkey, (n_chains, 2, 3)), 1)
    key, subkey = random.split(key)
    batch_mcmc_params = np.array([mcmc_params for i in range(n_chains)])

    # Initialize optimizer

    opt_init, opt_update, opt_get_params = optimizers.adam(1e-2)
    opt_state = opt_init(wf_params)

    def step(i, key, prev_configs, prev_mcmc_params, opt_state):
        keys = random.split(key, n_chains+1)

        batch_configs, batch_accepts, batch_mcmc_params = batch_run_mcmc(
            keys[:-1], 
            opt_get_params(opt_state), 
            prev_configs[:, -1, :, :], 
            prev_mcmc_params
        )
        batch_configs_flat = np.concatenate(tuple(batch_configs))

        energies = batch_local_energy(opt_get_params(opt_state), batch_configs_flat)
        stats = pyblock.blocking.reblock(energies)
        optimal_block = pyblock.blocking.find_optimal_block(batch_configs_flat.shape[0], stats)[0]
        grads = batch_energy_grad(wf_params, batch_configs_flat, stats[optimal_block].mean, local_energy)

        opt_state = opt_update(i, grads, opt_state)

        return opt_state, stats[optimal_block].mean, batch_configs, batch_mcmc_params

    for i in range(4000):
      opt_state, value, batch_configs, batch_mcmc_params = step(i, subkey, batch_configs, batch_mcmc_params, opt_state)
      print("Energy at step {} : {}".format(i, value))

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
