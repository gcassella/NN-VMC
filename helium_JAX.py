import jax.numpy as np
import jax
from jax import random, grad, jacfwd, jacrev, vmap, jit, pmap
from jax.ops import index_add, index_update
from functools import partial
from jax.scipy.sparse.linalg import cg

key = random.PRNGKey(0)
key, subkey = random.split(key)

class Wavefunction():
    """
    Helper class to generate functions to evaluate wavefunction gradients.
    If I wrote this notebook again I wouldn't bother with this!! JAX really
    wants you to stick to pure functional programming, and rightly so!

    Attributes
    ----------

    hess : Callable[[ndarray], ndarray]
        Takes config state x, return Hessian matrix of f(x, p0) 
        w.r.t x
    p_grad : Callable[[ndarray, ndarray], ndarray]
        grad(f)(x, p) w.r.t p
    p_gradlog : Callable[[ndarray, ndarray], ndarray]
        grad(log(f))(x, p) w.r.t p
    p_gradlog_eval : Callable[[ndarray], ndarray]
        Evaluate p_gradlog(x, p0) for config state x
    p_grad_eval : Callable[[ndarray], ndarray]
        Evaluate p_grad(x, p0) for config state x
    lapl_eval : Callable[[ndarray], ndarray]
        Evaluate the trace of hess(x) for config state x, equivalent to
        evaluating the laplacian of f w.r.t x
    eval : Callable[[ndarray], ndarray]
        Evaluate f(x) for config state x
    pdf_eval : Callable[[ndarray], ndarray]
        Evaluate |f(x)|^2 for config state x

    Parameters
    ----------

    f : Callable[[ndarray, ndarray], ndarray]
        Wavefunction f(x, p) for config state x containing the electron
        coordinates as a (n_electron, 3) ndarray, and parameter vector p
        containing the variational parameters of f.
    p0 : ndarray
        Variational parameters of f.

    """

    def __init__(self, f, p0):
        self.f = f
        self.p = p0

        self.hess = jacfwd(jacrev(lambda x: self.f(x, self.p), 0), 0)
        self.p_grad = grad(self.f, 1)
        self.p_gradlog = grad(lambda x, p: np.log(self.f(x, p)), 1)

        # Cache evaluations to speed up?
        self.p_gradlog_eval = jit(lambda x: self.p_gradlog(x, self.p))
        self.p_grad_eval = jit(lambda x: self.p_grad(x, self.p))
        self.lapl_eval = jit(lambda x: np.trace(self.hess(x).reshape(x.shape[0]*x.shape[1], x.shape[0]*x.shape[1])))
        self.eval = jit(lambda x: self.f(x, self.p))
        self.pdf_eval = jit(lambda x: np.power(np.abs(self.eval(x)), 2))

@partial(jit, static_argnums=(1,))
def config_step(key, wf, config, config_prob, config_idx, step_size):
    key, subkey = random.split(key)
    move_proposal = random.normal(key, shape=(config.shape[1],))*step_size
    proposal = index_add(config, config_idx%config.shape[0], move_proposal)
    proposal_prob = wf.pdf_eval(proposal)

    uniform = random.uniform(subkey)
    accept = uniform < (proposal_prob / config_prob)

    new_config = np.where(accept, proposal, config)
    config_prob = np.where(accept, proposal_prob, config_prob)
    return new_config, config_prob, config_idx+1

@partial(jit, static_argnums=(1, 2, 3, 4))
def get_configs(key, wf, n_iter, n_equi, step_size, initial_config):
    """
    Carries out Metropolis-Hastings sampling according to the distribution |`wf`|**2.0.
    
    Performs `n_equi` equilibriation steps and `n_iter` sampling steps.
    """
    
    def mh_update(i, state):
      key, config, prob, idx = state
      _, key = random.split(key)
      new_config, new_prob, new_idx = config_step(
          key,
          wf,
          config,
          prob,
          idx,
          step_size
      )
      return (key, new_config, new_prob, new_idx)

    def mh_update_and_store(i, state):
      key, config, prob, idx, configs = state
      _, key = random.split(key)
      new_config, new_prob, new_idx = config_step(
          key,
          wf,
          config,
          prob,
          idx,
          step_size
      )
      new_configs = index_update(configs, idx, new_config)
      return (key, new_config, new_prob, new_idx, new_configs)

    prob = wf.pdf_eval(initial_config)
    key, config, prob, idx = jax.lax.fori_loop(0, n_equi, mh_update, (key, initial_config, prob, 0))
    init_configs = np.zeros((n_iter, *initial_config.shape))
    key, config, prob, idx, configs = jax.lax.fori_loop(0, n_iter, mh_update_and_store, (key, config, prob, 0, init_configs))

    return configs

@partial(jit, static_argnums=(1,))
def itime_hamiltonian(config, wf, tau=0.01):
    n_electron = config.shape[0]
    curr_wf = wf.eval(config)
    acc = 0
    # Calculate kinetic energy
    acc += -0.5*(1/curr_wf)*wf.lapl_eval(config)
    # Calculate electron-electron energy
    for i in range(n_electron):
        for j in range(n_electron):
            if i < j:
                acc += 1 / np.linalg.norm(np.subtract(config[i], config[j]))

    # Calculate electron-nucleus energy, assume z=ne FOR NOW
    for i in range(n_electron):
        acc -= n_electron / np.linalg.norm(config[i])
    # Forget about nucleus - nucleus energy FOR NOW

    return 1-tau*acc

# I don't like this but i can't think of a more elegant way of evaluating
# these operators atm without writing custom code for the ML wavefunction
# that unrolls the parameter list

@partial(jit, static_argnums=(1,))
def sr_op_ml(config, wf):
    gradlog = wf.p_gradlog_eval(config)
    ih = itime_hamiltonian(config, wf)
    
    # reuse gradlog to save memory
    gradlog = np.concatenate((np.array([1]), np.concatenate(tuple(np.concatenate((glw.flatten(), gb.flatten())) for (glw, gb) in gradlog))))
    return np.multiply(ih, gradlog)

@partial(jit, static_argnums=(1,))
def local_energy(config, wf):
    """
    Local energy operator. Uses JAX autograd to obtain laplacian for KE.
    """

    n_electron = config.shape[0]
    acc = 0
    # Calculate kinetic energy
    acc += -0.5*(1/wf.eval(config))*wf.lapl_eval(config)
    # Calculate electron-electron energy
    for i in range(n_electron):
        for j in range(n_electron):
            if i < j:
                acc += 1 / np.linalg.norm(np.subtract(config[i], config[j]))

    # Calculate electron-nucleus energy, assume z=ne FOR NOW
    for i in range(n_electron):
        acc -= n_electron / np.linalg.norm(config[i])

    return acc

@partial(jit, static_argnums=(1,2,))
def monte_carlo(configs, op, wf):
    """
    Performs a Monte Carlo integration using the `configs` walker positions
    of the expectation value of `op` for the wavefunction `wf`.

    Each MCMC chain is broken into samp_rate blocks which are averaged and
    their variance handled to eliminate error due to correlations between
    MCMC samples. See Sorella lecture notes.
    
    Returns the expectation value, variance and a list of the sampled values {O_i}
    """

    samp_rate = 100
    walker_values = vmap(lambda config: op(config, wf))(configs)
    op_output_shape = walker_values[0].shape
    num_blocks = (walker_values.shape[0]//samp_rate)
    blocks = walker_values[:samp_rate*(num_blocks)].reshape((num_blocks, samp_rate, *op_output_shape))
    k = blocks.shape[0]
    block_means = np.mean(blocks, axis=1)
    op_expec = np.mean(block_means, axis=0)
    op_var = 1/(k*(k-1))*np.sum(np.power(block_means - op_expec, 2), axis=0)
    return op_expec, op_var

# config generation and monte carlo integral evaluation mapped over n_chains
# mcmc walkers
run_mcmc = vmap(get_configs, in_axes=(0, None, None, None, None, 0), out_axes=0)
run_int = vmap(monte_carlo, in_axes=(0, None, None), out_axes=0)

def reduce_mc_outs(outs):
    """
    Calculates the mean and variance over the n_chains mcmc walkers, correctly
    preserving the statistics
    """
    
    k = outs[0].shape[0]
    mean = np.mean(outs[0], axis=0)
    variance = (1/k/(k-1))*np.sum(outs[1] + np.power(outs[0] - mean, 2), axis=0)
    return mean, variance

# A helper function to randomly initialize weights and biases
# for a dense neural network layer
def random_layer_params(m, n, key, scale=1):
  w_key, b_key = random.split(key)
  return [scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))]

# Initialize all layers for a fully-connected neural network with sizes "sizes"
def init_network_params(sizes, key):
  keys = random.split(key, len(sizes))
  return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

@jit
def tanh(x):
    return np.tanh(x)

def predict(x, params):
  # per-example predictions
  r = np.linalg.norm(x, axis=1)
  r1 = r[0]
  r2 = r[1]
  u = np.linalg.norm(np.subtract(x[1], x[0]))

  activations = np.array([r1, r2, u])
  for w, b in params[:-1]:
    outputs = np.dot(w, activations) + b
    activations = tanh(outputs)
  
  final_w, final_b = params[-1]
  outputs = np.dot(final_w, activations) + final_b
  return outputs[0]
 
layer_sizes = [3, 12, 12, 1]
key, subkey = random.split(key)
params = init_network_params(layer_sizes, key)

def nn_hylleraas(x, params):
    r = np.linalg.norm(x, axis=1)
    r1 = r[0]
    r2 = r[1]

    s = r1 + r2
    t = r1 - r2
    u = np.linalg.norm(np.subtract(x[1], x[0]))
    return np.exp(-2*s)*(1 + 0.5*u*np.exp(-u))*predict(x, params)

nn_hylleraas_wf = Wavefunction(nn_hylleraas, params)

n_equi = 1000
n_iter = 10000
n_chains = 300
xis = random.uniform(key, (n_chains, 2, 3))
keys = random.split(key, n_chains+1)
ml_wf = Wavefunction(nn_hylleraas, params)

p_wrapped = params

for i in range(400):
  print(i)
  keys = random.split(keys[-1], n_chains+1)
  configs = run_mcmc(keys[:-1], ml_wf, n_iter, n_equi, 0.5, xis)
  E_E, E_V = reduce_mc_outs(run_int(configs, local_energy, ml_wf))

  def odotx(x):
      """
      Calculates the value of $S_{ij}\cdot x_j$ stochastically. 
      
      This is VERY inefficient, because we don't store the stochastic evaluations 
      of the gradlogs inbetween evaluations. However, with the interface that JAX
      uses for conjugate gradient, I can't currently think of cute way of
      doing this (I can think of some very messy ways).
      """

      @partial(jit, static_argnums=(1,))
      def op(c, w):
        gradlog = w.p_gradlog_eval(c)
        gradlog = np.concatenate((np.array([1]), np.concatenate(tuple(np.concatenate((glw.flatten(), gb.flatten())) for (glw, gb) in gradlog))))

        return np.multiply(gradlog, np.dot(gradlog, x))

      E, V = reduce_mc_outs(run_int(configs, op, ml_wf))
      return E

  sr_E, sr_V = reduce_mc_outs(run_int(configs, sr_op_ml, ml_wf))

  dps, _ = cg(odotx, sr_E)

  # Bit of a rigmarole to flatten / unflatten the NN parameters
  p_flat = np.concatenate(tuple(np.concatenate((w.flatten(), b.flatten())) for (w, b) in p_wrapped))
  dps = dps[1:] / dps[0]
  p_flat = np.add(p_flat, dps)

  sizes = layer_sizes
  idx = 0
  p_wrapped = []
  for m, n in zip(sizes[:-1], sizes[1:]):
    p_wrapped.append(
        [p_flat[idx:idx + m*n].reshape((n, m)), p_flat[idx + m*n:idx + (m+1)*(n)]]
    )
    idx += (m+1)*(n)

  ml_wf = Wavefunction(nn_hylleraas, p_wrapped)
  print("{} pm {}".format(E_E, np.sqrt(E_V)))
