import jax.numpy as jnp
import numpy as np
import jax

from wavefunction import init_network_params
from train import train


def predict(p, c):
  # per-example predictions
  r = jnp.linalg.norm(c, axis=1)
  r1 = r[0]
  r2 = r[1]
  r3 = r[2]

  r12 = jnp.linalg.norm(jnp.subtract(c[1], c[0]))
  r13 = jnp.linalg.norm(jnp.subtract(c[2], c[0]))
  r23 = jnp.linalg.norm(jnp.subtract(c[2], c[1]))

  activations = jnp.array([r1, r2, r3, r12, r13, r23])
  for w, b in p[:-1]:
    outputs = jnp.dot(w, activations) + b
    activations = jnp.tanh(outputs)
  
  final_w, final_b = p[-1]
  outputs = jnp.dot(final_w, activations) + final_b
  return outputs[0]

def nn_lithium(p, c):
    # params structure:
    # (
    #   [pi1, pi2, pi3],
    #   network_params  
    # )
    r = jnp.linalg.norm(c, axis=-1)

    perm1 = jnp.array([c[0], c[1], c[2]])
    perm2 = jnp.array([c[1], c[0], c[2]])

    phi_1 = predict(p[1], perm1)
    phi_2 = predict(p[1], perm2)

    return jnp.exp(-jnp.dot(p[0], r))*(phi_1 - phi_2)

if __name__ == '__main__':
    key = jax.random.PRNGKey(0)

    # Initialize wavefunction

    layer_sizes = [6, 32, 32, 32, 1]

    key, subkey = jax.random.split(key)
    network_params = init_network_params(layer_sizes, subkey)

    wf_params = (
        jnp.array([3.0, 3.0, 3.0]),
        network_params
    )

    # Initialize MCMC

    n_equi = 2048
    n_iter = 16
    n_chains = 4096
    n_steps = 100000
    n_elec = 3
    step_size = 0.5
    eps = 1e-3  # Overlap matrix regularization factor
    # Regularization factor for parameter changes
    def dt(i): return 1 / (1e2 + i)
    wf = nn_lithium

    wf_params, batch_configs = train(
        subkey,
        n_equi,
        n_iter,
        n_chains,
        n_steps,
        n_elec,
        step_size,
        eps,
        dt,
        wf,
        wf_params
    )
