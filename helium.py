import jax.numpy as jnp
import numpy as np
import jax

from wavefunction import init_network_params, nn_hylleraas
from train import train


if __name__ == '__main__':
    key = jax.random.PRNGKey(0)

    # Initialize wavefunction

    layer_sizes = [3, 32, 32, 32, 1]
    key, subkey = jax.random.split(key)
    wf_params = (jnp.array([2.0]), init_network_params(layer_sizes, subkey))
    #wf_params = pickle.load(open('nn_wf.par', 'rb'))

    # Initialize MCMC

    n_equi = 2048
    n_iter = 16
    n_chains = 4096
    n_steps = 100000
    n_elec = 2
    step_size = 0.5
    eps = 1e-3  # Overlap matrix regularization factor
    # Regularization factor for parameter changes
    def dt(i): return 1 / (1 + i)
    wf = nn_hylleraas

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
