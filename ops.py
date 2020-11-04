import jax
from jax._src.numpy.lax_numpy import ravel
import jax.numpy as jnp

from functools import partial

from jax.flatten_util import ravel_pytree

def gen_local_energy(wf):
    hess = jax.jacfwd(jax.grad(wf, 1), 1)
    lapl_eval = lambda p, c: jnp.trace(hess(p, c).reshape(c.shape[0]*c.shape[1], c.shape[0]*c.shape[1]))

    def local_energy(p, c):
        """
        Local energy operator. Uses JAX autograd to obtain laplacian for KE.
        """

        n_electron = c.shape[0]
        acc = 0
        # Calculate kinetic energy
        acc += -0.5*(1/wf(p, c))*lapl_eval(p, c)
        # Calculate electron-electron energy
        for i in range(n_electron):
            for j in range(n_electron):
                if i < j:
                    acc += 1 / jnp.linalg.norm(jnp.subtract(c[i], c[j]))

        # Calculate electron-nucleus energy, assume z=ne FOR NOW
        for i in range(n_electron):
            acc -= n_electron / jnp.linalg.norm(c[i])

        return acc

    return local_energy

def gen_grad_operators(wf):
    log_grad = jax.jit(jax.grad(lambda p, c: jnp.log(jnp.abs(wf(p, c)))), static_argnums=(0,))
    def grad_op(p, c, local_energy_op, local_energy_exp):
        lg = log_grad(p, c)
        log_grad_flat, _ = ravel_pytree(lg)

        return jnp.multiply((local_energy_exp - local_energy_op(p, c)), log_grad_flat)

    def lg_op(p,c):
        lg = log_grad(p, c)
        log_grad_flat, _ = ravel_pytree(lg)

        return log_grad_flat

    def ovp(p, c, x, eps, log_grad_exp):
        lg = log_grad(p, c)
        log_grad_flat, _ = ravel_pytree(lg)

        return jnp.multiply(log_grad_flat - log_grad_exp, jnp.dot(log_grad_flat - log_grad_exp, x)) + eps*jax.ops.index_update(x, 0, 0.0)

    def get_rewrap(p):
        _, f = ravel_pytree(p)
        return f
    
    return grad_op, lg_op, ovp, get_rewrap