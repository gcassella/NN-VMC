import jax
import jax.numpy as jnp

import jax.flatten_util

def gen_local_energy(wf):
    hess = jax.jacfwd(jax.grad(wf, 1), 1)
    lapl_eval = lambda wf_params, config: jnp.trace(hess(wf_params, config).reshape(
        config.shape[0]*config.shape[1], 
        config.shape[0]*config.shape[1]
    ))

    def local_energy(wf_params, config):
        """
        Local energy operator. Uses JAX autograd to obtain laplacian for KE.
        """

        n_electron = config.shape[0]
        acc = 0

        # Calculate kinetic energy
        acc += -0.5*(1/wf(wf_params, config))*lapl_eval(wf_params, config)

        # Calculate electron-electron energy
        rs = jnp.linalg.norm(config[None, :, :] - config[:, None, :], axis=-1)
        vee = jnp.triu(1./ rs, k=1)
        acc += jnp.sum(vee)

        # Calculate electron-nucleus energy, assume z=ne FOR NOW
        acc -= jnp.sum(n_electron / jnp.linalg.norm(config, axis=-1))

        return acc

    return local_energy

def gen_grad_operators(wf):
    log_grad = jax.grad(lambda wf_params, config: jnp.log(jnp.abs(wf(wf_params, config))))
    def grad_op(wf_params, config, local_energy_op, local_energy_exp):
        """Gradient(Energy) operator"""
        lg = log_grad(wf_params, config)
        log_grad_flat, _ = jax.flatten_util.ravel_pytree(lg)

        return jnp.multiply((local_energy_exp - local_energy_op(wf_params, config)), log_grad_flat)

    def lg_op(wf_params, config):
        """Gradient(Log(Psi)) operator"""
        lg = log_grad(wf_params, config)
        log_grad_flat, _ = jax.flatten_util.ravel_pytree(lg)

        return log_grad_flat

    def ovp(wf_params, config, x, eps, log_grad_exp):
        """Overlap matrix - vector product operator"""
        lg = log_grad(wf_params, config)
        log_grad_flat, _ = jax.flatten_util.ravel_pytree(lg)

        return jnp.multiply(log_grad_flat - log_grad_exp, jnp.dot(log_grad_flat - log_grad_exp, x)) + eps*jax.ops.index_update(x, 0, 0.0)

    def get_rewrap(wf_params):
        """Helper function to re-wrap parameter pytree after NGD step"""
        _, f = jax.flatten_util.ravel_pytree(wf_params)
        return f
    
    return grad_op, lg_op, ovp, get_rewrap