import jax
import jax.numpy as jnp

from functools import partial

def hessian(f, x):
  _, hvp = jax.linearize(jax.grad(f), x)
  basis = jnp.eye(jnp.prod(jnp.array([*x.shape]))).reshape(-1, *x.shape)
  return jnp.stack([hvp(e) for e in basis]).reshape(x.shape + x.shape)

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

def gen_energy_gradient(wf):
    log_grad = lambda p, c: jax.grad(lambda p, c: jnp.log(jnp.abs(wf(p, c))))(p, c)
    def energy_grad(p, c, local_energy_exp, local_energy_op):
        el = local_energy_op(p, c)
        log_grad_val = log_grad(p, c)
        out = jax.tree_util.tree_map(
            lambda x: jnp.multiply((el - local_energy_exp), x),
            log_grad_val
        )
        return out

    return energy_grad

def gen_sr_operators(wf):
    log_grad = lambda p, c: jax.grad(lambda p, c: jnp.log(jnp.abs(wf(p, c))))(p, c)
    def sr_op(p, c, local_energy_op, tau):
        lg = log_grad(p, c)
        log_grad_flat = jnp.concatenate((lg[0], *tuple(jnp.concatenate((w.flatten(), b.flatten())) for (w, b) in lg[1])))
        log_grad_flat = jnp.concatenate((jnp.array([1]), log_grad_flat))

        return jnp.multiply((1.0 - local_energy_op(p, c)*tau), log_grad_flat)

    def ovp(p, c, x):
        lg = log_grad(p, c)
        log_grad_flat = jnp.concatenate((lg[0], *tuple(jnp.concatenate((w.flatten(), b.flatten())) for (w, b) in lg[1])))
        log_grad_flat = jnp.concatenate((jnp.array([1]), log_grad_flat))

        return jnp.multiply(log_grad_flat, jnp.dot(log_grad_flat, x))

    def rewrap(p, layer_sizes):
        idx = 0
        p_wrapped = []
        for m, n in zip(layer_sizes[:-1], layer_sizes[1:]):
          p_wrapped.append(
              (p[idx:idx + m*n].reshape((n, m)), p[idx + m*n:idx + (m+1)*(n)])
          )
          idx += (m+1)*(n)

        return p_wrapped
    
    return sr_op, ovp, rewrap