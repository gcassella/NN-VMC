from wavefunction import nn_hylleraas, hylleraas
import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

import pickle

def onepar(p, c):
    r = jnp.linalg.norm(c, axis=1)
    return jnp.exp(-p[0]*(r[0] + r[1]))

class Wavefunction():
    def __init__(self, f, params):
        self.iscomplex = False
        self._value = 0

        self.f = f
        self.parameters = {"p" : params}

        self.log_pgrad = jax.grad(lambda p, c: jnp.log(jnp.abs(self.f(p, c))))
        self.log_grad = lambda p, c, e: jax.grad(lambda p, c: jnp.log(jnp.abs(self.f(p, c))), 1)(p, c)[e]
        self.hess = jax.jacfwd(jax.grad(self.f, 1), 1)
        self.second_derivs = lambda p, c: jnp.diag(self.hess(p, c).reshape(c.shape[0]*c.shape[1], c.shape[0]*c.shape[1]))
        self.batch_log_grad = jax.jit(jax.vmap(self.log_grad, in_axes=(None, 0, None)))
        self.batch_log_pgrads = jax.jit(jax.vmap(self.log_pgrad, in_axes=(None, 0)))
        self.batch_second_derivs = jax.jit(jax.vmap(self.second_derivs, in_axes=(None, 0)))
        self.batch_predict = jax.jit(jax.vmap(self.f, in_axes=(None, 0)))

    def recompute(self, configs):
        self.configs = configs.configs
        self._value = self.batch_predict(self.parameters["p"], self.configs)
        return np.array(jnp.log(jnp.abs(self._value)))

    def updateinternals(self, e, epos, mask=None):
        self.configs = jax.ops.index_update(self.configs, jax.ops.index[:, e, :], epos.configs)
        self._value = self.batch_predict(self.parameters["p"], self.configs)
        return np.array(jnp.log(jnp.abs(self._value)))

    def value(self):
        return np.array(jnp.log(jnp.abs(self._value)))

    def testvalue(self, e, epos, mask=None):
        new_configs = jax.ops.index_update(self.configs, jax.ops.index[:, e, :], epos.configs)

        return np.array(self.batch_predict(self.parameters["p"], new_configs) / self._value)
    
    def pgradient(self):
        log_pgrads = self.batch_log_pgrads(self.parameters["p"], self.configs)
        log_grad_flat, _ = ravel_pytree(log_pgrads)
        
        return {"pars": np.array(log_grad_flat)}

    def gradient(self, e, epos):
        new_configs = jax.ops.index_update(self.configs, jax.ops.index[:, e, :], epos.configs)
        return np.array(self.batch_log_grad(self.parameters["p"], new_configs, e).T)

    def laplacian(self, e, epos):
        new_configs = jax.ops.index_update(self.configs, jax.ops.index[:, e, :], epos.configs)
        second_derivs = self.batch_second_derivs(self.parameters["p"], new_configs)
        lapl = jnp.sum(second_derivs[:, e*3:(e+1)*3], axis=-1)
        return np.array(lapl / self.batch_predict(self.parameters["p"], new_configs))

if __name__ == "__main__":
    import pyscf
    import pyqmc

    #from dask.distributed import Client, LocalCluster
    #ncore = 4
    #cluster = LocalCluster(n_workers=ncore, threads_per_worker=1)
    #client = Client(cluster)
    client = None

    nn_hylleraas_p = pickle.load(open('good_nn_wf.par', 'rb'))
    #hylleraas_p = [[0.9998273, 0.2546163, 0.1520364, -0.04904639], None]
    #simple_p = [1.6875]
    
    wf = Wavefunction(nn_hylleraas, nn_hylleraas_p)
    nconfig = 1000
    
    mol = pyscf.gto.M(atom="He 0. 0. 0.")
 
    configs = pyqmc.initial_guess(mol, nconfig)

    T = 1000
    T1 = int(8 * T / 9)
    T2 = int(T / 9)

    taus = np.array([0.02, 0.08])
    nsteps = np.array([T1, T2])
    ELs = []
    EL_errs = []

    for tau, nstep in zip(taus, nsteps):
        df, _, _ = pyqmc.rundmc(
            wf,
            configs,
            nsteps=nstep,
            accumulators={"energy": pyqmc.EnergyAccumulator(mol)},
            tstep=tau,
            verbose=True,
            client=client,
            hdf_file="dmc_taulim_{}.hdf5".format(nstep)
        )

        ELs.append(np.mean(df['energytotal']))
        EL_errs.append(np.var(df['energytotal']))

    np.savetxt("dmc_taulim.log", np.column_stack((taus, ELs, EL_errs)))

