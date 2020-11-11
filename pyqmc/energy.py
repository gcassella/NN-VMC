#import numpy as np
import jax.numpy as jnp
import pyqmc.eval_ecp as eval_ecp
from pyqmc.distance import RawDistance


def ee_energy(configs):
    ne = configs.shape[1]
    if ne == 1:
        return jnp.zeros(configs.shape[0])
    ee = jnp.zeros(configs.shape[0])
    ee, ij = RawDistance().dist_matrix(configs)
    ee = jnp.linalg.norm(ee, axis=2)
    return jnp.sum(1.0 / ee, axis=1)


def ei_energy(mol, configs):
    ei = 0.0
    for c, coord in zip(mol.atom_charges(), mol.atom_coords()):
        delta = configs - coord[jnp.newaxis, jnp.newaxis, :]
        deltar = jnp.sqrt(jnp.sum(delta ** 2, axis=2))
        ei += -c * jnp.sum(1.0 / deltar, axis=1)
    return ei


def ii_energy(mol):
    ei = 0.0
    d = RawDistance()
    rij, ij = d.dist_matrix(mol.atom_coords()[jnp.newaxis, :, :])
    if len(ij) == 0:
        return jnp.array([0.0])
    rij = jnp.linalg.norm(rij, axis=2)[0, :]
    iitot = 0
    c = mol.atom_charges()
    for (i, j), r in zip(ij, rij):
        iitot += c[i] * c[j] / r
    return iitot


def get_ecp(mol, configs, wf, threshold):
    return eval_ecp.ecp(mol, configs, wf, threshold)


def kinetic(configs, wf):
    nconf, nelec, ndim = configs.shape
    ke = jnp.zeros(nconf)
    for e in range(nelec):
        ke += -0.5 * jnp.real(wf["laplacian"](configs, e, configs[:, e]))
    return ke


def energy(mol, configs, wf, threshold):
    """Compute the local energy of a set of configurations.
    
    Args:
      mol: A pyscf-like 'Mole' object. nelec, atom_charges(), atom_coords(), and ._ecp are used.

      configs: a nconfiguration x nelectron x 3 numpy array
       
      wf: A Wavefunction-like object. Functions used include recompute(), lapacian(), and testvalue()

    Returns: 
      a dictionary with energy components ke, ee, ei, and total
      """
    ee = ee_energy(configs)
    ei = ei_energy(mol, configs)
    ecp_val = get_ecp(mol, configs, wf, threshold)
    ii = ii_energy(mol)
    ke = kinetic(configs, wf)
    # print(ke,ee,ei,ii)
    return {
        "ke": ke,
        "ee": ee,
        "ei": ei,
        "ecp": ecp_val,
        "total": ke + ee + ei + ecp_val + ii,
    }
