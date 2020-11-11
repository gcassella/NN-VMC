import os
#import numpy as np
import pyqmc.mc as mc
import sys
import h5py

import jax
import jax.numpy as jnp

from functools import partial

def limdrift(g, tau, acyrus=0.25):
    """
    Use Cyrus Umrigar's algorithm to limit the drift near nodes.

    Args:
      g: a [nconf,ndim] vector

      tau: time step
      
      acyrus: the maximum magnitude

    Returns: 
      The vector with the cut off applied and multiplied by tau.
    """
    tot = jnp.linalg.norm(g, axis=1) * acyrus
    mask = tot > 1e-8
    taueff = jnp.ones(tot.shape) * tau
    taueff = jnp.where(
      mask, 
      (jnp.sqrt(1 + 2 * tau * tot) - 1) / tot,
      taueff
    )
    return g * taueff[:, jnp.newaxis]


def limdrift_cutoff(g, tau, cutoff=1):
    """
    Limit a vector to have a maximum magnitude of cutoff while maintaining direction

    Args:
      g: a [nconf,ndim] vector
      
      cutoff: the maximum magnitude

    Returns: 
      The vector with the cut off applied and multiplied by tau.
    """
    return mc.limdrift(g, cutoff) * tau

@partial(jax.jit, static_argnums=(1,9,10,11))
def dmc_step(
    key,
    wf,
    configs,
    df,
    weights,
    tstep,
    branchcut_start,
    branchcut_stop,
    eref,
    accumulators,
    ekey,
    drift_limiter,
):
    nconfig, nelec = configs.shape[0:2]
    #wf.recompute(configs)

    eloc = accumulators[ekey[0]](configs, wf)[ekey[1]].real
    acc = jnp.zeros(nelec)
    for e in range(nelec):
        # Propose move
        grad = drift_limiter(jnp.real(wf["gradient"](configs, e, configs[:, e]).T), tstep)
        key, subkey = jax.random.split(key)
        gauss = jax.random.normal(subkey, (nconfig, 3))*jnp.sqrt(tstep)
        newepos = configs[:, e, :] + gauss + grad
        #newepos = configs.make_irreducible(e, eposnew)
        # Compute reverse move
        new_grad = drift_limiter(jnp.real(wf["gradient"](configs, e, newepos).T), tstep)
        forward = jnp.sum(gauss ** 2, axis=1)
        backward = jnp.sum((gauss + grad + new_grad) ** 2, axis=1)
        # forward = np.sum((configs[:, e, :] + grad - eposnew) ** 2, axis=1)
        # backward = np.sum((eposnew + new_grad - configs[:, e, :]) ** 2, axis=1)
        t_prob = jnp.exp(1 / (2 * tstep) * (forward - backward))
        # Acceptance -- fixed-node: reject if wf changes sign
        wfratio = wf["testvalue"](configs, e, newepos)
        ratio = jnp.abs(wfratio) ** 2 * t_prob
        if not wf["iscomplex"]:
            ratio *= jnp.sign(wfratio)
        key, subkey = jax.random.split(key)
        accept = ratio > jax.random.uniform(subkey, (nconfig,))
        # Update wave function
        proposed = jax.ops.index_update(
            configs,
            jax.ops.index[:, e, :],
            newepos
        )
        configs = jnp.where(accept[:, jnp.newaxis, jnp.newaxis], proposed, configs)
        #wf.updateinternals(e, newepos, mask=accept)
        acc = jax.ops.index_update(
            acc,
            e,
            jnp.mean(accept)
        )
    # weights
    energydat = accumulators[ekey[0]](configs, wf)
    elocnew = energydat[ekey[1]].real
    tdamp = limit_timestep(
        weights, elocnew, eloc, eref, branchcut_start, branchcut_stop
    )
    wmult = jnp.exp(-tstep * 0.5 * tdamp * (eloc + elocnew - 2 * eref))
    wmult = jnp.where(wmult > 2.0, 2.0, wmult)
    weights *= wmult
    wavg = jnp.mean(weights)
    avg = {}
    for k, accumulator in accumulators.items():
        dat = accumulator(configs, wf) if k != ekey[0] else energydat
        for m, res in dat.items():
            avg[k + m] = jnp.einsum("...i,i...->...", weights, res) / (
                nconfig * wavg
            )
    avg["weight"] = wavg
    avg["acceptance"] = jnp.mean(acc)
    df.append(avg)

    return df, configs

def dmc_propagate(
    key,
    wf,
    configs,
    weights,
    tstep,
    branchcut_start,
    branchcut_stop,
    eref,
    nsteps=5,
    accumulators=None,
    ekey=("energy", "total"),
    drift_limiter=limdrift,
):
    """
    Propagate DMC without branching
    
    Args:
      wf: A Wave function-like class. recompute(), gradient(), and updateinternals() are used, as well as anything (such as laplacian() ) used by accumulators

      configs: Configs object, (nconfig, nelec, 3) - initial coordinates to start calculation.

      weights: (nconfig,) - initial weights to start calculation

      tstep: Time step for move proposals. Introduces time step error.

      nsteps: number of DMC steps to take

      accumulators: A dictionary of functor objects that take in (coords,wf) and return a dictionary of quantities to be averaged. np.mean(quantity,axis=0) should give the average over configurations. If none, a default energy accumulator will be used.

      ekey: tuple of strings; energy is needed for DMC weights. Access total energy by accumulators[ekey[0]](configs, wf)[ekey[1]

      drift_limiter: a function that takes a gradient and a cutoff and returns an adjusted gradient


    Returns: (df,coords,weights)
      df: A list of dictionaries nstep long that contains all results from the accumulators.

      coords: The final coordinates from this calculation.

      weights: The final weights from this calculation
      
    """
    assert accumulators is not None, "Need an energy accumulator for DMC"
    df = []
    for _ in range(nsteps):
        key, subkey = jax.random.split(key)
        df, configs = dmc_step(
            subkey,
            wf,
            configs,
            df,
            weights,
            tstep,
            branchcut_start,
            branchcut_stop,
            eref,
            accumulators,
            ekey,
            drift_limiter,
        )

    df_ret = {}
    weight = jnp.asarray([d["weight"] for d in df])
    avg_weight = weight / jnp.mean(weight)
    for k in df[0].keys():
        df_ret[k] = jnp.mean(jnp.array([d[k] * w for d, w in zip(df, avg_weight)]), axis=0)
    df_ret["weight"] = jnp.mean(weight)

    return df_ret, configs, weights

def limit_timestep(weights, elocnew, elocold, eref, start, stop):
    """
    Stabilizes weights by scaling down the effective tstep if the local energy is too far from eref.

    Args:
      weights: (nconfigs,) array
        walker weights
      elocnew: (nconfigs,) array
        current local energy of each walker
      elocold: (nconfigs,) array
        previous local energy of each walker
      eref: scalar
        reference energy that fixes normalization
      start: scalar
        number of sigmas to start damping tstep
      stop: scalar
        number of sigmas where tstep becomes zero
    
    Return:
      tdamp: scalar
        Damping factor to multiply timestep; always between 0 and 1. The damping factor is 
            1 if eref-eloc < branchcut_start*sigma, 
            0 if eref-eloc > branchcut_stop*sigma,  
            decreases linearly inbetween.
    """
    # JAX does not like this kind of stuff!
    #if start is None or stop is None:
    #    return 1
    #assert (
    #    stop > start
    #), "stabilize weights requires stop>start. Invalid stop={0}, start={1}".format(
    #    stop, start
    #)
    eloc = jnp.stack([elocnew, elocold])
    fbet = jnp.amax(eref - eloc, axis=0)
    return jnp.clip((1 - (fbet - start)) / (stop - start), 0, 1)


def branch(key, configs, weights):
    """
    Perform branching on a set of walkers  by stochastic reconfiguration

    Walkers are resampled with probability proportional to the weights, and the new weights are all set to be equal to the average weight.
    
    Args:
      configs: (nconfig,nelec,3) walker coordinates

      weights: (nconfig,) walker weights

    Returns:
      configs: resampled walker configurations

      weights: (nconfig,) all weights are equal to average weight
    """
    nconfig = configs.shape[0]
    wtot = jnp.sum(weights)
    probability = jnp.cumsum(weights / wtot)
    key, subkey = jax.random.split(key)
    base = jax.random.uniform(subkey)
    newinds = jnp.searchsorted(probability, (base + jnp.arange(nconfig) / nconfig) % 1.0)
    configs = configs[newinds]
    weights = jnp.ones((nconfig, ))*wtot/nconfig
    return configs, weights

def dmc_file(hdf_file, data, attr, configs, weights):
    import pyqmc.hdftools as hdftools

    if hdf_file is not None:
        with h5py.File(hdf_file, "a") as hdf:
            if "configs" not in hdf.keys():
                hdftools.setup_hdf(hdf, data, attr)
                hdf.create_dataset(
                  "configs",
                  configs.shape,
                  chunks=True,
                  maxshape=(None, *configs.shape[1:]),
                )
            if "weights" not in hdf.keys():
                hdf.create_dataset("weights", weights.shape)
            hdftools.append_hdf(hdf, data)
            hdf["configs"].resize(configs.shape)
            hdf["configs"][...] = configs
            hdf["weights"][:] = weights


def rundmc(
    key,
    wf,
    configs,
    weights=None,
    tstep=0.01,
    nsteps=1000,
    branchtime=5,
    stepoffset=0,
    branchcut_start=3,
    branchcut_stop=6,
    drift_limiter=limdrift,
    verbose=False,
    accumulators=None,
    ekey=("energy", "total"),
    propagate=dmc_propagate,
    feedback=1.0,
    hdf_file=None,
    client=None,
    npartitions=None,
    **kwargs,
):
    """
    Run DMC 
    
    Args:
      wf: A Wave function-like class. recompute(), gradient(), and updateinternals() are used, as well as anything (such as laplacian() ) used by accumulators

      configs: (nconfig, nelec, 3) - initial coordinates to start calculation. 

      weights: (nconfig,) - initial weights to start calculation, defaults to uniform.

      nsteps: number of DMC steps to take

      tstep: Time step for move proposals. Introduces time step error.

      branchtime: number of steps to take between branching

      accumulators: A dictionary of functor objects that take in (coords,wf) and return a dictionary of quantities to be averaged. np.mean(quantity,axis=0) should give the average over configurations. If none, a default energy accumulator will be used.

      ekey: tuple of strings; energy is needed for DMC weights. Access total energy by accumulators[ekey[0]](configs, wf)[ekey[1]

      verbose: Print out step information 

      drift_limiter: a function that takes a gradient and a cutoff and returns an adjusted gradient

      stepoffset: If continuing a run, what to start the step numbering at.

    Returns: (df,coords,weights)
      df: A list of dictionaries nstep long that contains all results from the accumulators.

      coords: The final coordinates from this calculation.

      weights: The final weights from this calculation
      
    """
    # Restart from HDF file
    if hdf_file is not None and os.path.isfile(hdf_file):
        with h5py.File(hdf_file, "r") as hdf:
            stepoffset = hdf["step"][-1] + 1
            configs.load_hdf(hdf)
            weights = jnp.array(hdf["weights"])
            eref = hdf["eref"][-1]
            esigma = hdf["esigma"][-1]
            if verbose:
                print("Restarted calculation")
    else:
        warmup = 2
        key, subkey = jax.random.split(key)
        df, configs = mc.vmc(
            subkey,
            wf,
            configs,
            accumulators=accumulators,
            client=client,
            npartitions=npartitions,
            verbose=verbose,
        )
        en = df[ekey[0] + ekey[1]][warmup:]
        eref = jnp.mean(en).real
        esigma = jnp.sqrt(jnp.var(en) * jnp.mean(df["nconfig"]))
        if verbose:
            print("eref start", eref, "esigma", esigma)

    nconfig = configs.shape[0]
    if weights is None:
        weights = jnp.ones(nconfig)

    npropagate = int(jnp.ceil(nsteps / branchtime))
    df = []
    for step in range(npropagate):
        key, subkey = jax.random.split(key)
        df_, configs, weights = dmc_propagate(
            subkey,
            wf,
            configs,
            weights,
            tstep,
            branchcut_start * esigma,
            branchcut_stop * esigma,
            eref=eref,
            nsteps=branchtime,
            accumulators=accumulators,
            ekey=ekey,
            drift_limiter=drift_limiter,
            **kwargs,
        )

        df_["eref"] = eref
        df_["step"] = step + stepoffset
        df_["esigma"] = esigma
        df_["tstep"] = tstep
        df_["weight_std"] = jnp.std(weights)
        df_["nsteps"] = branchtime

        dmc_file(hdf_file, df_, {}, configs, weights)
        # print(df_)
        df.append(df_)
        eref = df_[ekey[0] + ekey[1]] - feedback * jnp.log(jnp.mean(weights))
        key, subkey = jax.random.split(key)
        configs, weights = branch(subkey, configs, weights)
        if verbose:
            print(
                "energy",
                df_[ekey[0] + ekey[1]],
                "eref",
                df_["eref"],
                "sigma(w)",
                df_["weight_std"],
            )

    df_ret = {}
    for k in df[0].keys():
        df_ret[k] = jnp.asarray([d[k] for d in df])
    return df_ret, configs, weights
