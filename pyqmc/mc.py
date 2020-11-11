# This must be done BEFORE importing numpy or anything else.
# Therefore it must be in your main script.
import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
#import numpy as np
import jax.numpy as jnp
import jax
import h5py

from functools import partial

def initial_guess(key, mol, nconfig, r=1.0):
    """ Generate an initial guess by distributing electrons near atoms
    proportional to their charge.

    assign electrons to atoms based on atom charges
    assign the minimum number first, and assign the leftover ones randomly
    this algorithm chooses atoms *with replacement* to assign leftover electrons

    Args: 

     mol: A PySCF-like molecule object. Should have atom_charges(), atom_coords(), and nelec

     nconfig: How many configurations to generate.

     r: How far from the atoms to distribute the electrons

    Returns: 

     A numpy array with shape (nconfig,nelectrons,3) with the electrons randomly distributed near 
     the atoms.
    
    """

    epos = jnp.zeros((nconfig, jnp.sum(jnp.array([*mol.nelec])), 3))
    wts = mol.atom_charges()
    wts = wts / jnp.sum(wts)

    for s in [0, 1]:
        neach = jnp.array(
            jnp.floor(mol.nelec[s] * wts), dtype=int
        )  # integer number of elec on each atom
        nleft = (
            mol.nelec[s] * wts - neach
        )  # fraction of electron unassigned on each atom
        nassigned = jnp.sum(neach)  # number of electrons assigned
        totleft = int(mol.nelec[s] - nassigned)  # number of electrons not yet assigned
        ind0 = s * mol.nelec[0]
        epos = jax.ops.index_update(
            epos,
            jax.ops.index[:, ind0 : ind0 + nassigned, :],
            jnp.repeat(mol.atom_coords(), neach, axis=0)
        ) # assign core electrons
        if totleft > 0:
            bins = jnp.cumsum(nleft) / totleft
            key, subkey = jax.random.split(key)
            inds = jnp.argpartition(
                jax.random.uniform(subkey, (nconfig, len(wts))), totleft, axis=1
            )[:, :totleft]
            epos = jax.ops.index_update(
                epos,
                jax.ops.index[:, ind0 + nassigned : ind0 + mol.nelec[s], :],
                mol.atom_coords()[inds]
            )  # assign remaining electrons

    key, subkey = jax.random.split(key)
    epos = epos + r * jax.random.normal(subkey, epos.shape)  # random shifts from atom positions
    # epos = OpenConfigs(epos)
    return epos


def limdrift(g, cutoff=1):
    """
    Limit a vector to have a maximum magnitude of cutoff while maintaining direction

    Args:
      g: a [nconf,ndim] vector
      
      cutoff: the maximum magnitude

    Returns: 
      The vector with the cut off applied.
    """
    tot = jnp.linalg.norm(g, axis=1)
    mask = tot > cutoff
    g = jnp.where(mask[:, jnp.newaxis], cutoff * g / tot[:, jnp.newaxis], g)
    return g


def vmc_file(hdf_file, data, attr, configs):
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
            hdftools.append_hdf(hdf, data)
            hdf["configs"].resize(configs.shape)
            hdf["configs"][...] = configs

@partial(jax.jit, static_argnums=(1,4,5,6,))
def vmc_step(key, wf, configs, block_avg, tstep, nsteps, accumulators):
    acc = 0.0
    nconf, nelec, _ = configs.shape
    for e in range(nelec):
        # Propose move
        grad = limdrift(jnp.real(wf["gradient"](configs, e, configs[:, e]).T))
        key, subkey = jax.random.split(key)
        gauss = jax.random.normal(subkey, shape=(nconf, 3))*jnp.sqrt(tstep)
        newcoorde = configs[:, e, :] + gauss + grad * tstep
        # newcoorde = configs.make_irreducible(e, newcoorde)
        # Compute reverse move
        new_grad = limdrift(jnp.real(wf["gradient"](configs, e, newcoorde).T))
        forward = jnp.sum(gauss ** 2, axis=1)
        backward = jnp.sum((gauss + tstep * (grad + new_grad)) ** 2, axis=1)
        # Acceptance
        t_prob = jnp.exp(1 / (2 * tstep) * (forward - backward))
        ratio = jnp.multiply(wf["testvalue"](configs, e, newcoorde) ** 2, t_prob)
        key, subkey = jax.random.split(key)
        accept = ratio > jax.random.uniform(subkey, shape=(nconf,))
        # Update wave function
        proposed = jax.ops.index_update(
            configs,
            jax.ops.index[:, e, :],
            newcoorde
        )
        configs = jnp.where(accept[:, jnp.newaxis, jnp.newaxis], proposed, configs)
        
        acc += jnp.mean(accept) / nelec
        
    # Rolling average on step
    for k, accumulator in accumulators.items():
        dat = accumulator.avg(configs, wf)
        for m, res in dat.items():
            if k + m not in block_avg:
                block_avg[k + m] = res / nsteps
            else:
                block_avg[k + m] += res / nsteps

    block_avg["acceptance"] = acc

    return block_avg, configs

def vmc_worker(key, wf, configs, tstep, nsteps, accumulators):
    """
    Run VMC for nsteps.

    Return a dictionary of averages from each accumulator.  
    """
    block_avg = {}
    #wf.recompute(configs)

    for _ in range(nsteps):
        key, subkey = jax.random.split(key)
        block_avg, configs = vmc_step(subkey, wf, configs, block_avg, tstep, nsteps, accumulators)

    return block_avg, configs

def vmc(
    key,
    wf,
    configs,
    nblocks=10,
    nsteps_per_block=10,
    nsteps=None,
    tstep=0.5,
    accumulators=None,
    verbose=False,
    stepoffset=0,
    hdf_file=None,
    client=None,
    npartitions=None,
):
    """Run a Monte Carlo sample of a given wave function.

    Args:
      wf: A Wave function-like class. recompute(), gradient(), and updateinternals() are used, as well as 
      anything (such as laplacian() ) used by accumulators
      
      configs: Initial electron coordinates

      nblocks: Number of VMC blocks to run 

      nsteps_per_block: Number of steps to run per block

      nsteps: (Deprecated) Number of steps to run, maps to nblocks = 1, nsteps_per_block = nsteps

      tstep: Time step for move proposals. Only affects efficiency.

      accumulators: A dictionary of functor objects that take in (configs,wf) and return a dictionary of quantities to be averaged. np.mean(quantity,axis=0) should give the average over configurations. If None, then the coordinates will only be propagated with acceptance information.
      
      verbose: Print out step information 

      stepoffset: If continuing a run, what to start the step numbering at.
  
      hdf_file: Hdf_file to store vmc output.

      client: an object with submit() functions that return futures

      nworkers: the number of workers to submit at a time

    Returns: (df,configs)
       df: A list of dictionaries nstep long that contains all results from the accumulators. These are averaged across all walkers.

       configs: The final coordinates from this calculation.
       
    """
    if nsteps is not None:
        nblocks = nsteps
        nsteps_per_block = 1

    if accumulators is None:
        accumulators = {}
        if verbose:
            print("WARNING: running VMC with no accumulators")

    # Restart
    if hdf_file is not None:
        with h5py.File(hdf_file, "a") as hdf:
            if "configs" in hdf.keys():
                stepoffset = hdf["block"][-1] + 1
                configs = jnp.array(hdf["configs"])
                if verbose:
                    print("Restarting calculation from step", stepoffset)

    df = []

    for block in range(nblocks):
        key, subkey = jax.random.split(key)
        if verbose:
            print(f"-", end="", flush=True)
        block_avg, configs = vmc_worker(
            subkey, wf, configs, tstep, nsteps_per_block, accumulators
        )
        # Append blocks
        block_avg["block"] = stepoffset + block
        block_avg["nconfig"] = nsteps_per_block * configs.shape[0]
        vmc_file(hdf_file, block_avg, dict(tstep=tstep), configs)
        df.append(block_avg)
    if verbose:
        print("vmc done")

    df_return = {}
    for k in df[0].keys():
        df_return[k] = jnp.asarray([d[k] for d in df])
    return df_return, configs
