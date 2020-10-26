from collections import namedtuple
import jax.numpy as jnp
import jax

# We can't pass NamedTuples around as vmapped arrays
# in JAX, so instead we keep a dict which specifies
# the order of mcmcparams in an array
mcmc_pmap = {
    'step_size' : 0,
    'n_iter' : 1,
    'n_equi' : 2,
    'n_step' : 3,
    'accepted' : 4,
    'curr_prob' : 5
}

def init_mcmc(wf, step_size, n_equi, n_iter):

    p = jnp.array([step_size, n_iter, n_equi, 0, 0, 0])

    pdf = lambda p, c: 2*wf(p, c)

    def step_mcmc(key, wf_params, config, mcmc_params):
        key, subkey = jax.random.split(key)
        move_proposal = jax.random.normal(subkey, shape=(config.shape[1],))*mcmc_params[mcmc_pmap["step_size"]]
        proposal = jax.ops.index_add(config, 
                                     mcmc_params[mcmc_pmap["n_step"]].astype(jnp.uint32)%config.shape[0], 
                                     move_proposal)
        proposal_prob = pdf(wf_params, proposal)

        key, subkey = jax.random.split(key)
        uniform = jnp.log(jax.random.uniform(subkey))
        accept = uniform < proposal_prob - mcmc_params[mcmc_pmap["curr_prob"]]

        new_config = jnp.where(accept, proposal, config)
        mcmc_params_new = jnp.array([
            step_size,
            n_iter,
            n_equi,
            mcmc_params[mcmc_pmap["n_step"]]+1,
            mcmc_params[mcmc_pmap["accepted"]] + jnp.where(accept, 1, 0),
            jnp.where(accept, proposal_prob, mcmc_params[mcmc_pmap["curr_prob"]])])

        return new_config, mcmc_params_new, accept

    def run_mcmc(key, wf_params, initial_config, mcmc_params):
        def mh_update(i, state):
          key, config, mcmc_params_prev = state
          key, subkey = jax.random.split(key)
          new_config, mcmc_params_new, _ = step_mcmc(
              subkey,
              wf_params,
              config,
              mcmc_params_prev
          )
          return (key, new_config, mcmc_params_new)
 
        def mh_update_and_store(i, state):
          key, config, mcmc_params_prev, config_buffer, accept_buffer = state
          key, subkey = jax.random.split(key)
          new_config, mcmc_params_new, accept = step_mcmc(
              subkey,
              wf_params,
              config,
              mcmc_params_prev
          )
          accept_buffer = jax.ops.index_update(accept_buffer, i, accept)
          config_buffer = jax.ops.index_update(config_buffer, i, new_config)
          return (key, new_config, mcmc_params_new, config_buffer, accept_buffer)

        prob = pdf(wf_params, initial_config)
        mcmc_params_new = jnp.array([
            mcmc_params[mcmc_pmap["step_size"]],
            mcmc_params[mcmc_pmap["n_iter"]],
            mcmc_params[mcmc_pmap["n_equi"]],
            mcmc_params[mcmc_pmap["n_step"]],
            mcmc_params[mcmc_pmap["accepted"]],
            prob
        ])

        key, subkey = jax.random.split(key)
        key, config, mcmc_params_new = jax.lax.fori_loop(
            jnp.uint32(0), 
            (mcmc_params[mcmc_pmap["n_equi"]]).astype(jnp.uint32), 
            mh_update, 
            (subkey, initial_config, mcmc_params_new)
        )

        accept_buffer = jnp.zeros((n_iter,), dtype=bool)
        config_buffer = jnp.zeros((n_iter, *initial_config.shape))
        key, subkey = jax.random.split(key)
        key, config, mcmc_params_new, config_buffer, accept_buffer = jax.lax.fori_loop(
            jnp.uint32(0), 
            (mcmc_params[mcmc_pmap["n_iter"]]).astype(jnp.uint32), 
            mh_update_and_store, 
            (subkey, config, mcmc_params_new, config_buffer, accept_buffer)
        )

        return config_buffer, accept_buffer, mcmc_params_new

    return run_mcmc, p

if __name__ == '__main__':
    
    def simple_wf(p, c):
        r = jnp.linalg.norm(c, axis=1)
        r1 = r[0]
        r2 = r[1]
        return jnp.log(jnp.abs(jnp.exp(-p[0]*(r1 + r2))))
        
    key = jax.random.PRNGKey(0)

    wf_params = jnp.array([2.0])

    key, subkey = jax.random.split(key)
    initial_config = jax.random.normal(subkey, (2, 3))

    run_mcmc, mcmc_params = init_mcmc(simple_wf, 0.5, 1000, 1000)
    key, subkey = jax.random.split(key)
    configs, accepts, mcmc_params = run_mcmc(subkey, wf_params, initial_config, mcmc_params)
    print(configs)