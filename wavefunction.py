import jax.numpy as jnp
import jax

from jax import random

# A helper function to randomly initialize weights and biases
# for a dense neural network layer
def random_layer_params(m, n, key, scale=1):
  w_key, b_key = random.split(key)
  return (scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,)))

# Initialize all layers for a fully-connected neural network with sizes "sizes"
def init_network_params(sizes, key):
  keys = random.split(key, len(sizes))
  return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

def tanh(x):
    return jnp.tanh(x)

def predict(p, c):
  # per-example predictions
  r = jnp.linalg.norm(c, axis=1)
  r1 = r[0]
  r2 = r[1]
  u = jnp.linalg.norm(jnp.subtract(c[1], c[0]))

  activations = jnp.array([r1, r2, u])
  for w, b in p[:-1]:
    outputs = jnp.dot(w, activations) + b
    activations = tanh(outputs)
  
  final_w, final_b = p[-1]
  outputs = jnp.dot(final_w, activations) + final_b
  return outputs[0]
 
def nn_hylleraas(p, c):
    r = jnp.linalg.norm(c, axis=-1)
    r1 = r[0]
    r2 = r[1]

    s = r1 + r2
    t = r1 - r2
    u = jnp.linalg.norm(jnp.subtract(c[1], c[0]))
    return jnp.exp(-2*s)*(1 + 0.5*u*jnp.exp(-u))*predict(p, c)