import jax.numpy as jnp
import jax

from jax import random

# A helper function to randomly initialize weights and biases
# for a dense neural network layer
def random_layer_params(m, n, key, scale=1):
  w_key, b_key = random.split(key)
  return (scale * random.uniform(w_key, (n, m), minval=-1, maxval=1), jnp.zeros((n,)))

# Initialize all layers for a fully-connected neural network with sizes "sizes"
def init_network_params(sizes, key):
  keys = random.split(key, len(sizes))
  return [random_layer_params(m, n, k, scale=1/jnp.sqrt(m)) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

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
    return jnp.exp(-2*s)*(1 + 0.5*u*jnp.exp(-p[0][0]*u))*predict(p[1], c)

def hylleraas(p, c):
    r = jnp.linalg.norm(c, axis=-1)
    r1 = r[0]
    r2 = r[1]

    s = r1 + r2
    t = r1 - r2
    u = jnp.linalg.norm(jnp.subtract(c[1], c[0]))
    return jnp.exp(-2*s)*(1 + 0.5*u*jnp.exp(-p[0][0]*u))*(1+p[0][1]*s*u + p[0][2]*t**2.0 + p[0][3]*u**2.0)