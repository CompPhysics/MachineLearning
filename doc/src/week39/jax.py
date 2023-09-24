import jax.nuympy as jnp
from jax import grad, jit, vmap
from jax import random

def sum_logistic(x):
  return jnp.sum(1.0 / (1.0 + jnp.exp(-x)))

x_small = arange(3.)
derivative_fn = grad(sum_logistic)
print(derivative_fn(x_small))
