import jax
import jax.numpy as jnp
import torch


def multi_test(a, b, c):
	# all have the same leading dimension, but each has different trailing dim.
	d = a**2
	e = b**2
	f = c**2
	return d, e, f


seed = 17013
key = jax.random.PRNGKey(seed)

key, subkey = jax.random.split(key)
a = jax.random.normal(subkey, (2, 3, 4))

key, subkey = jax.random.split(key)
b = jax.random.normal(subkey, (2, 4, 5))

key, subkey = jax.random.split(key)
c = jax.random.normal(subkey, (2, 5, 6))

d, e, f = jax.vmap(multi_test, (0,0,0), 0)(a, b, c)

print('a', a.shape, a)
print('b', b.shape, b)
print('c', c.shape, c)
print('d', d.shape, d)
print('e', e.shape, e)
print('f', f.shape, f)

# ok!  returning multiple variables from a vmap works just fine,
# so long as they all have the same leading dimension!
