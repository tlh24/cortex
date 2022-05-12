# mess around with nonlniear least squares.

import math
import numpy as np
import jax
import jax.numpy as jnp
import pdb
import torch
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 180

seed = 17016
key = jax.random.PRNGKey(seed)

def f(x):
	o = jnp.array([x[0], x[1], x[2], x[0]*x[1], x[0]*x[2]])
	return o

g = jax.jacfwd(f)

for j in range(1):
	key, subkey = jax.random.split(key)
	xt = jax.random.uniform(subkey, (3,)) * 2.0
	key, subkey = jax.random.split(key)
	y = f(xt) + jax.random.uniform(subkey, (5,)) * 0.0001

	x = jnp.ones((3,))
	for k in range(10):
		z = f(x)
		err = y - z
		w = g(x)
		u = err @ w
		x = x + 0.1*u # idk??!
		#print(k,'   error',err)
		#print(k,'   x', x)
	print('x', x)
	print('xt', xt)
	print('total', jnp.sum(x-xt))
# pdb.set_trace()

x = torch.arange(-2.0, 2.0, 0.01)
x2 = torch.flip(x, (0,))
fig, axs = plt.subplots(1, 3, figsize=(18, 8))
z = torch.outer(x2, x)
o = torch.ones(x.shape)
y = torch.outer(x2, o) + torch.outer(o, x)
zy = z - y
im = axs[0].imshow(z.numpy())
plt.colorbar(im, ax=axs[0])
axs[0].set_title('product')
im = axs[1].imshow(y.numpy())
plt.colorbar(im, ax=axs[1])
axs[1].set_title('sum')
im = axs[2].imshow(zy.numpy())
plt.colorbar(im, ax=axs[2])
axs[2].set_title('difference')

plt.show()
