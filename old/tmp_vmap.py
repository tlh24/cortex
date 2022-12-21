import jax
import math
import numpy as np
from numpy.random import randn
from numpy.random import rand
import torch
import matplotlib.pyplot as plt
import jax.numpy as jnp
import ipdb

import functools
import pdb


seed = 17013
key = jax.random.PRNGKey(seed)

key, subkey = jax.random.split(key)
pre = jax.random.randint(subkey, (2,), 2, 4)
key, subkey = jax.random.split(key)
post = jax.random.randint(subkey, (2, 3), 5, 10)

wts = jnp.zeros((2, 3, 4))

key, subkey = jax.random.split(key)
src = jax.random.randint(subkey, (2, 3, 4), 0, 2)

# weight update is a nonlinear function of the outer product between pre and post. 

# need to decompose this into two vmaps.

def getpre(p, src_):
	return p[src_]

def weightup(qpre, qpost):
	return qpre * qpost

q = jax.vmap(getpre, (None, 2), 2)(pre, src)
print('pre')
print(pre)
print('src')
print(src)
print('q')
print(q)
print(q.shape)
# above: addressing (gather) works!  get a 2 x 3 x 4 matrix.
# now need to combine this pre with a 2 x 3 post matrix. 
dw = jax.vmap(weightup, (2, None), 2)(q, post)
print('post')
print(post)
print('dw')
print(dw)
print(dw.shape)

# see if we can do this all at once -- 
dw2 = jax.vmap(weightup, (2, None), 2) \
		(jax.vmap(getpre, (None, 2), 2)(pre, src), post)
print('dw2')
print(dw2)

