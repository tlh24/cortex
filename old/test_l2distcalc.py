import jax
import math
import numpy as np
from numpy.random import randn
from numpy.random import rand
import torch
import matplotlib.pyplot as plt
import jax.numpy as jnp
import ipdb
import time
import functools
import pdb
# import cma
import traceback
plt.rcParams['figure.dpi'] = 180

POP = 3

def behavioral_novelty(act_data, siz):
	# act_data = jnp.reshape(act_data, (POP, 72*NEURONS))
	# from a given dataset, want to maximize the minimum distance between
	# any behavior and all other behaviors.
	@jax.jit
	def l2_dist2(m, ai, bi):
		return jax.lax.fori_loop(0, siz, \
			lambda i,c: c+(m[ai,i]-m[bi,i])**2, 0.0)

	def l2_dist_cond(m, ai, bi):
		return jax.lax.cond(bi > ai, l2_dist2, lambda a,b,c: 0.0, m,ai,bi)
	def l2_dist3(m, ind, vi):
		return jax.vmap(l2_dist_cond, (None, 0, None), 0)(m, ind, vi)

	ind1 = jnp.arange(0,siz)
	ind2 = jnp.arange(0,siz)
	dist = jax.vmap(l2_dist3, (None, None, 0), 0)(act_data, ind1, ind2)
	# this is lower-triangular,
	# e.g [1,0] is the distance from 1 to 0 (which is symmetric, hence
	# copy it for minimum calculation..)
	dist = dist + jnp.transpose(dist) + jnp.eye(siz) * 1e6
	return dist

seed = 17016
key = jax.random.PRNGKey(seed)

key, subkey = jax.random.split(key)
rand_numbz = jax.random.uniform(subkey, (POP, 4))

print(rand_numbz)
b = behavioral_novelty(rand_numbz, POP)
print(b)
