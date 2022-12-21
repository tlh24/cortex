import jax
import jax.numpy as jnp

def distance(x, siz):
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
	dist = jax.vmap(l2_dist3, (None, None, 0), 0)(x, ind1, ind2)

	dist = dist + jnp.transpose(dist)

seed = 17016
key = jax.random.PRNGKey(seed)
key, subkey = jax.random.split(key)
x = jax.random.uniform(subkey, (1638, 7000))
d = distance(x, 1638)
