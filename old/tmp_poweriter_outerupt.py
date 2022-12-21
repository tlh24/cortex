import math
import jax
import jax.numpy as jnp
import jax.random.split as rsplit
import pdb
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 170

scaler = 1.0

@jax.jit
def make_stim(s, x, y):
	# make a simple on-off vector pattern based on bits s, x, y
	# see notes for a depiction.
	out = jnp.zeros((3,3))
	if s:
		# shape 0 is L rotated 180 deg
		out = out.at[x, y].set(1)
		out = out.at[x, y+1].set(1)
		out = out.at[x+1, y].set(1)
	else:
		out = out.at[x, y].set(1)
		out = out.at[x+1, y+1].set(1)
	return out

@jax.jit
def outerupt2(v):
	# outer product upper-triangle, terms with two or one factor.
	nin = v.shape[0]
	nout = int( (nin+1) * nin * 0.5 + nin )
	r = jnp.zeros(nout)
	e = 0
	for i in range(nin):
		for j in range(i, nin):
			if not(i == j):
				r = r.at[e].set( v[i] * v[j] * scaler )
				e = e + 1
	for i in range(nin):
		r = r.at[e].set( v[i] )
		e = e + 1
	return r

@jax.jit
def outerupt3(v):
	# outer product upper-triangle, but for terms with three, two, and one factor
	nin = v.shape[0]
	nout = int( nin * (nin+1) * (nin+2) / 6.0 + \
		nin * (nin+1) * 0.5 + nin )
	# above took me a surprisingly long time to figure out... sigh
	r = jnp.zeros((nout,))
	e = 0
	for i in range(nin):
		for j in range(i, nin):
			for k in range(j, nin):
				r = r.at[e].set( v[i] * v[j] * v[k] * scaler * scaler )
				e = e + 1
	for i in range(nin):
		for j in range(i, nin):
			r = r.at[e].set( v[i] * v[j] * scaler )
			e = e + 1
	for i in range(nin):
		r = r.at[e].set( v[i] )
		e = e + 1
	return r

@jax.jitdef make_stim(s, x, y):
	# make a simple on-off vector pattern based on bits s, x, y
	# see notes for a depiction.
	out = zeros(3,3)
	if s:
		# shape 0 is L rotated 180 deg
		out[x, y] = 1
		out[x, y+1] = 1
		out[x+1, y] = 1
	else:
		out[x, y] = 1
		out[x+1, y+1] = 1
	return out
def outerupt2i(v):
	vv = jnp.concatenate((v, (1.0 - v)), 0)
	vv = jnp.clip(vv, 0.0, 2.0)
	return outerupt2(vv)

@jax.jit
def outerupt3i(v):
	vv = jnp.concatenate((v, (1.0 - v)), 0)
	vv = jnp.clip(vv, 0.0, 2.0)
	return outerupt3(vv)

outerupt2_jc = jax.jacfwd(outerupt2i)
outerupt3_jc = jax.jacfwd(outerupt3i)

# print(outerupt3j(y))
#x = jnp.zeros((3,))
#a = outerupt3i(x)
#x = jnp.ones((3,))
#b = outerupt3i(x)
#plt.plot(a, 'o-')
#plt.plot(b, 'x-')
#plt.show()
#print(jnp.sum(a))
#print(jnp.sum(b)) # check check

# now see if we can invert, if gradual.
seed = 17016
key = jax.random.PRNGKey(seed)

for i in range(10):
	key, subkey = jax.random.split(key)
	xt = jax.random.uniform(subkey, (3,)) * 1.0
	y = outerupt2i(xt)

	x = jnp.ones((3,)) * 0.5
	for k in range(10):
		z = outerupt2i(x)
		err = y - z
		w = outerupt2_jc(x)
		u = err @ w
		x = x + 0.1*u
		#print(k,'   error',err)
		#print(k,'   x', x)

	print('x', x)
	print('xt', xt)
	print('total', jnp.sum(x-xt))

# two terms work pretty well; three products does not work, alas.
# ( at least not with only three latent variables)
# but that's ok we can build dit up out of multiple layers
# (just have to think through it well!)

animate = True
plot_rows = 4
plot_cols = 3
figsize = (20, 10)
if animate:
	plt.ion()
	fig, axs = plt.subplots(plot_rows, plot_cols, figsize=figsize)
initialized = False
im = [ [0]*plot_cols for i in range(plot_rows)]
cbar = [ [0]*plot_cols for i in range(plot_rows)]

def plot_tensor(r, c, v, name, lo, hi):
	global initialized
	if not initialized:
		# seed with random data so we get the range right
		cmap_name = 'PuRd' # purple-red
		if lo == -1*hi:
			cmap_name = 'seismic'
		data = np.linspace(lo, hi, v.shape[0] * v.shape[1])
		data = np.reshape(data, (v.shape[0], v.shape[1]))
		im[r][c] = axs[r,c].imshow(data, cmap = cmap_name)
		cbar[r][c] = plt.colorbar(im[r][c], ax=axs[r,c])
	else:
		im[r][c].set_data(v)
		# cbar[r][c].update_normal(im[r][c]) # probably does nothing
		axs[r,c].set_title(name)

def hebb_update(w_, inp, outp, outpavg, outplt, lr):
	# see older source for the history and reasoning behind this.
	dw = jnp.outer(outp, inp)
	dw2 = jnp.clip(dw, -1.0, 1.0)
	dw = jnp.pow(dw2 * 3.0, 3.0) # cube is essential!
	ltp = jnp.clip(dw, 0.0, 2.0) # make ltp / ltd asymmetrc
	ltd = jnp.clip(dw, -2.0, 0.0) # to keep weights from going to zero
	lr = lr / math.pow(inp.size(0), 0.35)
	dw = ltp*lr + ltd*lr
	dw = ltp*lr + ltd*1.3*lr*(0.9+\
		jnp.outer(outpavg*1.5, jnp.ones((inp.size(0),) )))

	w_ = w_ + dw

	scale = jnp.clip(outplt, 0.0, 1e6)
	scale = jnp.exp(scale * -0.005)
	dws = jnp.outer(scale, jnp.ones((inp.size(0),)))
	w_ = w_ * dws
	w_ = jnp.clip(w_, -0.00025, 1.25)
	return w_, dw

M = 54
P = 4
Pp = P*2
Q = int(  Pp * (Pp+1) * (Pp+2) / 6.0 + 0.5 * Pp * (Pp+1) + Pp ) # 164


w_f = jnp.zeros(Q, M)
w_b = jnp.zeros(M, Q)
l2ra = jnp.zeros((Q,))
l2lt = jnp.zeros((Q,))

N = 10

for i in range(int(N)):
	pbd.set_trace()
	q = i % 8
	st, sx, sy = (int(q/4)%2, q%2, int(q/2)%2)
	l1e = make_stim(st, sx, sy)
	l1e = reshape(l1e, (9,))
	l1o = outerupt2(l1e)

	key,sk = rsplit(key)
	noiz = jax.random.normal(sk, (Q,)) * 0.05 # this might not be required?
	l2x = w_f @ l1o + noiz
	# need to convert this to compressed rep.
	l2c = jnp.ones((P,)) * 0.5 # can use a better seed later
	for k in range(10):
		l2r = outerupt2i(l2c)
		err = l2x - l2r
		w = outerupt2_jc(l2c)
		u = err @ w
		l2c = l2c + 0.1*u
		#print(k,'   error',err)
		#print(k,'   x', x)

	l2r = outerupt2i(l2c)
	l2ra = 0.95 * l2ra + 0.05 * l2r
	l2lt = 0.99*l2lt + 0.05*(l2r - 0.5) # leaky integral.

	l1x = w_b @ l2r
	# now need to do the same nonlinear least squares,
	# only we know what the state is.
	l1c = l1e # Seed value. I hope this copies it.
	for k in range(10):
		l1r = outerupt2i(l1c)
		err = l1x - l1r
		w = outerupt2_jc(l1c)
		u = err @ w
		l1c = l1c + 0.1*u

	err = l1e - l1c # scalar error
	w = outerupt2i_jc(l1e)
	l1r = w @ err

	# now, should be able to perform nonlinear hebbian learning on l1r and l2r & gradually minimize reconstruction error.

	w_f, dwf = hebb_update(w_f, l1r, l2r, l2ra, l2lt, lr*1.5)

	w_b, dwb = hebb_update(w_b, l2r, l1r, \
		0.5*jnp.ones((M,)), jnp.zeros((M,)), lr*1.2)

	if True:
		plot_tensor(0, 0, l1e, 'l1e input', 0.0, 1.0)
		plot_tensor(1, 0, l1c, 'l1c compress', 0.0, 1.0)
		plot_tensor(2, 0, l1o, 'l1o expanded', 0.0, 1.0)
		plot_tensor(3, 0, l1r, 'l1r reconstruct', -1.0, 1.0)
		plot_tensor(4, 0, l1x, 'l1x approx', -1.0, 1.0)

		plot_tensor(0, 1, l2x, 'l2x approx' , 0.0, 1.0)
		plot_tensor(1, 1, l2c, 'l2c compress' , 0.0, 1.0)
		plot_tensor(2, 1, l2r, 'l2r reconstruct' , 0.0, 1.0)
		plot_tensor(3, 1, l2ra,'l2ra recon avg' , 0.0, 1.0)
		plot_tensor(4, 1, err, 'err' , 0.0, 1.0)

		plot_tensor(0, 2, w_f, 'w_f', -1.5, 1.5)
		plot_tensor(1, 2, dwf, 'dwf', -0.05*lr, 0.05*lr)
		plot_tensor(2, 2, w_b, 'w_b', -0.1, 0.1)
		plot_tensor(3, 2, dwb, 'dwb', -0.25*lr, 0.25*lr)

		fig.tight_layout()
		fig.canvas.draw()
		fig.canvas.flush_events()
		time.sleep(0.02)
		initialized = True
		# print(q, st, sx, sy, anneal, boost)
		print(anneal, boost)
		if i == N - 1:
			time.sleep(10)
