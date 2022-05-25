import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".50"

import math
import jax
import jax.numpy as jnp
from jax.random import split as rsplit
import numpy as np
import torch
import torchvision
import pdb
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 180
import time

#from jax.config import config
#config.update("jax_debug_nans", True)

torch_device = 0
print("torch cuda devices", torch.cuda.device_count())
print("torch device", torch.cuda.get_device_name(torch_device))
#device = torch.device("cpu")
#torch.set_default_tensor_type('torch.FloatTensor')

scaler = 1.0

def to_jax(torchvar):
	return jnp.asarray(torchvar.cpu().numpy())

# can't jit this one .. just memorize later.
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

# really need to represent this as a matrix or tensor product -- not too big though!
# def outerupt2v(m):
	# don't need all the products ... only a local 2d region.
	# should be possible with a tensor product ..
	# if m is a 2d matrix, want something like m x A x m --> out
	# if m is 28 x 28
	# then m ⊗ m is 784 x 784
	# aux matrix b is 28 x 28 with off-diagonal terms
	# m ⊗ a is still 784 x 784, but it won't be full
	# I guess we could do this with an indexing vector and vmap .. ?
	# probably the easiest.
	# can do all sorts of funky things with this indexing!
	# including interesting downsampling.

def outerupt_indx2(siz, roi, do_parity):
	# this is for 2D images!!
	e = 0
	r1 = []
	c1 = []
	r2 = []
	c2 = []
	for i in range(siz):
		for j in range(siz):
			parity = (i + j) % 2
			for ii in range(siz):
				for jj in range(siz):
					if (ii > i and jj >= j) or (ii >= i and jj > j):
						d = (ii - i)*(ii - i) + (jj -j)*(jj - j)
						d = math.sqrt(d)
						parity2 = (ii + jj) % 2
						if d < roi and (parity != parity2 or (not do_parity)):
							r1.append(i)
							c1.append(j)
							r2.append(ii)
							c2.append(jj)
	r1 = jnp.array(r1, dtype=int)
	c1 = jnp.array(c1, dtype=int)
	r2 = jnp.array(r2, dtype=int)
	c2 = jnp.array(c2, dtype=int)
	return (r1, c1, r2, c2)

def outerupt_indx2_flat(siz, roi, do_parity):
	(r1, c1, r2, c2) = outerupt_indx2(siz, roi, do_parity)
	i1 = r1 * siz + c1
	i2 = r2 * siz + c2
	return (i1, i2)

def outerupt_indx(siz, roi):
	# this is for vectors!
	i1 = []
	i2 = []
	for i in range(siz):
		for j in range(siz):
			if j > i and j-i <= roi:
				i1.append(i)
				i2.append(j)
	i1 = jnp.array(i1)
	i2 = jnp.array(i2)
	return (i1, i2)

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

@jax.jit
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

seed = 17016
key = jax.random.PRNGKey(seed)

# two terms work pretty well; three products does not work, alas.
# ( at least not with only three latent variables)
# but that's ok we can build dit up out of multiple layers
# (just have to think through it well!)


@jax.jit
def hebb_update(w_, inp, outp, outpavg, outplt, lr):
	# see older source for the history and reasoning behind this.
	d = inp.shape[0]
	dw = jnp.outer(outp, inp)
	dw2 = jnp.clip(dw, -1.0, 1.0)
	dw = (dw2 ** 3.0) * 3.0 # cube is essential!
	ltp = jnp.clip(dw, 0.0, 2.0) # make ltp / ltd asymmetrc
	ltd = jnp.clip(dw, -2.0, 0.0) # to keep weights from going to zero
	lr = lr / math.pow(d, 0.35)
	#dw = ltp*lr + ltd*1.5*lr*(0.9+\
		#jnp.outer(outpavg*1.5, jnp.ones((d,) )))
	dw = ltp*lr + ltd*1.5*lr

	w_ = w_ + dw

	scale = jnp.clip(outplt, 0.0, 1e6)
	scale = jnp.exp(scale * -0.005)
	dws = jnp.outer(scale, jnp.ones((d,)))
	w_ = w_ * dws
	w_ = jnp.clip(w_, -0.25, 1.25)
	return w_, dw


stim = jnp.zeros((8, 9))
for q in range(8):
	st, sx, sy = (int(q/4)%2, q%2, int(q/2)%2)
	v = make_stim(st, sx, sy)
	v = jnp.reshape(v, (9,))
	stim = stim.at[q].set(v)

batch_size = 60000
train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor()
                             ])),
  batch_size=batch_size, shuffle=True, pin_memory=True)
test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('files/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor()
                             ])),
  batch_size=batch_size, shuffle=True, pin_memory=True)


mnist = enumerate(train_loader)
# I guess convert all of them at the same time.. ?
batch_idx, (indata, target) = next(mnist)
#indata_jax = to_jax(indata)
#indata_rsz = jax.image.resize(indata_jax, (10000, 1, 16, 16), 'bilinear')
#indata1 = indata
#indata1[:, 0, 0, 0] = 1.0 # so the outer product works properly.
indata_jax = to_jax(indata)
indata_jax = jnp.reshape(indata_jax, (60000, 784))



#d = jnp.arange(5.)
#d = jnp.tensordot(jnp.ones((10,)), d, 0)
#e = jnp.arange(5.)
#e = jnp.flip(e)
#e = jnp.tensordot(jnp.ones((10,)), e, 0)
#f=  jnp.tensordot(d, e, 0) # this doesn't work.. dims (10, 5, 10, 5)
## want dims (10, 5, 5)
#d = jnp.reshape(d, (10, 1, 5)) # aka to jnp.expand_dims(d, 1)
#e = jnp.reshape(e, (10, 1, 5))
#f = jax.vmap(lambda a,b: a*b, (0, 0))(d, e)
## this does work; has the proper output dimensions.
## but is highly redundant, only need the upper triangle.
#indr, indc = jnp.triu_indices(5)
#h = f[:, indr, indc]
## note: need to add in a ones column to get the linear terms.

#d = jnp.reshape(indata_rsz, (10000, 256))
#d = jnp.reshape(d, (10000, 1, 256))
#e = jnp.reshape(indata_rsz, (10000, 256))
#e = jnp.reshape(e, (10000, 256, 1))
#f = jax.vmap(lambda a,b: a*b, (0, 0))(d, e)
#indr, indc = jnp.triu_indices(256)
#h = f[:, indr, indc]

#hy = jnp.concatenate((h, jnp.reshape(indata, (10000, 784))), 1)

# try a different more efficient way of doing it?
#siz = 5
#(r1, c1, r2, c2) = outerupt_indx(siz, 1)
#def index_mul(m, r1, c1, r2, c2):
	#return m[r1,c1] * m[r2,c2]
#m = jnp.arange(25.).reshape((5,5)) + 1
#q = jax.vmap(index_mul,(None, 0, 0, 0, 0), 0)(m, r1,c1,r2,c2)
## looks okay.
## but to get a jacobian matrix, need input and output to be vectors.

#i1 = r1 * siz + c1
#i2 = r2 * siz + c2
#def index_mul2(v, i1, i2):
	#return v[i1] * v[i2]
#m2 = jnp.reshape(m, (siz*siz, ))
#def outerupt_t(m2):
	#q2 = jax.vmap(index_mul2, (None, 0,0), 0)(m2, i1, i2)
	#return q2

#q2 = outerupt(m2)
#outerupt_jc = jax.jacfwd(outerupt_t)

#q3 = outerupt_jc(jnp.ones((25,)))

# all that seems to work ok.
# time for the real thing.
siz = 28
roi = 5

P = 32
Pp = P*2

fi1, fi2 = outerupt_indx2_flat(siz, roi, True)

bi1, bi2 = outerupt_indx(P, 48)

def index_mul2(v, i1, i2):
	return v[i1] * v[i2]

def outerupt_f(m2):
	q2 = jax.vmap(index_mul2, (None, 0,0), 0)(m2, fi1, fi2)
	return jnp.concatenate((m2,q2), -1)

@jax.jit
def outerupt_b(m):
	# m2 = jnp.concatenate((m, (1-m)), 0)
	m2 = m
	q2 = jax.vmap(index_mul2, (None, 0,0), 0)(m2, bi1, bi2)
	return jnp.concatenate((m2,q2), -1)

outerupt_f_jc = jax.jacfwd(outerupt_f)
outerupt_b_jc = jax.jacfwd(outerupt_b)

@jax.jit
def nlls_f(des, est): # nonlinear least squares
	# des is the target expanded value,
	# est is the estimate of the compressed val
	for k in range(10):
		fwd = outerupt_f(est)
		err = des - fwd
		#print("nlls_f", k, jnp.sum(err))
		w = outerupt_f_jc(est)
		u = err @ w
		est = est + 0.1*u
		est = jnp.clip(est, 0.0, 1.0)
	return est

@jax.jit
def nlls_b(des, est): # nonlinear least squares
	for k in range(10):
		fwd = outerupt_b(est)
		err = des - fwd
		#print("nlls_b", k, jnp.sum(err))
		w = outerupt_b_jc(est)
		u = err @ w
		u = jnp.clip(u, -1.0, 1.0)
		est = est + 0.1*u
		# est = jnp.clip(est, 0.0, 1.0)
	return est

fs = outerupt_f_jc(jnp.ones((siz*siz,)))
bs = outerupt_b_jc(jnp.ones((P,)))

# M = fs.shape[0]
M = 784
Q = bs.shape[0]
lr = 0.0025

w_f = jnp.zeros((Q, M))
w_b = jnp.zeros((M, Q))
l2ra = jnp.zeros((Q,))
l2lt = jnp.zeros((Q,))
l2ca = jnp.zeros((P,))
l2ch = jnp.zeros((P,))

animate = True
plot_rows = 5
plot_cols = 3
figsize = (20, 11)
if animate:
	plt.ion()
	fig, axs = plt.subplots(plot_rows, plot_cols, figsize=figsize)
initialized = False
im = [ [0]*plot_cols for i in range(plot_rows)]
cbar = [ [0]*plot_cols for i in range(plot_rows)]

def plot_tensor(r, c, v, name, lo, hi):
	global initialized
	if len(v.shape) == 1:
		n = v.shape[0]
		if n == 3:
			v = jnp.concatenate((v, jnp.zeros((1,))), 0)
			v = jnp.reshape(v, (2,2))
		if n == 4:
			v = jnp.reshape(v, (2,2))
		elif n == 9:
			v = jnp.reshape(v, (3,3))
		elif n == 16:
			v = jnp.reshape(v, (4,4))
		elif n == 27:
			v = jnp.reshape(v, (3,9))
		elif n == 32:
			v = jnp.reshape(v, (4,8))
		elif n == 44:
			v = jnp.reshape(v, (4,11))
		elif n == 48:
			v = jnp.reshape(v, (4,12))
		elif n == 120:
			v = jnp.reshape(v, (10,12))
		elif n == 136:
			v = jnp.reshape(v, (8,17))
		elif n == 189:
			v = jnp.reshape(v, (7,27))
		elif n == 252:
			v = jnp.reshape(v, (9,28))
		elif n == 528:
			v = jnp.reshape(v, (16,33))
		elif n == 680:
			v = jnp.reshape(v, (17,40))
		elif n == 784:
			v = jnp.reshape(v, (28,28))
		elif n == 1024:
			v = jnp.reshape(v, (32,32))
		elif n == 7696:
			v = jnp.reshape(v, (4*13,4*37))
		elif n == 25948:
			v = jnp.reshape(v, (4*13,499))
	if len(v.shape) == 2:
		if v.shape[0] == 25948 and v.shape[1] == 252:
			v = jnp.reshape(v, (13*9*7, 16*499))
		elif v.shape[1] == 25948 and v.shape[0] == 252:
			v = jnp.reshape(v, (13*9*7, 16*499))
		elif v.shape[1] == 7696 and v.shape[0] == 252:
			v = jnp.reshape(v, (4*13*9, 4*37*28))
		elif v.shape[0] == 7696 and v.shape[1] == 252:
			v = jnp.reshape(v, (4*13*9, 4*37*28))
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

N = indata.shape[0]
for ii in range(1000):
	for i in range(N):
		l1e = indata_jax[i]
		l1o = l1e
		# l1o = outerupt_f(l1e)

		key,sk = rsplit(key)
		noiz = jax.random.normal(sk, (Q,)) * 0.08 # this might not be required?
		l2x = w_f @ l1o + noiz

		l2c = jnp.ones((P,)) * 0.5
			# can use a better seed later (more info etc)
		l2c = nlls_b(l2x, l2c)
		# need to do a bit of homeostasis here for better codes...
		mn = jnp.mean(l2c)
		if mn < 0.1:
			l2c = l2c * 1.0532
		if mn > 0.7:
			l2c = l2c / 1.0456

		l2r = jnp.clip(outerupt_b(l2c), 0.0, 1.0)
		l2ch = jnp.clip((l2c - 0.65)*10, 0.0, 1.0)
		l2rh = outerupt_b(l2ch) # binary mask

		# not sure if these running stats are required .. ?
		l2ca = 0.95 * l2ca + 0.05 * l2c
		l2ra = outerupt_b(l2ca)

		l1x = w_b @ l2r

		l1r = l1o - l1x
		#w = outerupt2_jc(l1e)
		#l1r = w @ err

		## need to boost!
		l2x = w_f @ l1o + 0.2 * l2r

		l2c = jnp.ones((P,)) * 0.5
			# can use a better seed later (more info etc)
		l2c = nlls_b(l2x, l2c)

		l2r = jnp.clip(outerupt_b(l2c), 0.0, 1.0)

		l1x = w_b @ l2r

		l1r = l1o - l1x

		# now, should be able to perform nonlinear hebbian learning on l1r and l2r & gradually minimize reconstruction error.
		if ii < 10:
			w_f, dwf = hebb_update(w_f, l1r, l2r, l2ra, l2rh, lr*1.7) # was 1.5

		w_b, dwb = hebb_update(w_b, l2r, l1r, \
			0.5*jnp.ones((M,)), jnp.zeros((M,)), lr*1.2) # was 1.2

		w_f = jnp.clip(w_f, -0.001, 0.5)

		if i%100 == 99:
			# now need to do the same nonlinear least squares,
			# only we know what the state is.
			#l1c = l1e # Seed value. goal is to not have to change it below..
			#l1c = nlls_f(l1x, l1c)

			plot_tensor(0, 0, l1e, 'l1e input', 0.0, 1.0)
			#plot_tensor(1, 0, l1c, 'l1c reconstruct', 0.0, 1.0)
			plot_tensor(2, 0, l1o, 'l1o expanded', 0.0, 1.0)
			plot_tensor(3, 0, l1r, 'l1r error', -1.0, 1.0)
			plot_tensor(4, 0, l1x, 'l1x approx', -1.0, 1.0)

			plot_tensor(0, 1, l2x, 'l2x approx' , 0.0, 1.0)
			plot_tensor(1, 1, l2c, 'l2c compress' , 0.0, 1.0)
			plot_tensor(2, 1, l2r, 'l2r recon' , 0.0, 1.0)
			plot_tensor(3, 1, l2ra,'l2ra recon avg' , 0.0, 1.0)
			plot_tensor(4, 1, l2rh, 'l2rh clip activity' , -1.0, 1.0)

			plot_tensor(0, 2, w_f, 'w_f', -0.005, 0.005)
			plot_tensor(1, 2, dwf, 'dwf', -0.1*lr, 0.1*lr)
			plot_tensor(2, 2, jnp.transpose(w_b), 'w_b', -0.05, 0.05)
			plot_tensor(3, 2, jnp.transpose(dwb), 'dwb', -0.25*lr, 0.25*lr)

			fig.tight_layout()
			fig.canvas.draw()
			fig.canvas.flush_events()
			time.sleep(0.02)
			initialized = True
			# print(q, st, sx, sy, anneal, boost)
			if i == N - 1:
				time.sleep(10)

			print(ii, i, jnp.sum(w_f), jnp.sum(w_b), jnp.sum(dwf), jnp.sum(dwb))
