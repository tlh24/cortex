import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".50"

import math
import jax
import jax.numpy as jnp
from jax.random import split as rsplit
import numpy as np
from sklearn.decomposition import PCA
import torch
import torchvision
import pdb
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 180
import time
import umap

#from jax.config import config
#config.update("jax_debug_nans", True)
print(jax.devices())
print(jax.devices()[0].platform, jax.devices()[0].device_kind)

torch_device = 1
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
		torchvision.transforms.RandomAffine(40.0, translate=(0.06,0.06), scale=(0.90,1.1), shear=4.0,
		interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
                               torchvision.transforms.ToTensor()
                             ])),
  batch_size=batch_size, shuffle=True, pin_memory=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('files/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor()
                             ])),
  batch_size=10000, shuffle=True, pin_memory=True)


mnist_test = enumerate(test_loader)
batch_idx, (testdata, testtarget) = next(mnist_test)
#indata_jax = to_jax(indata)
#indata_rsz = jax.image.resize(indata_jax, (10000, 1, 16, 16), 'bilinear')
#indata1 = indata
#indata1[:, 0, 0, 0] = 1.0 # so the outer product works properly.

testdata_jax = to_jax(testdata)
testdata_jax = jnp.reshape(testdata_jax, (10000, 784))
testtarget_jax = to_jax(testtarget)

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

P = 20
Pp = P*2

fi1, fi2 = outerupt_indx2_flat(siz, roi, True)

bi1, bi2 = outerupt_indx(P, 48)

def index_mul2(v, i1, i2):
	return v[i1] * v[i2]

@jax.jit
def outerupt_f(m2):
	q2 = jax.vmap(index_mul2, (None, 0,0), 0)(m2, fi1, fi2)
	return jnp.concatenate((m2,q2), -1)

@jax.jit
def outerupt_b(m2):
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

M = fs.shape[0]
Q = bs.shape[0]
lr = 0.001

@jax.jit
def sigmoid(x):
	return 0.5 * (jnp.tanh(x / 2) + 1)

@jax.jit
def swish(x):
	return x * 0.5 * (jnp.tanh(x / 2) + 1)

# this model produces a binary compression of the digits
# all the hidden units are (basically) either off or on.
# loss plateaus around 20, though.
# Imperfect but perceptually nice reconstruction..
#@jax.jit
#def run_model(w_f, w_b, l1e):
	#l1o = outerupt_f(l1e)

	#l2x = w_f @ l1o

	#l2c = jnp.clip(l2x, 0.0, 1.0)

	#l2r = outerupt_b(l2c)

	#l1i = jnp.clip(w_b @ l2r, -0.0, 1.0)

	#return l2c, l1i

# this model converges much slower (more difficult gradients?)
# end loss is ~ 26.
# most units are on at the end
#@jax.jit
#def run_model(w_f, w_b, l1e):
	## dumb simple model, with no non-linear least squares compression
	## see what it comes up with.
	#l1o = outerupt_f(l1e)

	#l2x = w_f @ l1o

	#l2c = sigmoid(l2x)

	#l2r = outerupt_b(l2c)

	#l1i = sigmoid(w_b @ l2r)

	#return l2c, l1i

# swish doesn't work well here!

# this model converges quickly & has a low ultimate loss (~10)
# representation remains vectoral -- varied activation at the hidden layer.
#@jax.jit
#def run_model(w_f, w_b, l1e):
	## dumb simple model, with no non-linear least squares compression
	## see what it comes up with.
	#l1o = outerupt_f(l1e)

	#l2x = w_f @ l1o

	#l2c = sigmoid(l2x)

	#l2r = outerupt_b(l2c)

	#l1i = jnp.clip(w_b @ l2r, 0.0, 2.0)

	#return l2c, l1i

# this model converges quickly & has a low ultimate loss (~10)
# representation remains vectoral -- varied activation at the hidden layer.
@jax.jit
def run_model(w_f, w_b, b_2, l1e):
	# dumb simple model, with no non-linear least squares compression
	# see what it comes up with.
	l1o = outerupt_f(l1e)

	l2x = w_f @ l1o

	l2c = sigmoid(l2x) + b_2

	l2r = outerupt_b(l2c)

	l1i = jnp.clip(w_b @ l2r, 0.0, 1.0)

	return l2c, l1i

@jax.jit
def compute_loss(w_f, w_b, b_2, l1e):
	l2c, l1i = run_model(w_f, w_b, b_2, l1e)
	return jnp.sum((l1e - l1i)**2)

# for gradient to work, w_f and w_b need to be non-zero
key,sk = rsplit(key)
w_f = jax.random.normal(sk, (P,M)) * 0.0025
key,sk = rsplit(key)
w_b = jax.random.normal(sk, (784,Q)) * 0.0025
b_2 = jnp.zeros((P,))

slowloss = 0.0

print("w_f", w_f.shape)
print("w_b", w_b.shape)

def mp(indx):
	a = jnp.zeros((10,))
	a = a.at[indx].set(1.0)
	return a

N = 60000
BATCH = 16
for ii in range(32):
	mnist = enumerate(train_loader)
	batch_idx, (indata, intarget) = next(mnist)
	indata_jax = to_jax(indata)
	indata_jax = jnp.reshape(indata_jax, (60000, 784))
	intarget_jax = to_jax(intarget)
	intarget_hot = jax.vmap(mp, 0, 0)(intarget_jax)

	for i in range(int(N/BATCH)):

		def stp(w_f, w_b, b_2, l1e):
			loss, (w_f_grad, w_b_grad, b_2_grad) = jax.value_and_grad(\
				compute_loss, (0, 1, 2))(w_f, w_b, b_2, l1e)
			return loss, w_f_grad, w_b_grad, b_2_grad

		loss, w_f_grad, w_b_grad, b_2_grad = jax.vmap(stp, (None, None, None, 0), 0)\
			(w_f, w_b, b_2, indata_jax[i*BATCH : (i+1)*BATCH])

		w_f_grad = jnp.sum(w_f_grad, 0)
		w_b_grad = jnp.sum(w_b_grad, 0)
		b_2_grad = jnp.sum(b_2_grad, 0)
		w_f = w_f - w_f_grad * lr
		w_b = w_b - w_b_grad * lr
		b_2 = b_2 - b_2_grad * lr

		if i % 5 == 4:
			w_f = 0.999 * w_f
			w_b = 0.999 * w_b

		loss = jnp.mean(loss)
		slowloss = 0.98 * slowloss + 0.02*loss
		if i %100 == 99:
			print(i*BATCH, slowloss, loss, jnp.sum(jnp.abs(w_f_grad)), jnp.sum(jnp.abs(w_b_grad)))

print("w_f", w_f.shape)
print("w_b", w_b.shape)

def run_model_batch(w_f, w_b, b_2, dat):
	 
	def stp(w_f, w_b, b_2, l1e):
		l2c, l1i = run_model(w_f, w_b, b_2, l1e)
		loss = jnp.sum((l1e - l1i)**2)
		return l2c, loss
		
	l2c_all, loss_all = jax.vmap(stp, (None, None, None, 0), 0)(w_f, w_b, b_2, dat)
	return l2c_all, loss_all

# run this on test & asses reconstruction / prediciton. 
l2c_train, loss_train = run_model_batch(w_f, w_b, b_2, indata_jax)
print('mean train loss', jnp.mean(loss_train))

l2c_test, loss_test = run_model_batch(w_f, w_b, b_2, testdata_jax)
print('mean test loss', jnp.mean(loss_test))

# make test one-hots..
testtarget_hot = jax.vmap(mp, 0, 0)(testtarget_jax)

# these regressions need to be done on the CPU
TN = 200
l2c_train1 = np.concatenate((np.asarray(l2c_train[0:TN]), np.ones((TN,1))), 1)
(ww,resid,rank,sing) = np.linalg.lstsq(\
	l2c_train1, np.asarray(intarget_hot[0:TN]), rcond=None)
# use on test (all samples)
l2c_test1 = np.concatenate((np.asarray(l2c_test), np.ones((10000,1))), 1)
pred = l2c_test1 @ ww
indx = jnp.argmax(pred, 1)
pred_hot = jax.vmap(mp, 0, 0)(indx)
err = (testtarget_hot - pred_hot)**2
print('compressed one-hot label prediction error', np.mean(err))

# now do the same for no compression (control) 
indata1 = np.concatenate((np.asarray(indata_jax[0:TN]), np.ones((TN,1))), 1)
(ww,resid,rank,sing) = np.linalg.lstsq(\
	indata1, np.asarray(intarget_hot[0:TN]), rcond=None)
# all samples again.
testdata1 = np.concatenate((np.asarray(testdata_jax), np.ones((10000,1))), 1)
pred = testdata1 @ ww
indx = jnp.argmax(pred, 1)
pred_hot = jax.vmap(mp, 0, 0)(indx)
err = (testtarget_hot - pred_hot)**2
print('naive one-hot label prediction error', np.mean(err))

# also need to compare to PCA (which should be the same as linear regression, but double check just to make sure)
pca = PCA(n_components=P, svd_solver='full')
pca.fit(np.asarray(indata_jax)) # use the full matrix
pcaindata = pca.transform(np.asarray(indata_jax[0:TN]))
pcaindata1 = np.concatenate((pcaindata, np.ones((TN,1))), 1)
(ww,resid,rank,sing) = np.linalg.lstsq(\
	pcaindata1, np.asarray(intarget_hot[0:TN]), rcond=None)

pcatest = pca.transform(np.asarray(testdata_jax))
pcatest1 = np.concatenate((pcatest, np.ones((10000,1))), 1)
pred = pcatest1 @ ww
indx = jnp.argmax(pred, 1)
pred_hot = jax.vmap(mp, 0, 0)(indx)
err = (testtarget_hot - pred_hot)**2
print('PCA one-hot label prediction error', np.mean(err))

# for shits and giggles, let's try to cluster on the compressed data.
d = np.asarray(l2c_train)
e = umap.UMAP(n_components=2, verbose=True).fit_transform(d)

target = np.asarray(intarget)
rgb = np.zeros((60000, 3))
rgb[:,0] = 1.0 - np.clip(target / 4.5, 0.0, 1.0)
rgb[:,1] = np.clip(target / 4.5, 0.0, 1.0) - np.clip((target-4.5)/4.5, 0.0, 1.0)
rgb[:,2] = np.clip((target-4.5)/4.5, 0.0, 1.0)

plt.scatter(e[:,0], e[:,1], c=rgb)
plt.show()

# display a few samples of input - compress - reconstruct.
def prime_factors(n):
	i = 2
	factors = []
	while i * i <= n:
		if n % i:
			i += 1
		else:
			n //= i
			factors.append(i)
	if n > 1:
		factors.append(n)
	return factors

def vec2mtrx(ar):
	n = ar.shape[0]
	factors = prime_factors(n)
	lf = len(factors)
	if lf > 2:
		# need to divide up into two parts.. naively.
		rows = 1
		for i in range(math.ceil(lf/2.0)):
			rows *= factors[i]
		cols = int(n / rows)
		ar = jnp.reshape(ar, (rows, cols))
	else:
		ar = jnp.reshape(ar, (1, n))
	return ar

for i in range(12):
	l1e = indata_jax[i]
	l2c, l1i = run_model(w_f, w_b, b_2, l1e)
	fig, axs = plt.subplots(1,3, figsize=(18, 8))
	im = axs[0].imshow(jnp.reshape(l1e, (28,28)))
	plt.colorbar(im, ax=axs[0])
	im = axs[1].imshow(vec2mtrx(l2c))
	plt.colorbar(im, ax=axs[1])
	im = axs[2].imshow(jnp.reshape(l1i, (28,28)))
	plt.colorbar(im, ax=axs[2])
	plt.show()
