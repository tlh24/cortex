import os
import math
import numpy as np
import torch
import torchvision
import random
import pdb
import time
import matplotlib.pyplot as plt
from torch import clamp, matmul, outer, mul, add, sub, ones, zeros, reshape

def make_stim(s, x, y):
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

def randbool():
	return torch.randint(0, 2, (1,))

def test_stim():
	for i in range(10):
		s = randbool()
		x = randbool()
		y = randbool()
		o = make_stim(s, x, y)
		plt.imshow(o.numpy())
		plt.show()

# test_stim()

def hebb_update(w_, inp, outp, outpavg, lr):
	#dw = torch.outer(torch.pow(outp, 3), inp) # move the nonlinearity to the output
	dw = torch.outer(outp, inp)
	# note: dw needs to be both positive and negative.
	# I tried clamping to [0 1] so as to feed into a power nonlinearity,
	# but this resulted in the weights all coverging to ~1.
	# negative hebbian updates are necessary.
	dw2 = clamp(dw, -1.0, 1.0)
	dw = torch.pow(dw2 * 5.0, 3.0) # cube is essential!
	#dw = dw2
	ltp = clamp(dw, 0.0, 2.0) # make ltp / ltd asymmetrc
	ltd = clamp(dw, -2.0, 0.0) # to keep weights from going to zero
	lr = lr / math.pow(inp.size(0), 0.35)
	dw = ltp*lr + ltd*lr*(0.9+outer(outpavg*1.5, ones(inp.size(0))))
		# above rule effective at encouraging sparsity
		# but this doesn't work for disentanglement...
	#dw = ltp*lr*0.8 + ltd*lr*1.2
	# the cube nonlinearity seems to work better than straight hebbian, it pretty reliably converges.
	if print_dw:
		print('hebb')
		print(dw)
	w_ = torch.add(w_, dw)
	# also perform down scaling.
	scale = torch.clamp(outp, 0.7, 1e6)
	scale = torch.sub(scale, 0.7)
	scale = torch.exp(torch.mul(scale, -0.06))
	# upscaling makes the learning unstable!
	dws = torch.outer(scale, torch.ones(inp.size(0)))
	w_ = torch.mul(w_, dws)
	w_ = clamp(w_, -0.025, 0.9) # careful about this -- reverse weights need to be large-2.0, 2.0
	# clamping seems unnecessary if the rules are stable--?
	# no it's needed for and-or task.
	return w_, dw

def repvec(v, n):
	d = v.shape[0]
	q = torch.outer(v, ones(n))
	return torch.reshape(q, (d*n,))

def outerupt(v):
	# offset = 0 : output length is N * N+1 / 2
	# offset = 1 : output length is N * N-1 / 2
	indx = torch.triu_indices(v.shape[0], v.shape[0], offset=1)
	o = torch.outer(v, torch.abs(v))
	return o[indx[0], indx[1]]

def plot_tensor(r, c, v, name, lo, hi):
	if len(v.shape) == 1:
		if v.shape[0] == 6:
			v = torch.reshape(v, (2,3))
		elif v.shape[0] == 8:
			v = torch.reshape(v, (2,4))
		elif v.shape[0] == 9:
			v = torch.reshape(v, (3,3))
		elif v.shape[0] == 12:
			v = torch.reshape(v, (3,4))
		elif v.shape[0] == 16:
			v = torch.reshape(v, (4,4))
		elif v.shape[0] == 45:
			v = torch.reshape(v, (1,45))
		elif v.shape[0] == 126:
			v = torch.reshape(v, (6,21))
		elif v.shape[0] == 224:
			v = torch.reshape(v, (8,28))
		else:
			d = v.shape[0]
			v = torch.reshape(v, (1,d))
	if len(v.shape) == 2:
		if v.shape[1] == 28 * 28:
			q = torch.reshape(v, (HID1, 28, 28))
			# display a sample of the tuning functions
			b = [torch.cat([q[i*12+j] for j in range(12)], 1) for i in range(6)]
			v = torch.cat(b)
	if not initialized:
		# seed with random data so we get the range right
		cmap_name = 'PuRd' # purple-red
		if lo == -1*hi:
			cmap_name = 'seismic'
		data = np.linspace(lo, hi, v.shape[0] * v.shape[1])
		data = np.reshape(data, (v.shape[0], v.shape[1]))
		im[r][c] = axs[r,c].imshow(data, cmap = cmap_name)
		cbar[r][c] = plt.colorbar(im[r][c], ax=axs[r,c])
	im[r][c].set_data(v.numpy())
	cbar[r][c].update_normal(im[r][c]) # probably does nothing
	axs[r,c].set_title(name)

nhid = 4
n_f = 36 # 9*8 / 2
n_b = int( (2*nhid*(2*nhid-1)/2)*(2*nhid) ) # 224
w_f = zeros(4*nhid, n_f)
w_b = zeros(4*9, n_b)
print_dw = False
lr = 0.008

animate = True
plot_rows = 4
plot_cols = 3
figsize = (20, 12)
if animate:
	plt.ion()
	fig, axs = plt.subplots(plot_rows, plot_cols, figsize=figsize)
initialized = False
im = [ [0]*plot_cols for i in range(plot_rows)]
cbar = [ [0]*plot_cols for i in range(plot_rows)]
l2 = zeros(nhid)
l2a = zeros(nhid)
supervised = False
N = 1e6

for i in range(int(N)):
	anneal = 1.0 - float(i) / float(2e5)
	anneal = 0.0 if anneal < 0.0 else anneal
	# st, sx, sy = (i%2, int(i/2)%2, int(i/4)%2)
	st, sx, sy = (randbool(), randbool(), randbool())
	l1e = make_stim(st, sx, sy)
	l1e = reshape(l1e, (9,))
	l1o = outerupt(l1e) # product of all the boolean variables.
	l1o = reshape(l1o, (n_f,))

	l2b = w_f @ l1o + torch.randn(4*nhid) * ( 0.03 if supervised else 0.1 ) * anneal
	l2 = torch.sum(reshape(l2b, (nhid, 4)), 1)
	l2a = 0.996 * l2a + 0.004 * l2
	if supervised:
		# supervised learning of the forward weights.
		l2t = zeros(nhid)
		l2t[0] = float(st) # make sure that, when perfectly seeded,
		l2t[1] = float(sx) # forwards network can perfectly reconstruct input.
		l2t[2] = float(sy)
		l2a = ones(nhid) * 0.5
		l2b = repvec(l2t, 4) - l2b
	l2 = clamp(l2, 0.0, 1.5)
	l2p = clamp(2.0*l2a - l2, 0.0, 1.0)
	l2n = torch.cat((l2, l2p), 0)
	# we need at least three terms for revese (alas)
	# in biology the basal dendrites have far fewer synapses...
	l2o = torch.reshape(outer(torch.reshape(outerupt(l2n), (28,)), l2n), (n_b,))
	#l2o = l2o + torch.randn(n_b)* 0.1
	l1ib = w_b @ l2o
	l1i = torch.sum(reshape(l1ib, (9, 4)), 1) # 4 dendritic bins here, too
	l1i = torch.clamp(l1i, 0.0, 1.5)
	#print(l1i)

	l1u = l1e - l1i
	l1ub = repvec(l1u, 4) + torch.randn(36) * 0.1 * anneal

	l1uo = outerupt(l1u)
	l1eo = outerupt(l1e)

	# def hebb_update(w_, inp, outp, outpavg, lr):
	if supervised:
		w_f, dwf = hebb_update(w_f, l1eo, l2b, repvec(l2a,4), lr*1.3)
	else:
		w_f, dwf = hebb_update(w_f, l1uo, l2b, repvec(l2a,4), lr*1.3)
		# complement weights, too.
		# w_f, dwf = hebb_update(w_f, -1.0*l1uo, repvec(l2p,4), repvec(l2a,4), lr*0.8)

	w_b, dwb = hebb_update(w_b, l2o, l1ub, 0.5*ones(9*4), lr)

	if not animate:
		fig, axs = plt.subplots(plot_rows, plot_cols, figsize=figsize)
	if not animate or i % 499 == 498:
		plot_tensor(0, 0, l1e, 'l1e', 0.0, 1.0)
		plot_tensor(1, 0, l1i, 'l1i', 0.0, 1.0)
		plot_tensor(2, 0, l1u, 'l1u', -1.0, 1.0)
		plot_tensor(3, 0, l1uo, 'l1uo', -1.0, 1.0)

		plot_tensor(0, 1, l2b, 'l2b' , 0.0, 1.0)
		plot_tensor(1, 1, l2a, 'l2a', 0.0, 1.0)
		plot_tensor(2, 1, l2o, 'l2o', 0.0, 1.0)

		plot_tensor(0, 2, w_f, 'w_f', -0.5, 0.5)
		plot_tensor(1, 2, dwf, 'dwf', -0.05*lr, 0.05*lr)
		plot_tensor(2, 2, w_b, 'w_b', -0.1, 0.1)
		plot_tensor(3, 2, dwb, 'dwb', -0.25*lr, 0.25*lr)

		fig.tight_layout()
		fig.canvas.draw()
		fig.canvas.flush_events()
		if animate:
			time.sleep(0.2)
			initialized = True
			print(anneal)
			if i == N - 1:
				time.sleep(10)
		else:
			plt.show()


