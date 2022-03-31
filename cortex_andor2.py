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
	dw = torch.pow(dw2 * 3.0, 3.0) # cube is essential!
	#dw = dw2
	ltp = clamp(dw, 0.0, 2.0) # make ltp / ltd asymmetrc
	ltd = clamp(dw, -2.0, 0.0) # to keep weights from going to zero
	lr = lr / math.pow(inp.size(0), 0.35)
	dw = ltp*lr + ltd*1.0*lr*(0.9+outer(outpavg*1.5, ones(inp.size(0))))
		# above rule effective at encouraging sparsity
		# but this doesn't work for disentanglement...
	#dw = ltp*lr*0.8 + ltd*lr*1.2
	# the cube nonlinearity seems to work better than straight hebbian, it pretty reliably converges.
	if print_dw:
		print('hebb')
		print(dw)
	w_ = torch.add(w_, dw)
	# also perform down scaling.
	scale = torch.clamp(outp, 0.5, 1e6)
	scale = torch.sub(scale, 0.5)
	scale = torch.exp(torch.mul(scale, -0.08))
	# upscaling makes the learning unstable!
	dws = torch.outer(scale, torch.ones(inp.size(0)))
	w_ = torch.mul(w_, dws)
	w_ = clamp(w_, -0.025, 1.25) # careful about this -- reverse weights need to be large-2.0, 2.0
	# clamping seems unnecessary if the rules are stable--?
	# no it's needed for and-or task.
	return w_, dw

def inhib_update(w_, l, li, lr):
	# this function approximates competitive inhibition:
	# if A and B are both active, but A > B, then A inhibits B,
	# but B does not inhibit A.
	# this function also scales the weight matrix exponetially
	# if any of the inputs are > 2.0.
	# inputs: w_, inhibitory weights
	# l, the current layer e+i activation aka u.
	# li, layer inhibition (for down scaling)
	# lr, learning rate.
	lr = lr / math.pow(l.size(0), 0.35)
	s = l.size(0)
	a = outer(l, torch.ones(s))
	b = outer(torch.ones(s), l + torch.randn(s) * 0.0)
	dw = (b @ b) @ a - (a @ a) @ b
	# dw = (a *b - a*a) *b, where a is pre and b is post.
	# want to break symmetry and scale by post activity
	dw = torch.clamp(dw, -0.05, 0.2) # adjust this to change sparsity
	dw = torch.mul(dw, lr)
	if False:
		print('ls')
		print(l)
		print('li')
		print(li)
		print('inhib')
		print(dw)
		print(' ')
	w_ = torch.add(w_, dw)
	# checked: this is needed for sparsity.
	# surprisingly, adding weight down-scaling makes them *larger* - ??
	scale = torch.clamp(li, 2.0, 1e6)
	scale = torch.sub(scale, 2.0)
	scale = torch.exp(torch.mul(scale, -0.06))
	one = torch.ones(w_.size(1))
	dw = torch.outer(scale, one)
	w_ = torch.mul(w_, dw)
	return w_

def repvec(v, n):
	d = v.shape[0]
	q = torch.outer(v, ones(n))
	return torch.reshape(q, (d*n,))

scaler = 1.0 # blind guess bro

# these manual for loops are going to be so very slow, need to put in JAX!
def outerupt2(v):
	# outer product upper-triangle, terms with two or one factor.
	nin = v.shape[0]
	nout = int( (nin+1) * nin * 0.5 + nin )
	r = zeros(nout)
	e = 0
	for i in range(nin):
		for j in range(i, nin):
			if not(i == j):
				r[e] = v[i] * v[j] * scaler
				e = e + 1
	for i in range(nin):
		r[e] = v[i]
		e = e + 1
	return r

def outerupt3(v):
	# outer product upper-triangle, but for terms with three, two, and one factor
	nin = v.shape[0]
	nout = int( nin * (nin+1) * (nin+2) / 6.0 + \
		nin * (nin+1) * 0.5 + nin )
	# this took me a surprisingly long time to figure out... sigh
	r = zeros(nout)
	e = 0
	for i in range(nin):
		for j in range(i, nin):
			for k in range(j, nin):
				r[e] = v[i] * v[j] * v[k] * scaler * scaler
				e = e + 1
	for i in range(nin):
		for j in range(i, nin):
			r[e] = v[i] * v[j] * scaler
			e = e + 1
	for i in range(nin):
		r[e] = v[i]
		e = e + 1
	return r

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
		elif v.shape[0] == 164:
			v = torch.reshape(v, (4,41))
		elif v.shape[0] == 224:
			v = torch.reshape(v, (8,28))
		elif v.shape[0] == 260:
			v = torch.reshape(v, (10,26))
		elif v.shape[0] == 764: # 765 is more easily factorizable
			v = torch.cat((v, zeros(1)), 0)
			v = torch.reshape(v, (17,45))
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



M = 54
K = 16
P = 4
KP = int(K/P)
Pp = P*2 # 8
Q = int(  Pp * (Pp+1) * (Pp+2) / 6.0 + 0.5 * Pp * (Pp+1) + Pp ) # 164
R = 9
w_f = zeros(K, M) # 45 x 16
w_b = zeros(R, Q) # 764 x 9
print_dw = False
lr = 0.02

animate = True
plot_rows = 4
plot_cols = 3
figsize = (19, 11)
if animate:
	plt.ion()
	fig, axs = plt.subplots(plot_rows, plot_cols, figsize=figsize)
initialized = False
im = [ [0]*plot_cols for i in range(plot_rows)]
cbar = [ [0]*plot_cols for i in range(plot_rows)]
l2 = zeros(P)
l2a = zeros(P)
l1ua = zeros(9)
supervised = False
N = 1e6

for i in range(int(N)):
	anneal = 1.0 - float(i) / float(2e5)
	anneal = 0.0 if anneal < 0.0 else anneal
	q = i % 4
	st, sx, sy = (int(q/4)%2, q%2, int(q/2)%2)
	# st, sx, sy = (randbool(), randbool(), randbool())
	l1e = make_stim(st, sx, sy)
	l1e = reshape(l1e, (9,))
	l1o = outerupt2(l1e)

	noiz = torch.randn(K) * ( 0.03 if supervised else 0.1 )
	l2d = w_f @ l1o + noiz
	l2 = torch.sum(reshape(l2d, (P, KP)), 1)
	l2a = 0.996 * l2a + 0.004 * l2
	if supervised:
		# 'gentile' supervised learning of the forward weights.
		l2t = zeros(P)
		l2t[0] = float(st) # make sure that, when perfectly seeded,
		l2t[1] = float(sx) # forwards network can perfectly reconstruct input.
		l2t[2] = float(sy)
		l2 = 0.05 * l2 + 0.95 * l2t
		l2a = ones(P) * 0.5
		l2s = repvec(l2, KP) - l2d
	l2 = clamp(l2, 0.0, 1.5)
	l2p = clamp(2.0*l2a - l2, 0.0, 1.5)
	l2n = torch.cat((l2, l2p), 0)
	# we need at least three multiplicative terms for revese (alas)
	# or we need two layers
	# yet in biology the basal dendrites have far fewer synapses...
	l2o = outerupt3(l2n)
	l1i = w_b @ l2o
	l1i = torch.clamp(l1i, 0.0, 1.5)

	l1u = l1e - l1i
	l1ua = 0.996 * l1ua + 0.004 * l1u

	# I think we need to 'boost' it to encourage a better latent representation.
	# ala the up-down-up-down algorithm in contrastive divergence
	l1o = outerupt2(clamp(l1e - 0.4*l1i, 0.0, 1.5))
	l2d = w_f @ l1o + noiz

	l1uo = outerupt2(l1u)
	l1eo = outerupt2(l1e)

	# def hebb_update(w_, inp, outp, outpavg, lr):
	if supervised:
		w_f, dwf = hebb_update(w_f, l1eo, l2s, repvec(l2a,KP), lr*1.0)
	else:
		w_f, dwf = hebb_update(w_f, l1uo, l2d, repvec(l2a,KP), lr*2.0)

	w_b, dwb = hebb_update(w_b, l2o, l1u, 0.5*ones(9), lr*1.0)

	if not animate:
		fig, axs = plt.subplots(plot_rows, plot_cols, figsize=figsize)
	if not animate or i % 53 == 52:
		plot_tensor(0, 0, l1e, 'l1e', 0.0, 1.0)
		plot_tensor(1, 0, l1i, 'l1i', 0.0, 1.0)
		plot_tensor(2, 0, l1u, 'l1u', -1.0, 1.0)
		plot_tensor(3, 0, l1eo, 'l1eo', -1.0, 1.0)

		plot_tensor(0, 1, l2d, 'l2d' , 0.0, 1.0)
		plot_tensor(1, 1, l2a, 'l2a', 0.0, 1.0)
		plot_tensor(2, 1, l2o, 'l2o', 0.0, 1.0)
		plot_tensor(3, 1, l1ua, 'l1ua', -1.0, 1.0)

		plot_tensor(0, 2, w_f, 'w_f', -1.5, 1.5)
		plot_tensor(1, 2, dwf, 'dwf', -0.05*lr, 0.05*lr)
		plot_tensor(2, 2, w_b, 'w_b', -0.1, 0.1)
		plot_tensor(3, 2, dwb, 'dwb', -0.25*lr, 0.25*lr)

		fig.tight_layout()
		fig.canvas.draw()
		fig.canvas.flush_events()
		if animate:
			time.sleep(0.02)
			initialized = True
			print(q, st, sx, sy, anneal)
			if i == N - 1:
				time.sleep(10)
		else:
			plt.show()


