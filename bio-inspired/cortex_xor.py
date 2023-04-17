import os
import math
import numpy as np
import torch
import random
import pdb
import matplotlib.pyplot as plt
import matplotlib.animation as animation

print_dw = False
fixed_init = False

# test solutions to the XOR problem using modified hopfield networks.  

def inhib_update(w_, l, li, lr):
	# this function approximates competitive inhibition: 
	# if A and B are both active, but A > B, then A inhibits B, 
	# but B does not inhibit A. 
	# this function also scales the weight matrix exponetially
	# if any of the inputs are > 2.0. 
	# inputs: w_, inhibitory weights
	# l, the current layer e+i activation aka u. 
	# li, layer i (for scaling)
	# lr, learning rate. 
	dw = torch.outer(l, l) # this doesn't break symmetry
	dw2 = torch.outer(torch.pow(l, 2.0), torch.ones(l.size(0))) # this does!
	# it also forces the diagonal to zero. 
	dw2 = dw - dw2
	dw2 = torch.clamp(dw2, -0.005, 0.1) # adjust this to change sparsity
	dw2 = torch.mul(dw2, lr)
	if print_dw:
		print('inhib')
		print(dw2)
	w_ = torch.add(w_, dw2)
	# checked: this is needed for sparsity. 
	# surprisingly, adding weight down-scaling makes them *larger* - ? 
	scale = torch.clamp(li, 2.5, 1e6)
	scale = torch.sub(scale, 2.5)
	scale = torch.exp(torch.mul(scale, -0.04))
	one = torch.ones(w_.size(0))
	dw = torch.outer(scale, one)
	return torch.mul(w_, dw)


def hebb_update(w_, inp, outp, lr):
	dw = torch.outer(outp, inp) 
	# note: dw needs to be both positive and negative.  
	# I tried clamping to [0 1] so as to feed into a power nonlinearity, 
	# but this resulted in the weights all coverging to ~1. 
	# negative hebbian updates are necessary. 
	dw2 = torch.clamp(dw, -1.0, 1.0)
	dw = torch.mul(torch.pow(torch.mul(dw2, 3.0), 3.0), lr)
	# the cube nonlinearity seems to work better than straight hebbian, it pretty reliably converges.
	if print_dw:
		print('hebb')
		print(dw)
	w_ = torch.add(w_, dw)
	# also perform scaling. 
	scale = torch.clamp(outp, 1.5, 1e6)
	scale = torch.sub(scale, 1.5)
	scale = torch.exp(torch.mul(scale, -0.04)) 
	#scale2 = torch.clamp(outpavg, 0.0, 0.06) # this is unnecessary for convergence. 
	#scale2 = torch.sub(0.06, scale2) # if outpavg is zero, this is 0.06
	#scale2 = torch.exp(torch.mul(scale2, 0.2)) # then e^0.012 ~= 1.012
	#scale = scale * scale2 # so if the output is too low, scale it up. 
	dw = torch.outer(scale, torch.ones(inp.size(0)))
	return torch.mul(w_, dw)

def initialize_weights_fix(r, c): # dim0, dim1
	a = torch.arange(-0.5, 0.5, 1.0/(r*c))
	w = torch.reshape(a, (c, r)) # jax code is right-multiply
	w = torch.transpose(w, 0, 1)
	return w

w_f = torch.mul(torch.rand(4, 6), math.sqrt(2.0 / 6.0))
w_b = torch.zeros(6, 4) # let hebb fill these in. 
w_l2i = torch.zeros(4, 4)
if fixed_init: 
	w_f = initialize_weights_fix(4, 6)
	w_b = initialize_weights_fix(6, 4)
	w_l2i = initialize_weights_fix(4, 4)

l1a = torch.zeros(6)
l2a = torch.zeros(4)

indata = [[0,0], [0,1], [1,0], [1,1]]

N = 400
if fixed_init or print_dw: 
	N == 4

for i in range(N): 
	j = i % 4
	ind = indata[j]; 
	l1e_ = [ind[0], ind[0]^1, ind[1], ind[1]^1, ind[0]^ind[1], (ind[0]^ind[1])^1]
	l1e = torch.tensor(l1e_).to(torch.float)

	l2e = torch.clamp(torch.matmul(w_f, l1e), 0.0, 2.5)
	l2i = torch.clamp(torch.matmul(w_l2i, l2e), 0.0, 5.0)
	l2u = l2e - l2i
	
	l1i = torch.clamp(torch.matmul(w_b, l2u), 0.0, 5.0)
	l1u = l1e - l1i
	
	# update forward weight based on inhibited l1 state. 
	# e.g. at equilibrium, will be zero. 
	w_f = hebb_update(w_f, l1u, l2u, 0.02)
	w_l2i = inhib_update(w_l2i, l2u, l2i, 0.04)
	# this *usually* works, until it doesn't (due to weight initializations, suppose)
	# similarly, update reverse weight .. 
	w_b = hebb_update(w_b, l2u, l1u, 0.02)
	
	#if (i % 97) == 0: 
	print("l1e", l1e)
	print("l1i", l1i)
	print("l1u", l1u)
	print('----------')
	print("l2e", l2e)
	print("l2i", l2i)
	print("l2u", l2u)
	print("  ")
	
if True: 
	fig,axs = plt.subplots(1,3, figsize=(16,8))
	im = axs[0].imshow(w_f.numpy())
	plt.colorbar(im, ax=axs[0])
	axs[0].set_title('w_f')

	im = axs[1].imshow(w_b.numpy())
	plt.colorbar(im, ax=axs[1])
	axs[1].set_title('w_b')

	im = axs[2].imshow(w_l2i.numpy())
	plt.colorbar(im, ax=axs[2])
	axs[2].set_title('w_l2i')

	fig.tight_layout()
	plt.show()
	
