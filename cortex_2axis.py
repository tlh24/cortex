import os
import math
import numpy as np
import torch
import random
import pdb
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# test solutions to a two-axis problem using modified hopfield networks.  


def hebb_update(w_, inp, outp, outpavg, lr):
# 	pdb.set_trace()
	dw = torch.outer(outp, inp) 
	# note: dw needs to be both positive and negative.  
	# I tried clamping to [0 1] so as to feed into a power nonlinearity, 
	# but this resulted in the weights all coverging to ~1. 
	# negative hebbian updates are necessary. 
	dw2 = torch.clamp(dw, -1.0, 1.0)
	dw = torch.mul(torch.pow(torch.mul(dw2, 3.0), 3.0), lr)
	# the cube nonlinearity seems to work better than straight hebbian, it pretty reliably converges.
	#dw = torch.mul(dw2, lr)
	w_ = torch.add(w_, dw)
	# also perform scaling. 
	scale = torch.clamp(outp, 1.5, 1e6)
	scale = torch.sub(scale, 1.5)
	scale = torch.exp(torch.mul(scale, -0.04)) 
	scale2 = torch.clamp(outpavg, 0.0, 0.06)
	scale2 = torch.sub(0.06, scale2)
	scale2 = torch.exp(torch.mul(scale2, 0.2))
		# slow maximum homeostasis
	scale3 = torch.clamp(outpavg, 0.5, 3)
	scale3 = torch.sub(scale3, 0.5) 
	scale3 = torch.exp(torch.mul(scale3, -0.02))
	scale = scale * scale2 * scale3
	dw = torch.outer(scale, torch.ones(inp.size(0)))
	return torch.mul(w_, dw)

def inhib_update(w_, l, li, lr):
	# this function approximates competitive inhibition: 
	# if A and B are both active, but A > B, then A inhibits B, 
	# but B does not inhibit A. 
	# this function also scales the weight matrix exponetially
	# if any of the inputs are > 2.0. 
	# inputs: w_, inhibitory weights
	# l, the current layer e+i activation. 
	# li, layer i (for scaling)
	# lr, learning rate. 
	#pdb.set_trace()
	dw = torch.outer(l, l) # this doesn't break symmetry
	dw2 = torch.outer(torch.pow(l, 2.0), torch.ones(l.size(0))) # this does!
	# it also forces the diagonal to zero. 
	dw2 = dw - dw2
	dw2 = torch.clamp(dw2, -0.1, 0.1)
	# dw2 = torch.clamp(dw - dw2, 0.0, 1.0); # only positive weight updates
	dw2 = torch.mul(dw2, lr)
	w_ = torch.add(w_, dw2)
	scale = torch.clamp(li, 2.5, 1e6)
	scale = torch.sub(scale, 2.5)
	scale = torch.exp(torch.mul(scale, -0.04))
	one = torch.ones(w_.size(0))
	dw = torch.outer(scale, one)
	return torch.clamp(torch.mul(w_, dw), -1.0, 1.0)

w_f = torch.mul(torch.rand(2, 6), math.sqrt(2.0 / 6.0))
w_b = torch.zeros(6, 2) # let hebb fill these in. 
w_l2i = torch.zeros(2, 2)
l2a = torch.zeros(2)

indata = [[0,0,0], [1,0,1], [0,1,1], [1,1,1]] 
# to solve this problem, ignore the third digit, and copy the first two.

for i in range(10000): # realistically, need far fewer than 10k...
	j = i % 4
	ind = indata[j]; 
	l1e_ = [ind[0], ind[0]^1, ind[1], ind[1]^1, ind[2], ind[2]^1]
	l1e = torch.tensor(l1e_).to(torch.float)

	l2e = torch.clamp(torch.matmul(w_f, l1e), 0.0, 1.2)
	l2i = torch.clamp(torch.matmul(w_l2i, l2e), 0.0, 2.0)
	l2u = l2e - l2i
	l2a = l2a * 0.99 + l2u * 0.01
	
	l1i = torch.clamp(torch.matmul(w_b, l2u), 0.0, 1.2)
	l1u = l1e - l1i
	
	# update forward weight based on inhibited l1 state. 
	# e.g. at equilibrium, will be zero. 
	w_f = hebb_update(w_f, l1u, l2u, l2a, 0.01)
	w_l2i = inhib_update(w_l2i, l2u, l2i, 0.02)

	w_b = hebb_update(w_b, l2u, l1u, torch.ones(l1u.size(0)), 0.02)
	
	if (i % 97) == 0: 
		print("l1e", l1e)
		print("l1i", l1i)
		print("l2u", l2u)
		print("  ")
	
print("w_f", w_f)
print("w_b", w_b)
