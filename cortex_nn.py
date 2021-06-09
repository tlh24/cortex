import os
import math
import numpy as np
import torch
import random
import pdb
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# test solutions to the XOR problem using modified hopfield networks.  

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
	return torch.mul(w_, dw)


def hebb_update(w_, inp, outp, outpavg, lr):
	dw = torch.outer(outp, inp) 
	# note: dw needs to be both positive and negative.  
	# I tried clamping to [0 1] so as to feed into a power nonlinearity, 
	# but this resulted in the weights all coverging to ~1. 
	# negative hebbian updates are necessary. 
	dw2 = torch.clamp(dw, -1.0, 1.0)
	#dw = torch.mul(torch.pow(torch.mul(dw2, 3.0), 3.0), lr) # anything higher than 3.0 seems to be unstable, anything lower causes more error.
	# the cube nonlinearity seems to work better than straight hebbian, it pretty reliably converges.
	dw = torch.mul(dw2, lr)
	w_ = torch.add(w_, dw)
	# also perform scaling. 
		# instantaneous homeostasis
	scale = torch.clamp(outp, 1.6, 1e6) 
	scale = torch.sub(scale, 1.6)
	scale = torch.exp(torch.mul(scale, -0.04)) 
		# slow minimum homeostasis
	scale2 = torch.clamp(outpavg, 0.0, 0.1)
	scale2 = torch.sub(0.1, scale2) 
	scale2 = torch.exp(torch.mul(scale2, 0.2))
		# slow maximum homeostasis
	scale3 = torch.clamp(outpavg, 1.1, 1e6)
	scale3 = torch.sub(scale3, 1.1) 
	scale3 = torch.exp(torch.mul(scale3, -0.02))
	scale = scale * scale2 * scale3
	dw = torch.outer(scale, torch.ones(inp.size(0)))
	return torch.mul(w_, dw)

# the data generator
insize = 8
w_gen = torch.mul(torch.rand(insize, 4), 1.0/math.sqrt(3.0 * (insize/8.0))); 
#for i in range(10):
	#g2 = torch.clamp(torch.randn(4), 0.0, 1.0)
	#gen = torch.matmul(w_gen, g2)
	#print(g2)
	#print(gen)
	## it looks ok
	
error = torch.zeros(10)
	
for k in range(10): 
	w_f = torch.mul(torch.rand(4, insize), math.sqrt(1.0 / insize))
	w_b = torch.zeros(insize, 5) # let hebb fill these in. One extra for bias. 
	w_l2i = torch.zeros(4, 4)
	l2a = torch.ones(4)
	err = 0.0
	l2u5 = torch.ones(5)

	for i in range(10000):
		g2 = torch.clamp(torch.randn(4), 0.0, 1.0)
		if False:
			g2 = torch.tensor([1.0, 0.0, 0.0, 0.0])
			if i % 4 == 1:
				g2 = torch.tensor([0.0, 1.0, 0.0, 0.0])
			if i % 4 == 2:
				g2 = torch.tensor([0.0, 0.0, 1.0, 0.0])
			if i % 4 == 3:
				g2 = torch.tensor([0.0, 0.0, 0.0, 1.0])
		# the four-pattern task should be exactly solveable -- why is it not? 
		gen = torch.clamp(torch.matmul(w_gen, g2), 0.0, 1.5)
		#invgen = 1.0 - gen
		#catgen = torch.cat((gen, invgen))
		
		l1e = gen
		l2e = torch.clamp(torch.matmul(w_f, l1e), 0.0, 3.0)
		l2i = torch.clamp(torch.matmul(w_l2i, l2e), 0.0, 4.0)
		# l2u = torch.clamp(l2e - l2i, 0.0, 2.0) # allowing l2u to go negative impoves convergence, but it's not necessary.  Improvement is about 0.7 std.
		l2u = l2e - l2i
		#l2u = g2 # debug -- no, error is still not zero!
		l2a = l2a * 0.99 + l2u * 0.01
		
		l2u5[0:4] = l2u # include a bias term
		l1i = torch.clamp(torch.matmul(w_b, l2u5), 0.0, 2.5)
		l1u = l1e - l1i
		
		w_f = hebb_update(w_f, l1u, l2u, l2a, 0.005)
		#w_l2i = inhib_update(w_l2i, l2u, l2i, 0.005)
		
		w_b = hebb_update(w_b, l2u5, l1u, torch.ones(l1u.size(0)), 0.01)
		
		err = 0.99 * err + 0.01 * torch.mean(torch.abs(l1u))
		#print('error', err)
		#print('g2', g2, 'l2u', l2u)
		#print('l1e', l1e[0:8])
		#print('l1i', l1i[0:8])
		#print('l2u', l2u)
		
	print('error', err, 'l2a', l2a)
	error[k] = err
	
print(error)
print(torch.mean(error), torch.std(error))
#print(' ')
#print('w_f', w_f)
#print(' ')
#print('w_b', w_b)

