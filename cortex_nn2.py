import os
import math
import numpy as np
import torch
import random
import pdb
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 2-layer neural network, feedback alignment + hebbian  

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
	scale = torch.clamp(outp, 0.5, 3) 
	scale = torch.sub(scale, 0.5)
	scale = torch.exp(torch.mul(scale, -0.04)) 
		# slow minimum homeostasis
	scale2 = torch.clamp(outpavg, 0.0, 0.1)
	scale2 = torch.sub(0.1, scale2) 
	scale2 = torch.exp(torch.mul(scale2, 0.2))
		# slow maximum homeostasis
	scale3 = torch.clamp(outpavg, 0.7, 3)
	scale3 = torch.sub(scale3, 0.7) 
	scale3 = torch.exp(torch.mul(scale3, -0.02))
	scale = scale * scale2 * scale3
	dw = torch.outer(scale, torch.ones(inp.size(0)))
	return torch.clamp(torch.mul(w_, dw), -1.0, 1.0)

# the data generator
insize = 60
hidsize = 20
outsize = 4
N = 5000
w_gen1 = torch.mul(torch.randn(insize, hidsize), 1.0/math.sqrt(1.0 * (insize/8.0))); 
w_gen2 = torch.mul(torch.randn(hidsize, outsize), 1.0/math.sqrt(1.0 * (hidsize/8.0))); 
for i in range(0):
	g3 = torch.zeros(outsize)
	while torch.sum(g3) < 0.01:
		g3 = torch.clamp(torch.randn(outsize), 0.0, 1.0)
	g2 = torch.clamp(torch.matmul(w_gen2, g3), 0.0, 1.0)
	g1 = torch.clamp(torch.matmul(w_gen1, g2), 0.0, 1.0)
	print(g3)
	print(g2[0:8])
	print(g1[0:8])
	print('')
	# it looks ok
	
error = torch.zeros(10)
	
for k in range(1): 
	w_f2 = torch.mul(torch.randn(hidsize, insize), math.sqrt(0.2/insize))
	w_b2 = torch.zeros(insize, hidsize+1) # One extra for bias.
	w_l2i = torch.zeros(hidsize, hidsize)
	l2a = torch.ones(hidsize)
	l2u5 = torch.ones(hidsize+1)
	
	w_f3 = torch.mul(torch.randn(outsize, hidsize), 0.2*math.sqrt(0.2/hidsize))
	w_b3 = torch.zeros(hidsize, outsize+1)
	w_l3i = torch.zeros(outsize, outsize)
	l3a = torch.ones(outsize)
	l3u5 = torch.ones(outsize+1)
	
	err = 0.0

	for i in range(N):
		g3 = torch.zeros(outsize)
		while torch.sum(g3) < 0.01:
			g3 = torch.clamp(torch.randn(outsize), 0.0, 1.0)
		g2 = torch.clamp(torch.matmul(w_gen2, g3), 0.0, 1.0)
		g1 = torch.clamp(torch.matmul(w_gen1, g2), 0.0, 1.0)
		
		l1i = torch.zeros(insize)
		l2i = torch.zeros(hidsize)
		l2u = torch.zeros(hidsize)
		l1e = g1
		l1u = l1e * 0.5 # init? 
		
		for k in range(5):
			l2e = torch.clamp(torch.matmul(w_f2, l1e), 0.0, 1.0)
			l2li = torch.clamp(torch.matmul(w_l2i, l2u), 0.0, 1.0) # async
			l2u = l2e - l2li + l2i # NB! backward excitation!
			l2a = l2a * 0.99 + l2u * 0.01
			if torch.sum(torch.isnan(l2u)) > 0:
				pdb.set_trace()
			
			l3e = torch.clamp(torch.matmul(w_f3, l2u), 0.0, 1.0)
			l3li = torch.clamp(torch.matmul(w_l3i, l3e), 0.0, 1.0) # more async
			l3u = l3e - l3li + (torch.randn(outsize) * 0.01)
			l3a = l3a * 0.99 + l3u * 0.01
			if torch.sum(torch.isnan(l3u)) > 0:
				pdb.set_trace()
			
			l3u5[0:outsize] = l3u # include a bias term
			l2i = torch.clamp(torch.matmul(w_b3, l3u5), 0.0, 1.0)
			l2li = torch.clamp(torch.matmul(w_l2i, l2u), 0.0, 1.0) # async
			l2u = l2e - l2li + l2i + (torch.randn(hidsize) * 0.01)
			l2a = l2a * 0.99 + l2u * 0.01
			if torch.sum(torch.isnan(l2i)) > 0:
				pdb.set_trace()
			
			l2u5[0:hidsize] = l2u
			l1i = torch.clamp(torch.matmul(w_b2, l2u5), 0.0, 1.0)
			l1u = l1e - l1i # lateral inhibition here? 
			if torch.sum(torch.isnan(l1u)) > 0:
				pdb.set_trace()
			
			# propagate the error here; this is based on the annealed actctivations.
			l2r = torch.clamp(torch.matmul(w_f2, l1u), -1.0, 1.0)
			
			if i > N-5 and k >= 4:
				print(i,k)
				print('l1e', l1e[0:8])
				print('l1i', l1i[0:8])
				print('l1u', l1u[0:8])
				print('l2e', l2e[0:8])
				print('l2i', l2i[0:8])
				print('l2li', l2li[0:8])
				print('l2u', l2u[0:8])
				print('l2r', l2r[0:8])
				print('l2a', l2a[0:8])
				print('l3e', l3e[0:8])
				print('l3li', l3li[0:8])
				print('l3a', l3a[0:8])
				print('')
		
		w_f2 = hebb_update(w_f2, l1u, l2u, l2a, 0.001)
		w_l2i = inhib_update(w_l2i, l2u, l2li, 0.001)
		w_b2 = hebb_update(w_b2, l2u5, l1u, torch.ones(l1u.size(0)), 0.002)
		
		w_f3 = hebb_update(w_f3, l2r, l3u, l3a, 0.001)
		w_l3i = inhib_update(w_l3i, l3u, l3li, 0.001)
		w_b3 = hebb_update(w_b3, l3u5, l2r, torch.ones(l2u.size(0)), 0.002)
		
		er = torch.mean(torch.abs(l1u))
		err = 0.99 * err + 0.01 * er
		print('error', er, err)
		
	#print('error', err, 'l2a', l2a)
	error[k] = err
	
#print(error)
#print(torch.mean(error), torch.std(error))
#print(' ')
#print('w_f', w_f)
#print(' ')
#print('w_b', w_b)
