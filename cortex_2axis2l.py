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
	scale = torch.clamp(outp, 2.0, 1e6)
	scale = torch.sub(scale, 2.0)
	scale = torch.exp(torch.mul(scale, -0.04)) 
		# slow minimum homeostasis
	#add = torch.clamp(outpavg, -0.06, 0.06)
	#add = torch.sub(0.06, add)
	#add = torch.exp(torch.mul(add, 0.05)) - 1.0
		## slow maximum homeostasis
	#add2 = torch.clamp(outpavg, 0.7, 3)
	#add2 = torch.sub(add2, 0.7) 
	#add2 = torch.exp(torch.mul(add2, -0.02)) - 1.0
	
	dw = torch.outer(scale, torch.ones(inp.size(0))) # this gets stuck if there are negative weights! 
	#dw2 = torch.outer(add, torch.ones(inp.size(0)))
	#dw3 = torch.outer(add2, torch.ones(inp.size(0)))
	#w_ = torch.add(w_, dw2)
	#w_ = torch.add(w_, dw3) ## TBD this homeostasis doesn't work, causes instability!!!
	return torch.clamp(torch.mul(w_, dw), -2, 6)

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

noise_level = 0.025
print("noise level", noise_level)

indata = [[0,0,0], [1,0,1], [0,1,1], [1,1,1]] 
# to solve this problem, ignore the third digit, and copy the first two.
if False:
	# this solution works exactly and is stable (obviously) to hebbian update.
	w_f[0,:] = torch.tensor([1, 0, 0, 0, 0, 0])
	w_f[1,:] = torch.tensor([0, 0, 1, 0, 0, 0])
	w_b[0,:] = torch.tensor([ 1, 0, 0])
	w_b[1,:] = torch.tensor([-1, 0, 1])
	w_b[2,:] = torch.tensor([ 0, 1, 0])
	w_b[3,:] = torch.tensor([ 0,-1, 1])
	w_b[4,:] = torch.tensor([ 1, 1, 0])
	w_b[5,:] = torch.tensor([-1,-1, 1])

torch.set_printoptions(sci_mode=False, linewidth=140)
err = 0.0
N = 100
for k in range(1): 
	w_f2 = torch.mul(torch.rand(4, 6), math.sqrt(2.0 / 6.0))
	w_b2 = torch.zeros(6, 5) # let hebb fill these in. 
	w_l2i = torch.zeros(4, 4)
	l2a = torch.ones(4) * 0.5
	l2u5 = torch.ones(5)
	
	w_f3 = torch.mul(torch.rand(2, 4), math.sqrt(2.0 / 6.0))
	w_b3 = torch.zeros(4, 3)
	w_l3i = torch.zeros(2,2)
	l3a = torch.ones(2) * 0.5
	l3u5 = torch.ones(3)
	
	for i in range(N): # realistically, need far fewer than 10k...
		j = i % 4
		ind = indata[j]; 
		l1e_ = [ind[0], ind[0]^1, ind[1], ind[1]^1, ind[2], ind[2]^1]
		l1e = torch.tensor(l1e_).to(torch.float)
		l1u = l1e * 0.5
		l2u = torch.zeros(4)
		l2i = torch.zeros(4)

		for k in range(5):
			l2e = torch.clamp(torch.matmul(w_f2, l1e), 0.0, 1.0)
				# propagate the signal
			l2li = torch.clamp(torch.matmul(w_l2i, l2u), 0.0, 2.0)
			l2u = l2e - l2li + l2i # NB! backward excitation!
			l2u = l2u + torch.randn(4) * noise_level
			l2a = l2a * 0.99 + l2u * 0.01
			
			l3e = torch.clamp(torch.matmul(w_f3, l2u), 0.0, 1.5)
			l3li = torch.clamp(torch.matmul(w_l3i, l3e), 0.0, 3.0) # more async
			l3u = l3e - l3li + (torch.randn(2) * noise_level)
			l3a = l3a * 0.99 + l3u * 0.01
			
			l3u5[0:2] = l3u # include a bias term
			l2i = torch.clamp(torch.matmul(w_b3, l3u5), 0.0, 1.0)
			l2u = l2e - l2li + l2i # NB! backward excitation!
			l2u = l2u + torch.randn(4) * noise_level
			l2a = l2a * 0.99 + l2u * 0.01
			
			l2u5[0:4] = l2u
			l1i = torch.clamp(torch.matmul(w_b2, l2u5), 0.0, 1.0)
			l1u = l1e - l1i + (torch.randn(6) * noise_level)
			
			# propagate the error here; this is based on the annealed actvations.
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
		
		# update forward weight based on inhibited l1 state. 
		# e.g. at equilibrium, will be zero. 
		w_f2 = hebb_update(w_f2, l1u, l2u, l2a, 0.001)
		w_l2i = inhib_update(w_l2i, l2u, l2i, 0.002)
		w_b2 = hebb_update(w_b2, l2u5, l1u, torch.ones(l1u.size(0)), 0.002)
		
		w_f3 = hebb_update(w_f3, l2r, l3u, l3a, 0.001)
		w_l3i = inhib_update(w_l3i, l3u, l3li, 0.002)
		w_b3 = hebb_update(w_b3, l3u5, l2r, torch.ones(l2u.size(0)), 0.002)
		
		er = torch.mean(torch.abs(l1u))
		err = 0.99 * err + 0.01 * er
		
		if True: 
			print("i: ", i, "error: ", err)
			#print("l1e", l1e)
			#print("l1i", l1i)
			#print("l1u", l1u)
			#print("l2u", l2u)
			print("l3e", l3e)
	
	print("error:", err)
	
	if True:
		print("l1e", l1e)
		print("l1i", l1i)
		print("l1u", l1u)
		print("l2u", l2u)
		print("l2a", l2a)
		print("w_f2", w_f2)
		print("w_b2", w_b2)
		print("w_f3", w_f3)
		print("w_b3", w_b3)
		print(" ")
