import os
import math
import numpy as np
import torch
import torchvision
import random
import pdb
import time
import matplotlib.pyplot as plt
from torch import clamp, matmul, outer, mul, add, sub, ones, zeros

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
	# li, layer inhibition (for down scaling)
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
	one = torch.ones(w_.size(1))
	dw = torch.outer(scale, one) 
	w_ = torch.mul(w_, dw)
	return w_


def hebb_update(w_, inp, outp, outpavg, lr):
	dw = torch.outer(outp, inp) 
	# note: dw needs to be both positive and negative.  
	# I tried clamping to [0 1] so as to feed into a power nonlinearity, 
	# but this resulted in the weights all coverging to ~1. 
	# negative hebbian updates are necessary. 
	dw2 = clamp(dw, -1.0, 1.0)
	dw = torch.pow(dw2 * 3.0, 3.0)
	ltp = clamp(dw, 0.0, 1.0) # make ltp / ltd asymmetrc
	ltd = clamp(dw, -1.0, 0.0) # to keep weights from going to zero
	dw = ltp*lr + ltd*lr*1.5
	# the cube nonlinearity seems to work better than straight hebbian, it pretty reliably converges.
	if print_dw:
		print('hebb')
		print(dw)
	w_ = torch.add(w_, dw)
	# also perform scaling. 
	scale = torch.clamp(outp, 1.0, 1e6)
	scale = torch.sub(scale, 1.0)
	scale = torch.exp(torch.mul(scale, -0.06)) 
	scale2 = torch.clamp(outp, 0.0, 0.1) # turning this back on.
	scale2 = torch.sub(0.1, scale2) # if outpavg is zero, this is 0.06
	scale2 = torch.exp(torch.mul(scale2, 0.002)) # then e^0.012 ~= 1.012
	scale = scale * scale2 # so if the output is too low, scale it up. 
	dw = torch.outer(scale, torch.ones(inp.size(0)))
	w_ = torch.mul(w_, dw)
	w_ = clamp(w_, -0.02, 0.15)
	return w_

def initialize_weights_fix(r, c): # dim0, dim1
	a = torch.arange(-0.5, 0.5, 1.0/(r*c))
	w = torch.reshape(a, (c, r)) # jax code is right-multiply
	w = torch.transpose(w, 0, 1)
	return w


# need a known testing distribution! 
def gauss_stim(cx, cy, sigma):
	x = torch.arange(28) - cx
	y = torch.arange(28) - cy
	x = outer(ones(28), x)
	y = outer(y, ones(28))
	v = torch.exp((torch.pow(x, 2) + torch.pow(y, 2)) / (-2.0 * sigma))
	return v
	
def make_stim(i):
	xi = i % 3
	yi = (i//3) % 3
	cx = (xi+1)*7
	cy = (yi+1)*7
	return gauss_stim(cx, cy, 5.0)
	

batch_size = 1
	
#train_loader = torch.utils.data.DataLoader(
  #torchvision.datasets.MNIST('files/', train=True, download=True, 
                             #transform=torchvision.transforms.Compose([
                               #torchvision.transforms.ToTensor()
                             #])),
  #batch_size=1, shuffle=True, pin_memory=True)
#test_loader = torch.utils.data.DataLoader(
  #torchvision.datasets.MNIST('files/', train=False, download=True,
                             #transform=torchvision.transforms.Compose([
                               #torchvision.transforms.ToTensor()
                             #])),
  #batch_size=batch_size, shuffle=True, pin_memory=True)

w_f1 = torch.mul(torch.rand(128, 28*28), math.sqrt(0.25 / (28*28)))
w_b1 = torch.zeros(28*28, 128) # let hebb fill these in. 
w_l2i = torch.zeros(128, 128)
w_f2 = torch.mul(torch.rand(10, 128), math.sqrt(0.25 / (28*28)))
w_b2 = torch.zeros(128, 10) # let hebb fill these in. 
w_l3i = torch.zeros(10, 10)

l1a = torch.zeros(28*28)
l2a = torch.zeros(128)
l3a = torch.zeros(10)

N = 60000

animate = True 
if animate:
	plt.ion()
	fig, axs = plt.subplots(4, 3, figsize=(16, 9))
initialized = False
im = [ [0]*3 for i in range(4)]
cbar = [ [0]*3 for i in range(4)]

for k in range(10): 
	#mnist = enumerate(train_loader)
	for i in range(N): 
		#batch_idx, (indata, target) = next(mnist)
		indata = make_stim(i)
		indata = torch.reshape(indata, (28*28,))

		l1e = indata.to(torch.float)

		l2e = clamp(w_f1 @ l1e, -0.5, 2.5)
		l2li = clamp(w_l2i @ l2e, 0.0, 5.0)
		l2s = l2e - l2li # sparsified
		l1i = clamp(w_b1 @ l2s, 0.0, 5.0)
									  
		l3e = clamp(w_f2 @ l2s, 0.0, 2.5)
		l3li = clamp(w_l3i @ l3e, 0.0, 5.0)
		l3s = l3e - l3li
		l2i = clamp(w_b2 @ l3s, 0.0, 5.0)
		
		l1u = l1e - l1i; 
		l2u = l2s - l2i; 
		l2a = l2a * 0.99 + l2u * 0.01
		l3u = l3s; 
		l3a = l3a * 0.99 + l3u * 0.01
		
		w_f1 = hebb_update(w_f1, l1u, l2s, l2a, 0.002)
		w_f2 = hebb_update(w_f2, l2u, l3u, l3a, 0.002)
		w_b2 = hebb_update(w_b2, l3u, l2u, ones(128), 0.002)
		w_b1 = hebb_update(w_b1, l2s, l1u, ones(28*28), 0.002)
		
		w_l2i = inhib_update(w_l2i, l2u, l2li, 0.002)
		w_l3i = inhib_update(w_l3i, l3u, l3li, 0.002)
		
		# these are images; too complicated to write to stdout. 
		def plot_tensor(r, c, v, name, lo, hi):
			if v.shape[0] == 28*28:
				v = torch.reshape(v, (28,28))
			if v.shape[0] == 128:
				v = torch.reshape(v, (8,16))
			if v.shape[0] == 10:
				v = torch.reshape(v, (2,5))
			if not initialized:
				# seed with random data so we get the range right
				data = np.random.rand(v.shape[0], v.shape[1]) * (hi-lo) + lo
				im[r][c] = axs[r,c].imshow(data, cmap = 'turbo')
				cbar[r][c] = plt.colorbar(im[r][c], ax=axs[r,c])
			im[r][c].set_data(v.numpy())
			cbar[r][c].update_normal(im[r][c]) # probably does nothing
			axs[r,c].set_title(name)
			
		if (i % 1000) >= 990: 
			if not animate:
				fig, axs = plt.subplots(4, 3, figsize=(16, 9))
			plot_tensor(0, 0, l1e, 'l1e', 0.0, 2.5)
			plot_tensor(2, 0, l1i, 'l1i', 0.0, 2.5)
			plot_tensor(3, 0, l1u, 'l1u', -2.5, 2.5)
			
			plot_tensor(0, 1, l2e, 'l2e', 0.0, 2.5)
			plot_tensor(1, 1, l2li, 'l2li', 0.0, 5.0)
			plot_tensor(2, 1, l2s, 'l2s', -1.0, 1.0)
			plot_tensor(3, 1, l2u, 'l2u', -5.0, 2.5)
			
			plot_tensor(0, 2, l3e, 'l3e', 0.0, 2.5)
			plot_tensor(1, 2, l3li, 'l3li', 0.0, 5.0)
			plot_tensor(3, 2, l3u, 'l3u', -5.0, 2.5)
			
			fig.tight_layout()
			fig.canvas.draw()
			fig.canvas.flush_events()
			if animate: 
				time.sleep(0.1)
			initialized = True
