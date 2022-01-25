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

def hebb_update(w_, inp, outp, outpavg, lr):
	#dw = torch.outer(torch.pow(outp, 3), inp) # move the nonlinearity to the output
	dw = torch.outer(outp, inp)
	# note: dw needs to be both positive and negative.  
	# I tried clamping to [0 1] so as to feed into a power nonlinearity, 
	# but this resulted in the weights all coverging to ~1. 
	# negative hebbian updates are necessary. 
	dw2 = clamp(dw, -1.0, 1.0)
	dw = torch.pow(dw2 * 3.0, 3.0) # cube is essential
	#dw = dw2
	ltp = clamp(dw, 0.0, 2.0) # make ltp / ltd asymmetrc
	ltd = clamp(dw, -2.0, 0.0) # to keep weights from going to zero
	lr = lr / math.pow(inp.size(0), 0.35)
	dw = ltp*lr + ltd*lr*3.0
	# the cube nonlinearity seems to work better than straight hebbian, it pretty reliably converges.
	if print_dw:
		print('hebb')
		print(dw)
	w_ = torch.add(w_, dw)
	# also perform scaling. 
	scale = torch.clamp(outpavg, 0.75, 1e6)
	scale = torch.sub(scale, 0.75)
	scale = torch.exp(torch.mul(scale, -0.06)) 
	# scale2 makes the learning unstable! 
	# need another means of allocation and de-allocation of units... 
	#scale2 = torch.clamp(outp, 0.0, 0.1) 
	#scale2 = torch.sub(0.1, scale2) # if outpavg is zero, this is 0.06
	#scale2 = torch.exp(torch.mul(scale2, 0.002)) # then e^0.012 ~= 1.012
	#scale = scale * scale2 # so if the output is too low, scale it up. 
	dw = torch.outer(scale, torch.ones(inp.size(0)))
	w_ = torch.mul(w_, dw)
	w_ = clamp(w_, -0.0, 1.0) # careful about this -- reverse weights need to be large-2.0, 2.0
	# clamping seems unnecessary if the rules are stable--? 
	return w_

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

def test_inihib_update():
	plt.ion()
	fig, axs = plt.subplots(1, 3, figsize=(16, 10))
	im = [0,0,0]
	cbar = [0,0,0]
	for k in range(3): 
		data = np.linspace(0.0, 2.5, 9)
		data = np.reshape(data, (3, 3))
		im[k] = axs[k].imshow(data, cmap = 'turbo')
		cbar[k] = plt.colorbar(im[k], ax = axs[k])
	wf = torch.rand(9, 3) / 1
	wi = zeros(9,9)
	for k in range(100000): 
		x = zeros(3)
		x[k%3] = 1.0
		#x = torch.rand(3)
		le = clamp(wf @ x, 0.0, 2.5)
		li = clamp(wi @ le, 0.0, 5.0)
		ls = le - li
		wi = inhib_update(wi, ls, li, 0.01)
		if k % 101 == 100:
			im[0].set_data(np.reshape(le, (3, 3)))
			axs[0].set_title('le')
			im[1].set_data(np.reshape(li, (3, 3)))
			axs[1].set_title('li')
			im[2].set_data(np.reshape(ls, (3, 3)))
			axs[2].set_title('ls')
			fig.tight_layout()
			fig.canvas.draw()
			fig.canvas.flush_events()
			time.sleep(0.1)
			
#test_inihib_update()

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
	

#batch_size = 1
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
HID1 = 128
HID2 = 32

# note: these are uniform random numbers, so they average to 0.5
# unform would need the square root. 
w_f1 = torch.zeros(HID1, 28*28)
#w_f1 = torch.rand(HID1, 28*28) / 28
w_b1 = torch.zeros(28*28, HID1) # let hebb fill these in. 
w_l2i = torch.zeros(HID1, HID1)
w_f2 = torch.zeros(HID2, HID1)
w_b2 = torch.zeros(HID1, HID2) # let hebb fill these in. 
w_l3i = torch.zeros(HID2, HID2)

l1a = torch.zeros(28*28)
l2a = torch.zeros(HID1)
l3a = torch.zeros(HID2)

N = 60000

animate = True 
if animate:
	plt.ion()
	fig, axs = plt.subplots(5, 3, figsize=(20, 12))
initialized = False
im = [ [0]*3 for i in range(5)]
cbar = [ [0]*3 for i in range(5)]
lr = 0.005 # if the learning rate is too high, it goes chaoitc
for k in range(25): 
	#mnist = enumerate(train_loader)
	for i in range(N): 
		#batch_idx, (indata, target) = next(mnist)
		indata = make_stim(i)
		indata = torch.reshape(indata, (28*28,))
		# l1e = torch.cat((indata, -indata))
		l1e = indata

		l2e = clamp(w_f1 @ l1e, -0.5, 1.5) + torch.randn(HID1) * 0.1 
		l2li = clamp(w_l2i @ l2e, 0.0, 5.0)
		l2s = l2e - l2li # sparsified
		l1i = clamp(w_b1 @ l2s, 0.0, 1.5) 
		
		# iterate .. ?  gradient-boosting? 
		err = l1e - l1i
		l2e = clamp(w_f1 @ (l1e + 0.5*err), -0.5, 2.5) + torch.randn(HID1) * 0.1 
		l2li = clamp(w_l2i @ l2e, 0.0, 5.0)
		l2s = l2e - l2li # sparsified
		l1i = clamp(w_b1 @ l2s, 0.0, 1.0) 
									  
		l3e = clamp(w_f2 @ l2e, 0.0, 2.5) + torch.randn(HID2) * 0.1 
		l3li = clamp(w_l3i @ l3e, 0.0, 5.0)
		l3s = l3e - l3li
		l2i = clamp(w_b2 @ l3s, 0.0, 5.0)
		
		l1u = l1e - l1i 
		l2u = l2s - l2i
		l2a = l2a * 0.99 + l2s * 0.01
		l3u = l3s 
		l3a = l3a * 0.99 + l3u * 0.01
		
		w_f1 = hebb_update(w_f1, l1u, l2s, l2a, lr)
		w_f2 = hebb_update(w_f2, l2u, l3s, l3a, lr)
		w_b2 = hebb_update(w_b2, l3u, l2u, 0.5*ones(HID1), lr)
		w_b1 = hebb_update(w_b1, l2s, l1u, 0.5*ones(28*28), lr)
		
		#w_l2i = inhib_update(w_l2i, l2s, l2li, lr*1.0)
		#w_l3i = inhib_update(w_l3i, l3s, l3li, lr*1.0)
		
		# these are images; too complicated to write to stdout. 
		def plot_tensor(r, c, v, name, lo, hi):
			if len(v.shape) == 1:
				if v.shape[0] == 2*28*28:
					v = torch.reshape(v, (2,28,28))
					v = torch.cat((v[0,:,:], v[1,:,:]), 1)
				if v.shape[0] == 28*28:
					v = torch.reshape(v, (28,28))
				if v.shape[0] == 256:
					v = torch.reshape(v, (16,16))
				if v.shape[0] == 128:
					v = torch.reshape(v, (8,16))
				if v.shape[0] == 64:
					v = torch.reshape(v, (8,8))
				if v.shape[0] == 32:
					v = torch.reshape(v, (4,8))
				if v.shape[0] == 10:
					v = torch.reshape(v, (2,5))
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
				data = np.random.rand(v.shape[0], v.shape[1]) * (hi-lo) + lo
				im[r][c] = axs[r,c].imshow(data, cmap = cmap_name)
				cbar[r][c] = plt.colorbar(im[r][c], ax=axs[r,c])
			im[r][c].set_data(v.numpy())
			cbar[r][c].update_normal(im[r][c]) # probably does nothing
			axs[r,c].set_title(name)
			
		if (i % 3500) >= 3491: 
			if not animate:
				fig, axs = plt.subplots(4, 3, figsize=(16, 9))
			plot_tensor(0, 0, l1e, 'l1e', -2.0, 2.0)
			plot_tensor(1, 0, l1i, 'l1i', -2.0, 2.0)
			plot_tensor(2, 0, l1u, 'l1u', -1.0, 1.0)
			plot_tensor(3, 0, w_f1, 'w_f1', -0.2, 0.2)
			plot_tensor(4, 0, w_b1.T, 'w_b1', -0.5, 0.5)
			
			plot_tensor(0, 1, l2e, 'l2e', 0.0, 2.5)
			plot_tensor(1, 1, l2li, 'l2li', 0.0, 5.0)
			plot_tensor(2, 1, l2i, 'l2i', 0.0, 2.5)
			plot_tensor(3, 1, l2s, 'l2s', -2.5, 2.5)
			plot_tensor(4, 1, l2u, 'l2u', -1.0, 1.0)
			
			plot_tensor(0, 2, l3e, 'l3e', 0.0, 2.5)
			plot_tensor(1, 2, l3li, 'l3li', 0.0, 2.5)
			plot_tensor(2, 2, l3s, 'l3s', -2.5, 2.5)
			plot_tensor(3, 2, l3u, 'l3u', -2.5, 2.5)
			plot_tensor(4, 2, w_f2, 'w_f2', -2.0, 2.0)
			
			fig.tight_layout()
			fig.canvas.draw()
			fig.canvas.flush_events()
			if animate: 
				time.sleep(0.1)
			initialized = True
