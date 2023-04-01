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
	dw = ltp*lr + ltd*lr*(0.9+outer(outpavg*1.5, ones(inp.size(0))))
	# the cube nonlinearity seems to work better than straight hebbian, it pretty reliably converges.
	if print_dw:
		print('hebb')
		print(dw)
	w_ = torch.add(w_, dw)
	# also perform scaling. 
	scale = torch.clamp(outp, 1.25, 1e6)
	scale = torch.sub(scale, 1.25)
	scale = torch.exp(torch.mul(scale, -0.06)) 
	# scale2 makes the learning unstable! 
	# need another means of allocation and de-allocation of units... 
	#scale2 = torch.clamp(outpavg, 0.0, 0.1) 
	#scale2 = torch.sub(0.1, scale2) # if outpavg is zero, this is 0.06
	#scale2 = torch.exp(torch.mul(scale2, 0.002)) # then e^0.012 ~= 1.012
	#scale = scale * scale2 # so if the output is too low, scale it up. 
	dw = torch.outer(scale, torch.ones(inp.size(0)))
	w_ = torch.mul(w_, dw)
	w_ = clamp(w_, -0.000025, 0.9) # careful about this -- reverse weights need to be large-2.0, 2.0
	# clamping seems unnecessary if the rules are stable--? 
	return w_


def initialize_weights_fix(r, c): # dim0, dim1
	a = torch.arange(-0.5, 0.5, 1.0/(r*c))
	w = torch.reshape(a, (c, r)) # jax code is right-multiply
	w = torch.transpose(w, 0, 1)
	return w

def make_circle():
	accept = False
	while (not accept):
		cx = np.random.randn(1) * 12 + 16
		cy = np.random.randn(1) * 12 + 16
		r = np.random.rand(1) * 16 + 4
		if cx - r >= 1 and cx + r < 31 and cy - r >= 1 and cy + r < 31:
			accept = True
	# paint an anti-aliased line (+1 in line, 0 otherwise)
	inner = r - 2.0
	x = outer(torch.arange(0, 32), ones(32))
	y = outer(ones(32), torch.arange(0, 32))
	z = torch.sqrt((x - cx)**2 + (y - cy)**2)
	out1 = clamp(1-(z-r), 0.0, 1.0)
	out2 = clamp(z-inner, 0.0, 1.0)
	out = out1 * out2
	return out


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
	

HID1 = 128
HID2 = 32

# it's great we can just start with zeros here. 
w_f1 = torch.zeros(HID1, 28*28)
#w_f1 = torch.rand(HID1, 28*28) / 28
w_b1 = torch.zeros(28*28, HID1) # let hebb fill these in. 
w_l2i = torch.zeros(HID1, HID1)
w_f2 = torch.zeros(HID2, HID1)
w_b2 = torch.zeros(HID1, HID2) # let hebb fill these in. 
w_l3i = torch.zeros(HID2, HID2)

l1a = torch.zeros(28*28)
l2a = torch.ones(HID1) * 0.5
l3a = torch.ones(HID2) * 0.5

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
	mnist = enumerate(train_loader)
	for i in range(N): 
		batch_idx, (indata, target) = next(mnist)
		#indata = make_stim(i)
		indata = torch.reshape(indata, (28*28,))
		# l1e = torch.cat((indata, -indata))
		l1e = indata

		l2e = clamp(w_f1 @ l1e, -0.5, 1.5) + torch.randn(HID1) * 0.25
		l2li = clamp(w_l2i @ l2e, 0.0, 5.0)
		l2s = l2e - l2li # sparsified
		l1i = clamp(w_b1 @ l2s, 0.0, 1.5) 
		
		# iterate .. ?  gradient-boosting? 
		err = l1e - l1i
		noiz2 = torch.randn(HID1) * torch.exp(clamp(l2a,0.0,1.0) * -10) * 0.7
		l2e = clamp(w_f1 @ (l1e + 0.35*err), -0.5, 2.5) + noiz2
		l2li = clamp(w_l2i @ l2e, 0.0, 5.0)
		l2s = l2e - l2li # sparsified
		l1i = clamp(w_b1 @ l2s, 0.0, 1.0) 
		l2s = clamp(l2s, 0.0, 1.5) # otherwise, negative * negative = ltp
									  
		# redo the forward activation, no noise. 
		l2ep = clamp(w_f1 @ l1e, -0.5, 1.5)
		noiz3 = torch.randn(HID2) * torch.exp(clamp(l3a,0.0,1.0) * -10) * 0.7
		l3e = clamp(w_f2 @ l2ep, 0.0, 2.5) + noiz3
		l3li = clamp(w_l3i @ l3e, 0.0, 5.0)
		l3s = l3e - l3li
		l2i = clamp(w_b2 @ l3s, 0.0, 5.0)
		
		l1u = l1e - l1i 
		l2u = l2ep - l2i
		l2a = l2a * 0.995 + l2s * 0.005
		l3u = l3s 
		l3a = l3a * 0.995 + l3u * 0.005
		
		w_f1 = hebb_update(w_f1, l1u, l2s, l2a, lr)
		w_f2 = hebb_update(w_f2, l2u, l3s, l3a, lr*0.1)
		w_b2 = hebb_update(w_b2, l3u, l2u, 0.5*ones(HID1), lr*0.1)
		w_b1 = hebb_update(w_b1, l2s, l1u, 0.5*ones(28*28), lr)

		
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
			
		if (i % 3500) >= 3499: 
			if not animate:
				fig, axs = plt.subplots(4, 3, figsize=(16, 9))
			plot_tensor(0, 0, l1e, 'l1e', -2.0, 2.0)
			plot_tensor(1, 0, l1i, 'l1i', -2.0, 2.0)
			plot_tensor(2, 0, l1u, 'l1u', -1.0, 1.0)
			plot_tensor(3, 0, w_f1, 'w_f1', -0.5, 0.5)
			plot_tensor(4, 0, w_b1.T, 'w_b1', -0.5, 0.5)
			
			plot_tensor(0, 1, l2e, 'l2e', -2.5, 2.5)
			plot_tensor(1, 1, l2a, 'l2a', -1.0, 1.0)
			plot_tensor(2, 1, l2i, 'l2i', -2.5, 2.5)
			plot_tensor(3, 1, l2s, 'l2s', -2.5, 2.5)
			plot_tensor(4, 1, l2u, 'l2u', -1.0, 1.0)
			
			plot_tensor(0, 2, l3e, 'l3e', -2.5, 2.5)
			plot_tensor(1, 2, l3a, 'l3a', -1.0, 1.0)
			plot_tensor(2, 2, l3s, 'l3s', -2.5, 2.5)
			plot_tensor(3, 2, l3u, 'l3u', -2.5, 2.5)
			plot_tensor(4, 2, w_f2, 'w_f2', -2.0, 2.0)
			
			fig.tight_layout()
			fig.canvas.draw()
			fig.canvas.flush_events()
			if animate: 
				time.sleep(0.1)
			initialized = True
