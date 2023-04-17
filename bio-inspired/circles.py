import os
import math
import numpy as np
import torch
import torchvision
import random
import pdb
import time
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 90
from torch import clamp, matmul, outer, mul, add, sub, ones, zeros, reshape

torch_device = 0
print("torch cuda devices", torch.cuda.device_count())
print("torch device", torch.cuda.get_device_name(torch_device))

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
	return out, (cx, cy, r)

#for k in range(10):
	#v,_ = make_circle()
	#plt.imshow(v.numpy())
	#plt.show()


# see if we can make a self-organizing hierarchical key-query-value network...
NL1 = 32*32
NL2 = 100
keys = zeros(NL2, NL1, device=torch_device)
eta = 0.01
bi = zeros(32*10, 32*10)
l2a = zeros(NL2, device=torch_device)
l21 = ones(NL2, device=torch_device)
l11 = ones(NL1, device=torch_device)

animate = True
if animate:
	plt.ion()
	fig, axs = plt.subplots(1, 2, figsize=(20, 10))
initialized = False
im = [0,0]

# basically want to care about conjunctive inputs --
# but, rather than using a dot-product, rather use a key-query distance metric (different computational substrate..)
# problem is, when synaptic weights are zero, this is a 'don't care'
# for the dot product ...
# you don't have this with key-query.
# if the value is zero, then the distance is high when the input (query) is active but the hidden unit is not, and this causes a key update

for k in range(500):
	for i in range(2000):
		l1, latent = make_circle() # what to do with these latents??
		l1 = l1.to(device=torch_device)
		l1 = reshape(l1, (32*32,))
		l2 = outer(l21, l1) - keys
		l2 = torch.sqrt(torch.sum(l2**2, 1))
		l2 = l2 + torch.randn(NL2, device=torch_device) * 0.05 # needs to be adaptive?
		l2 = l2 - torch.min(l2)
		l2b = torch.exp(-4.0 * l2) # flip the sign
		# max of l2b will always be 1.0
		l2a = l2b * 0.01 + l2a * 0.99
		l2b = l2b - l2a
		#l2c = torch.exp(l2b)
		#l2c = l2c - torch.sum(l2c)
		#l2c = l2b - torch.mean(l2b) # sparsify .. ?
		l2c = clamp(l2b, 0.0, 5.0)
		#l2d = torch.gt(l2c, 0.15).float()

		keys = keys + eta*(torch.outer(l2c, l11))*(outer(l21, l1) - keys)
		keys = clamp(keys, 0.0, 1.0)
		#ds = torch.sum(keys, 1)
		#ds = clamp(ds, 100, 1e5)
		#ds = torch.exp(-0.004 * (ds-100))
		#keys = keys * outer(ds, l11)

	axs[0].clear()
	axs[0].plot(l2.cpu().numpy(), 'b')
	print(torch.mean(l2).cpu())
	axs[0].plot(l2a.cpu().numpy(), 'r')
	axs[0].plot(l2b.cpu().numpy(), 'k')
	axs[0].plot(l2c.cpu().numpy(), 'm')
	# axs[0].plot(l2d.cpu().numpy(), 'g')

	for i in range(100):
		v = reshape(keys[i,:], (32, 32))
		row = (i//10)*32
		col = (i%10)*32
		bi[row:row+32, col:col+32] = v

	data = bi.numpy()
	if k == 0:
		data = np.linspace(0.0, 1.0, 320 * 320)
		data = np.reshape(data, (320, 320))
		im[1] = axs[1].imshow(data)

	im[1].set_data(data)

	fig.tight_layout()
	fig.canvas.draw()
	fig.canvas.flush_events()

