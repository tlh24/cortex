import os
import math
import numpy as np
from sklearn.decomposition import PCA
import torch
import torchvision
import random
import pdb
import time
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 120
from torch import clamp, matmul, outer, mul, add, sub, ones, zeros, reshape
from torch import sum as tsum

from torch import nn, optim
from torch.nn import init
from torch.nn import functional as F
from torch.autograd import Function

import umap

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

def make_line():
	# very inefficient line drawing program ;-)
	cx = (np.random.rand(1) - 0.5) * 20 + 16
	cy = (np.random.rand(1) - 0.5) * 20 + 16
	ct = np.random.rand(1) * 2 * 3.1415926
	# same, paint an anti-aliased line.
	x = outer(torch.arange(0, 32), ones(32))
	y = outer(ones(32), torch.arange(0, 32))
	x = x - cx
	y = y - cy
	vx = math.cos(ct)
	vy = math.sin(ct)
	dp = x * vx + y * vy
	d2 = (x - vx*dp)**2 + (y - vy*dp)**2
		# no square root, make the line a bit wider
	out = clamp(1.0 - d2, 0.0, 1.0)
	return out, (cx, cy, ct)

# see what a hierarchy of overlapping umaps does.
# without a doubt, this is very slow.
# need to make it parallel, faster, and on the GPU!
# ( if .. it works .. )

NN = 256
dataset = np.zeros((NN, 32, 32))
latents = np.zeros((NN, 3))
for i in range(NN):
	c, latent = make_circle()
	dataset[i, :, :] = c
	latents[i, :] = np.squeeze(np.array(latent))
# need to map the latents to RGB.
minn = np.outer(np.ones(NN), np.amin(latents, 0))
maxx = np.outer(np.ones(NN), np.amax(latents, 0))
rgb = (latents - minn) / (maxx - minn)

# try just doing UMAP on the data directly.
d = dataset
d = np.reshape(d, (NN, 32*32))
e = umap.UMAP(n_components=3, verbose=True).fit_transform(d)
#fig, axs = plt.subplots(2, 2, figsize=(18, 12))


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(e[:,0], e[:,1], e[:,2], c=rgb)
plt.show()

batch_size = 60000
#train_loader = torch.utils.data.DataLoader(
#torchvision.datasets.MNIST('files/', train=True, download=True,
				#transform=torchvision.transforms.Compose([
					#torchvision.transforms.ToTensor()
					#])),
				#batch_size=batch_size, shuffle=True, pin_memory=True)
train_loader = torch.utils.data.DataLoader(
torchvision.datasets.MNIST('files/', train=True, download=True,
				transform=torchvision.transforms.Compose([
					torchvision.transforms.RandomAffine(25.0, translate=(0.06,0.06), scale=(0.95,1.05), shear=4.0, interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
					torchvision.transforms.ToTensor()
					])),
				batch_size=batch_size, shuffle=True, pin_memory=True)
mnist = enumerate(train_loader)
batch_idx, (dataset, target) = next(mnist)
d = dataset
d = reshape(d, (60000, 28*28))
e = umap.UMAP(n_components=2, verbose=True).fit_transform(d)

rgb = np.zeros((60000, 3))
rgb[:,0] = 1.0 - np.clip(target / 4.5, 0.0, 1.0)
rgb[:,1] = np.clip(target / 4.5, 0.0, 1.0) - np.clip((target-4.5)/4.5, 0.0, 1.0)
rgb[:,2] = np.clip((target-4.5)/4.5, 0.0, 1.0)

#fig = plt.figure()
#ax = fig.add_subplot(projection='3d')
#ax.scatter(e[:,0], e[:,1], e[:,2], c=rgb)
plt.scatter(e[:,0], e[:,1], c=rgb)
plt.show()

# conclusion: UMAP can recover the latent 3D structure directly, reasonably robustly, and quickly. (with default parameters!)
# you do not need any hierarchy or whatnot,
# but you do need more samples than the number of dimensions.
# e.g. 2000 works, but 500 does not)
# from the map it is somewhat trivial to invert (map supervised signal to embedded dimensions, do interpolation of the examples in embedded dim)
# but not totally clear that this is an 'algorithm' per se.
# instead, it's a detailed bidirectional map, which *could* concevably be used to make an 'algorithm'
# better would be to use the umap algorithm to discover the compression / decompression algorithm (??)
# make the map then compress it.

rng = np.random.default_rng()
# do all the umaps at once -- like a convolution

def do_umap(lin, ch_in, ch_out, Q): 
	lout = np.zeros((NN, Q, Q, ch_out))
	dout = np.zeros((NN*Q*Q, 25*ch_in))
	drgb = np.zeros((NN*Q*Q, 3))

	i = 0
	for r in range(Q):
		for c in range(Q):
			d = lin[:, r:r+5, c:c+5, :]
			d = np.reshape(d, (NN, 25*ch_in))
			dout[i*NN:i*NN+NN, :] = d
			drgb[i*NN:i*NN+NN, :] = rgb
			i = i+1
			
	threshold = 64000
	if dout.shape[0] > threshold:
		# transform is single-threaded, so this is 'not that much' faster.
		dout_sub = rng.choice(dout, threshold, replace=False)
		print('umap fit', Q, threshold, dout.shape)
		trans = umap.UMAP(n_components=ch_out, verbose=True).fit(dout_sub)
		print('umap transform')
		e = trans.transform(dout)
	else: 
		print('umapping', Q, dout.shape)
		e = umap.UMAP(n_components=ch_out, verbose=True).fit_transform(dout)

	if Q < 6: 
		fig = plt.figure()
		ax = fig.add_subplot(projection='3d')
		ax.scatter(e[:,0], e[:,1], e[:,2], c=drgb)
		plt.show()

	i = 0
	for r in range(Q):
		for c in range(Q):
			lout[:, r, c, :] = e[i*NN:i*NN+NN, :]
			i = i+1
	return lout
	
l2 = do_umap(np.expand_dims(dataset, -1), 1, 3, 32-4)
l3 = do_umap(l2, 3, 3, 24)
l4 = do_umap(l3, 3, 3, 20)
l5 = do_umap(l4, 3, 3, 16)
l6 = do_umap(l5, 3, 3, 12)
l7 = do_umap(l6, 3, 3, 8)
l8 = do_umap(l7, 3, 3, 4)
# do a final global compression.
l8s = np.reshape(l8, (NN, 4*4*3))
e = umap.UMAP(n_components=3).fit_transform(l8s)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(e[:,0], e[:,1], e[:,2], c=rgb)
plt.show()

# it looks like crap compared to the simple, and much much faster, direct application of umap to the data.
# Therefore, this is not a sensible way of making the algorithm hierarchical.
# That said: many complaints about "graph is not fully connected"... so umap itself might not be working.
