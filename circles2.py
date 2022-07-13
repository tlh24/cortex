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

## want to see what happens if we do PCA on these circles.
## what are the principle axes?  Can you predict the latents?
#N = 10000
#a = np.zeros((N, 1024))
#b = np.zeros((N, 3))
#for k in range(N):
	#out, (cx,cy,r) = make_circle()
	#a[k, :] = torch.reshape(out, (1024,)).numpy()
	#b[k,0] = cx
	#b[k,1] = cy
	#b[k,2] = r

#pca = PCA(n_components=10, svd_solver='full')
#pca.fit(a)
#pa = pca.transform(a)
#pa1 = np.concatenate((pa, np.ones((N,1))), 1)
#(ww,resid,rank,sing) = np.linalg.lstsq(pa1, b, rcond = None)
#pred = pa1 @ ww
#print(ww)

#fig, axs = plt.subplots(1, 3, figsize=(18, 8))
#for k in range(3):
	#axs[k].plot(b[1:100,k], 'b')
	#axs[k].plot(pred[1:100, k], 'r')
	#r2 = np.corrcoef(b[:,k], pred[:,k])
	#axs[k].set_title(f'r^2:%0.3f' % r2[0,1])

#plt.show()

# see if we can make a self-organizing hierarchical key-query-value network...
NL1 = 32*32
NL2 = 100
NL3 = 64
keys = zeros(NL2, NL1, device=torch_device)
gate = torch.rand((NL2, NL1), device=torch_device)
expand = torch.randn((NL3, 3), device=torch_device) / 10.0 # this should be automatic
ex_avg = torch.zeros(NL3, device=torch_device)
ex_var = torch.ones(NL3, device=torch_device)
keys2 = zeros(NL3, NL2, device=torch_device)
gate2 = torch.rand((NL3, NL2), device=torch_device)
eta = 0.01
eta2 = 0.01
bi = zeros(32*10, 32*10)
gi = zeros(32*10, 32*10)
bi2 = zeros(8*10, 8*10)
gi2 = zeros(8*10, 8*10)
l2a = zeros(NL2, device=torch_device)
l11 = ones(NL1, device=torch_device)
l21 = ones(NL2, device=torch_device)
l31 = ones(NL3, device=torch_device)

animate = True
plot_rows = 2
plot_cols = 3
if animate:
	plt.ion()
	fig, axs = plt.subplots(plot_rows, plot_cols, figsize=(35, 16))
im = [ [0]*plot_cols for i in range(plot_rows)]

# basically want to care about conjunctive inputs --
# but, rather than using a dot-product, rather use a key-query distance metric (different computational substrate..)
# problem is, when synaptic weights are zero, this is a 'don't care'
# for the dot product ...
# you don't have this with key-query.
# if the value is zero, then the distance is high when the input (query) is active but the hidden unit is not, and this causes a key update

for k in range(500):
	for i in range(2000):
		l1, latent = make_circle()
		l1 = l1.to(device=torch_device)
		l1 = reshape(l1, (32*32,))
		l2 = (outer(l21, l1) - keys) # * gate # FIXME
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
		l2c = clamp(l2b, 0.0, 1.0)
		#l2d = torch.gt(l2c, 0.15).float()

		keys = keys + eta*(outer(l2c, l11))*(outer(l21, l1) - keys)
		keys = clamp(keys, 0.0, 1.0)

		gate = gate + eta2*(outer(l2c, l1))
		gate = clamp(gate, 0.0, 1.0)
		ds = torch.sum(gate, 1)
		ds = clamp(ds, 200, 1e5)
		ds = torch.exp(0.0001234 * (ds-200)) -1
		gate = gate - outer(ds, l11)

		latent = torch.tensor(np.asarray(latent), device=torch_device).float()
		latent = torch.squeeze(latent)
		ex = expand @ latent
		ex_avg = 0.01*ex + 0.99*ex_avg
		ex_var = 0.005*((ex - ex_avg)**2) + 0.995*ex_var
		ex = (ex - ex_avg)/torch.sqrt(ex_var)
		ex = ex / torch.max(ex)
		ex = clamp(ex, 0.0, 1.0)
		# key2 update logic: 
		# if l2c is active, then move the key closer to ex
		keys2 = keys2 + eta*(torch.outer(l31, l2c))*(outer(ex, l21) - keys2)
		# gate2 logic: 
		# this sets the 'width' in effect of the matching
		# (it weights the K-Q match; smaller weights, more tolerance on the match. 
		gate2 = gate2 + eta2*(outer(ex, l2c))
		gate2 = clamp(gate2, 0.0, 1.0)
		ds = torch.sum(gate2, 1)
		ds = clamp(ds, 20, 1e5)
		ds = torch.exp(0.0001234 * (ds-20)) -1
		gate2 = gate2 - outer(ds, l21)

		# need to simulate forward and inverse graphics!
		# forward: latent -> ex -> select key(s)2 [64] -> l2_recon -> avg(keys)
		l2_recon = outer(ex, l21) - keys2
		l2_recon = l2_recon - min(l2_recon)
		l2_recon = torch.exp(-4.0 * l2_recon)

		# weighted average of first-layer keys
		l1_recon = torch.sum(outer(l2_recon, l11) * keys, 0) / torch.sum(l2_recon)
		l1_recon = reshape(l1_recon, (32, 32))

		# now inverse graphics:
		# l1 -> l2c -> keys2 -> expand^-1 (?!?) -> latent_recon
		# ah, I see.  If we don't want matrix inverses, need to invert with maps.
		# (also note that PCA works pretty well for predicting the latents..)
		# well, start with inverse.
		ex_recon = torch.sum(torch.outer(l31, l2c) * keys2, 1) / torch.sum(l2c)


	axs[0,0].clear()
	axs[0,0].plot(l2.cpu().numpy(), 'b')
	print(torch.mean(l2).cpu())
	axs[0,0].plot(l2a.cpu().numpy(), 'r')
	axs[0,0].plot(l2b.cpu().numpy(), 'k')
	axs[0,0].plot(l2c.cpu().numpy(), 'm')
	axs[0,0].set_title('blue = l2; red = l2a; black = l2b; magenta = l2c')
	# axs[0].plot(l2d.cpu().numpy(), 'g')
	
	axs[1,0].clear()
	axs[1,0].plot(ex.cpu().numpy(), 'b')
	axs[1,0].plot(ex_avg.cpu().numpy()/4.0, 'r')
	ex_std = torch.sqrt(ex_var)
	axs[1,0].plot(ex_std.cpu().numpy(), 'g')
	axs[1,0].set_title('blue = ex; red = ex_avg/4; green = ex_std')

	for i in range(NL2):
		row = (i//10)*32
		col = (i%10)*32
		v = reshape(keys[i,:], (32, 32))
		g = reshape(gate[i,:], (32, 32))
		bi[row:row+32, col:col+32] = v
		gi[row:row+32, col:col+32] = g
		
	# use the same topology as above to make interpretable.
	for i in range(NL2):
		row = (i//10)*8
		col = (i%10)*8
		v = reshape(keys2[:,i], (8, 8))
		g = reshape(gate2[:,i], (8, 8))
		bi2[row:row+8, col:col+8] = v
		gi2[row:row+8, col:col+8] = g

	if k == 0:
		data = np.linspace(0.0, 1.0, 320 * 320)
		data = np.reshape(data, (320, 320))
		im[0][1] = axs[0][1].imshow(data)
		im[0][2] = axs[0][2].imshow(data)
		data = np.linspace(0.0, 1.0, 80 * 80)
		data = np.reshape(data, (80, 80))
		im[1][1] = axs[1][1].imshow(data)
		im[1][2] = axs[1][2].imshow(data)

	data = bi.numpy()
	gata = gi.numpy()
	im[0][1].set_data(data)
	im[0][2].set_data(gata)
	
	data = bi2.numpy()
	gata = gi2.numpy()
	im[1][1].set_data(data)
	im[1][2].set_data(gata)

	fig.tight_layout()
	fig.canvas.draw()
	fig.canvas.flush_events()

