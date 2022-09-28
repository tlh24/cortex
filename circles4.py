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
MNIST = True
if MNIST:
	train_loader = torch.utils.data.DataLoader(
	torchvision.datasets.MNIST('files/', train=True, download=True,
										transform=torchvision.transforms.Compose([
											torchvision.transforms.RandomAffine(25.0, translate=(0.06,0.06), scale=(0.95,1.05), shear=4.0),
											torchvision.transforms.ToTensor()
										])),
	batch_size=60000, shuffle=True, pin_memory=True)

	QQ = 28
else:
	QQ = 32

NL1 = QQ*QQ
NL2 = 100
NL3 = 3
keys = zeros((NL2, NL1), device=torch_device)
gvar = ones((NL2, NL1), device=torch_device)
gate = ones((NL2, NL1), device=torch_device)

keys2 = zeros((NL3, NL2), device=torch_device)
gvar2 = ones((NL3, NL2), device=torch_device)
gate2 = ones((NL3, NL2), device=torch_device)
eta = 0.01
eta2 = 0.01
bi = zeros(QQ*10, QQ*10)
gi = zeros(QQ*10, QQ*10)
bi2 = zeros(1*10, 3*10)
gi2 = zeros(1*10, 3*10)
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

if not MNIST:
	NREP = 1399
	circles = zeros((NREP, 1024), device=torch_device)
	latents = zeros((NREP, 3), device=torch_device)
	for k in range(NREP):
		l1, latent = make_circle()
		l1 = l1.to(device=torch_device)
		circles[k,:] = torch.reshape(l1, (1024,))
		latent = torch.tensor(np.asarray(latent), device=torch_device).float()
		latent = torch.squeeze(latent)
		latents[k,:] = latent
else:
	mnist = enumerate(train_loader)
	batch_idx, (indata, intarget) = next(mnist)
	indata = torch.reshape(indata, (60000, 784)).to(torch_device)
	latents = torch.reshape(intarget, (60000, 1)).to(torch_device)
	# technically this one value should be enough..
	# but we'll need to engineer some ** SLACK VARIABLES **

activations = torch.zeros(10, device=torch_device)

for k in range(100):
	for i in range(600):
		indx = i+k*600
		if MNIST:
			l1 = indata[indx, :]
		else:
			l1 = circles[indx%NREP, :]
		l2 = (outer(l21, l1) - keys) * gate # vmap would be beter. or einsum.
		l2 = torch.sqrt(tsum(l2**2, 1))
		# l2 = l2 - torch.min(l2)
		#l2 = l2 / (torch.max(l2) + 0.11)
		#l2b = torch.exp(-0.5 * l2) # flip the sign
		## max of l2b will always be 1.0
		#l2a = l2b * 0.02 + l2a * 0.98
		#l2c = l2b - l2a
		#l2c = l2c + torch.randn(NL2, device=torch_device) * 0.08 # needs to be adaptive?
		#l2c = l2c / (torch.max(l2c) + 0.11)
		#l2c = torch.clamp(l2c, 0.0, 1.0)
		l2 = l2 + torch.randn(NL2, device=torch_device) * 0.05
		l2 = l2 - torch.min(l2)
		l2b = torch.exp(-0.5 * l2) # flip the sign
		# remove both the population mean and per-unit time-integrated mean
		l2b = l2b - torch.mean(l2b)
		l2a = l2b * 0.01 + l2a * 0.99
		l2b = l2b - l2a
		l2c = l2b / (torch.max(l2b) + 0.01) # unnecessary??
		#l2c = l2b
		l2c = clamp(l2c, 0.0, 5.0)

		# keys
		keys = keys + eta*(outer(l2c, l11))*\
			((outer(l21, l1) - keys))
		# keys = clamp(keys, 0.0, 1.0) # ??? depends on source distribution

		vara = ((outer(l21, l1) - keys)**2) # unweighted variance
		gvar = gvar + eta2*outer(l2c, l11)*(vara - gvar) # weight here
		#gvar = gvar + eta2*(vara - gvar)
		gate = 1.0 / torch.sqrt(gvar + 0.01)
		
		#gate = gate + eta2*(outer(l2c, l1))
		#gate = clamp(gate, 0.0, 1.0)
		#ds = torch.sum(gate, 1)
		#ds = clamp(ds, 200, 1e5)
		#ds = torch.exp(0.0001234 * (ds-200)) -1
		#gate = gate - outer(ds, l11)

		if not MNIST:
			latent = latents[indx%NREP, :]
		else:
			latent = latents[indx, :]
		# key2 update logic: 
		# if l2c is active, then move the key closer to latent
		keys2 = keys2 + eta*(outer(l31, l2c))*(outer(latent, l21) - keys2)
		# gate2 logic: 
		# this sets the effective 'width' of the matching
		# (it weights the K-Q match; larger weights, tighter match tolerance.)
		# thus should be inversely proportional to variance

		vara2 = outer(l31, l2c)*((outer(latent, l21) - keys2)**2)
		gvar2 = 0.995*gvar2 + 0.005*vara2
		gate2 = 1.0 / torch.sqrt(gvar2 + 0.01)
		#match = (outer(latent, l21) - keys2)**2
		#match = torch.exp(-4.0*match) # seems unprinipled; maybe should just use Gaussians
		#gate2 = gate2 + eta2*(outer(l31, l2c) * match
		#gate2 = clamp(gate2, 0.0, 1.0)
		#ds = tsum(gate2, 1)
		#ds = clamp(ds, 20, 1e5)
		#ds = torch.exp(0.0001234 * (ds-20)) -1
		#gate2 = gate2 - outer(ds, l21)

		# need to simulate forward and inverse graphics!
		# forward: latent -> select key(s)2 [64] -> l2_recon -> avg(keys)
		l2_recon = ((outer(latent, l21) - keys2) * gate2)**2
		l2_recon = tsum(l2_recon, 0)
		l2_recon = l2_recon - torch.min(l2_recon)
		l2_recon = torch.exp(-0.5 * l2_recon)
		l1_recon = tsum(outer(l2_recon, l11) * keys, 0) / tsum(l2_recon)

		# now inverse graphics:
		# l1 -> l2c -> keys2 -> latent_recon
		latent_recon = tsum(outer(l31, l2c) * keys2, 1) / tsum(l2c)

		l1_recon = reshape(l1_recon, (QQ,QQ))

		activations[int(latent)] = activations[int(latent)] + tsum(l2c)

	axs[0,0].clear()
	l2m = l2 - torch.min(l2)
	l2m = l2m / (torch.max(l2m) + 0.01)
	axs[0,0].plot(l2m.cpu().numpy(), 'b')
	print(torch.mean(l2).cpu())
	axs[0,0].plot(l2a.cpu().numpy(), 'r')
	axs[0,0].plot(l2b.cpu().numpy(), 'm')
	axs[0,0].plot(l2c.cpu().numpy(), 'k')
	# axs[0,0].plot(l2d.cpu().numpy(), 'g')
	axs[0,0].set_title('blue = l2; red = l2a; magenta = l2b; black = l2c')

	for i in range(100):
		row = (i//10)*QQ
		col = (i%10)*QQ
		v = reshape(keys[i,:], (QQ, QQ))
		g = reshape(gvar[i,:], (QQ, QQ))
		bi[row:row+QQ, col:col+QQ] = v
		gi[row:row+QQ, col:col+QQ] = g
		
	# use the same topology as above to make interpretable.
	for i in range(100):
		row = (i//10)*1
		col = (i%10)*3
		v = keys2[:,i]
		g = gvar2[:,i]
		bi2[row, col:col+3] = v
		gi2[row, col:col+3] = g

	if k == 0:
		data = np.linspace(0.0, 1.0, QQ*10 * QQ*10)
		data = np.reshape(data, (QQ*10, QQ*10))
		
		im[0][1] = axs[0][1].imshow(data)
		im[0][2] = axs[0][2].imshow(data)
		im[1][0] = axs[1][0].imshow(data)
		data = np.linspace(0.0, float(QQ), 10 * 30)
		data = np.reshape(data, (10, 30))
		im[1][1] = axs[1][1].imshow(data)
		data = np.linspace(0.0, 1.0, 10 * 30)
		data = np.reshape(data, (10, 30))
		im[1][2] = axs[1][2].imshow(data)
		
		axs[0][1].set_title('keys (l1->l2)')
		axs[0][2].set_title('gvar (l1->l2)')
		axs[1][1].set_title('keys (l3->l2)')
		axs[1][2].set_title('gvar (l3->l2)')

	data = bi.numpy()
	gata = gi.numpy()
	im[0][1].set_data(data)
	im[0][2].set_data(gata)
	
	data = bi2.numpy()
	gata = gi2.numpy()
	im[1][1].set_data(data)
	im[1][2].set_data(gata)

	data = l1_recon.cpu().numpy()
	im[1][0].set_data(data)

	fig.tight_layout()
	fig.canvas.draw()
	fig.canvas.flush_events()

	print('latent / latent recon', latent, latent_recon)

plt.close()
plt.ioff()
plt.plot(activations.cpu().numpy())
plt.show()
