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
torch.cuda.set_device(torch_device)
torch.set_default_tensor_type('torch.cuda.FloatTensor')

def make_circle():
	accept = False
	while (not accept):
		cx = torch.randn(1) * 12 + 16
		cy = torch.randn(1) * 12 + 16
		r = torch.rand(1) * 16 + 4
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
											torchvision.transforms.RandomAffine(25.0, translate=(0.06,0.06), scale=(0.95,1.05), shear=4.0, interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
											torchvision.transforms.ToTensor()
										])),
	batch_size=60000, shuffle=True, pin_memory=True)

	QQ = 28
else:
	QQ = 32

NL1 = QQ*QQ
NL2 = 256 # minimum 100
NL3 = 3
keys = torch.zeros((NL2, NL1))
keys2 = torch.zeros((NL1, NL2))

eta = 0.1
eta2 = 0.01
bi = zeros(QQ*10, QQ*10)
gi = zeros(QQ*10, QQ*10)
bi2 = zeros(QQ*10, QQ*10)
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
		latent = torch.tensor(torch.asarray(latent), device=torch_device).float()
		latent = torch.squeeze(latent)
		latents[k,:] = latent
else:
	torch.set_default_tensor_type('torch.FloatTensor')
	mnist = enumerate(train_loader)
	batch_idx, (indata, intarget) = next(mnist)
	indata = torch.reshape(indata, (60000, 784)).to(torch_device)
	latents = torch.reshape(intarget, (60000, 1)).to(torch_device)
	# technically this one value should be enough..
	# but we'll need to engineer some ** SLACK VARIABLES **
	torch.set_default_tensor_type('torch.cuda.FloatTensor')

activations = torch.zeros(10, device=torch_device)

# different strategy here: 
# l2 is the weighted average of the randomly-initialized keys. 
# l1i is the weighted average of randomly initialized reverse keys. 
for k in range(10000):
	for i in range(600):
		indx = i+k*600
		if MNIST:
			l1e = indata[indx%60000, :]
		else:
			l1e = circles[indx%NREP, :]
		l2 = keys @ l1e # linear projection
		l2 = l2 / (tsum(l1e, 0) + 1.0) # but treat it like a key average
		l2 = l2 + 0.05*torch.randn((NL2,))
		l2 = clamp(l2, 0.0, 1.0)
		l1i = keys2 @ l2
		l1i = l1i / (tsum(l2, 0) + 1.0) # speeds & stabilizes learning.
		# also .. the stimuli is always [0 1]
		l1u = l1e - l1i

		# this is not the 'advanced' Hebbian learning..
		# either cube or 1 power work here.
		# 1 converges very very quickly (larger updates) to so-so results.
		keys = keys + eta * ((outer(l2, l1u))**1)
		keys2 = keys2 + eta * ((outer(l1u, l2))**1)
		
		#keys = clamp(keys, 0.0, 1.0)
		#keys2 = clamp(keys2, 0.0, 1.0)

	axs[0,0].clear()
	l2m = l2 - torch.min(l2)
	l2m = l2m / (torch.max(l2m) + 0.01)
	axs[0,0].plot(l2.cpu().numpy(), 'b')
	print(torch.mean(l2).cpu())
	#axs[0,0].plot(l2a.cpu().numpy(), 'r')
	#axs[0,0].plot(l2b.cpu().numpy(), 'm')
	#axs[0,0].plot(l2c.cpu().numpy(), 'k')
	# axs[0,0].plot(l2d.cpu().numpy(), 'g')
	axs[0,0].set_title('blue = l2; red = l2a; magenta = l2b; black = l2c')

	for i in range(100):
		row = (i//10)*QQ
		col = (i%10)*QQ
		v = reshape(keys[i,:], (QQ, QQ))
		bi[row:row+QQ, col:col+QQ] = v
		v = reshape(keys2[:,i], (QQ, QQ))
		bi2[row:row+QQ, col:col+QQ] = v
		#gi[row:row+QQ, col:col+QQ] = g

	if k == 0:
		data = np.linspace(-1.0, 1.0, QQ*10 * QQ*10)
		data = np.reshape(data, (QQ*10, QQ*10))
		im[0][1] = axs[0][1].imshow(data, cmap='seismic')
		im[0][2] = axs[0][2].imshow(data, cmap='seismic')
		
		data = np.linspace(0.0, 1.0, QQ * QQ)
		data = np.reshape(data, (QQ*1, QQ*1))
		im[1][0] = axs[1][0].imshow(data)
		im[1][1] = axs[1][1].imshow(data)
		im[1][2] = axs[1][2].imshow(data)
		
		axs[0][1].set_title('keys (l1->l2)')
		axs[0][2].set_title('keys (l2->l1)')
		axs[1][0].set_title('l1e')
		axs[1][1].set_title('l1i')
		axs[1][2].set_title('l1u')

	data = bi.cpu().numpy()
	gata = bi2.cpu().numpy()
	im[0][1].set_data(data)
	im[0][2].set_data(gata)
	
	l1ec = np.reshape(l1e.cpu().numpy(), (QQ, QQ))
	l1ic = np.reshape(l1i.cpu().numpy(), (QQ, QQ))
	l1uc = np.reshape(l1u.cpu().numpy(), (QQ, QQ))
	im[1][0].set_data(l1ec)
	im[1][1].set_data(l1ic)
	im[1][2].set_data(l1uc)

	#data = l1_recon.cpu().numpy()
	#im[1][0].set_data(data)

	fig.tight_layout()
	fig.canvas.draw()
	fig.canvas.flush_events()

#plt.close()
#plt.ioff()
#plt.plot(activations.cpu().numpy())
#plt.show()
