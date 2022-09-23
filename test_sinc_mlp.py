from __future__ import division, print_function
from time import time
from os.path import join
from itertools import product
import unittest
import sys
import os
import tempfile
import shutil
import gc
import pdb

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 180
import math
from math import sqrt

import torch

from torch import nn, optim
from torch.nn import init
from torch.nn import functional as F
from torch.autograd import Function

torch_device = 0
print("torch cuda devices", torch.cuda.device_count())
print("torch device", torch.cuda.get_device_name(torch_device))
torch.cuda.set_device(torch_device)
torch.set_default_tensor_type('torch.cuda.FloatTensor')

network = nn.Sequential(
	nn.Linear(2, 9),
	nn.LeakyReLU(0.2),
	nn.Linear(9, 1))

optimizer = optim.AdamW(network.parameters(), lr=1e-3, weight_decay=5e-4)
lossfunc = torch.nn.SmoothL1Loss()

slowloss = 0.0
niters = 5000
losses = np.zeros((2,niters))
for k in range(niters):
	x = torch.randn((2,))
	r = torch.sqrt(torch.sum(x**2))
	y = math.sin(r) / (r + 0.0001)
	x.grad = None
	y.grad = None
	network.zero_grad()
	predict = network(x)
	loss = lossfunc(y, predict)
	loss.backward()
	optimizer.step()
	slowloss = 0.99*slowloss + 0.01 * loss.detach()
	if k % 200 == 0 :
		print(f'{k} loss: {loss}; slowloss {slowloss}')
	losses[0,k] = loss
	losses[1,k] = slowloss

	if k % 50 == 0:
		# plot the approximation
		w2 = network[2].weight[0].detach()
		y = torch.zeros(100, 100)
		hid = torch.zeros(100, 100, 9)
		for i in range(100):
			x1 = 5.0 - i / 10.0
			for j in range(100):
				x2 = j / 10.0 - 5.0
				x = torch.tensor((x1, x2))
				y[i,j] = network(x)
				q = network[0].weight @ x + network[0].bias
				hid[i,j] = torch.clamp(q, 0.0, 1e6)

		figsize = (12, 6)
		fig, axs = plt.subplots(3, 4, figsize=figsize)
		im = axs[0,0].imshow(y.cpu().detach().numpy())
		plt.colorbar(im, ax=axs[0,0])

		for r in range(3):
			for c in range(3):
				j = r*3+c
				im = axs[r,c+1].imshow(torch.squeeze(hid[:,:,j]).cpu().detach().numpy())
				plt.colorbar(im, ax=axs[r,c+1])
				axs[r,c+1].set_title(str(j)+", "+str(float(w2[j])))
				axs[r,c+1].get_xaxis().set_visible(False)
				axs[r,c+1].get_yaxis().set_visible(False)

		plt.savefig(f'images_sinc/sinc_%04d.png'%(k/50))
		plt.close()

# print the weights.
print(network[0].weight)
print(network[0].bias)
print(network[2].weight)
w2 = network[2].weight[0].detach()
print(network[2].bias)

# then you need to:
# ffmpeg -stream_loop 4 -framerate 20 -i  sinc_%04d.png -c:v libx264 -r 30 -pix_fmt yuv420p sinc_mlp_4x_2.mp4
