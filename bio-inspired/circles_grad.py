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

for i in range(10):
	out, (cx, cy, ct) = make_line()
	print(cx, cy, ct)
	plt.imshow(out.numpy())
	plt.show()

# see if gradient descent can learn the invariance of a circle.
# given cx, cy, r and a parametric predictive model, can we estimate the constants?
# answer: yes, if they start with the correct signs,
# and we don't infer the constant per-pixel keys.

def random_seed(b):
	torch.manual_seed((os.getpid() * int(time.time())) % 123456789 + b)
	return b

def random_weights(cols):
	# return torch.clamp(torch.mul(torch.randn(cols), math.sqrt(2.0/cols)), -1.0, 1.0)
	return torch.tensor([1.5, 1.5, -1.2])

class KQlayer(nn.Module):
	def __init__(self):
		super().__init__()
		self.g_ = torch.nn.Parameter(data = random_weights(3), requires_grad=True)

		self.yc = outer(ones(32), torch.arange(0, 32))
		self.xc = outer(torch.arange(0, 32), ones(32))

	def forward(self, inp):
		xx = (self.xc - inp[0])**2 * self.g_[0]
		yy = (self.yc - inp[1])**2 * self.g_[1]
		x = torch.sqrt(xx + yy) + (inp[2]) * self.g_[2]
		x = torch.exp(-0.5*x*x)
		x = torch.reshape(x, (1024,))
		return x

	def printp(self):
		print(self.g_)


kql = KQlayer()
optimizer = optim.AdamW(kql.parameters(), lr = 1e-3, betas=(0.9, 0.99), weight_decay=2e-3)
mseloss = torch.nn.MSELoss()
random_seed(123043)
slowloss = 0.01

for k in range(10000):
	y, (cx, cy, cr) = make_circle()
	inp = torch.tensor(np.array([cx, cy, cr]))
	kql.zero_grad()
	p = kql.forward(inp)
	loss = mseloss(p, torch.reshape(y, (1024,)))
	loss.backward()
	optimizer.step()

	if k%100 == 99:
		slowloss = 0.99 * slowloss + 0.01 * loss.detach()
		print(f'{k} loss: {loss:.3e}; slowloss {slowloss:.3e}')

print(inp)
kql.printp()

fig, axs = plt.subplots(1, 2, figsize=(18, 10))
axs[0].imshow(y.detach().numpy())
p = torch.reshape(p, (32,32))
axs[1].imshow(p.detach().numpy())
plt.show()
