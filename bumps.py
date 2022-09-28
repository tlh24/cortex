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


# gaussian bumps
def make_stim(x, y):
	xs = outer(ones(32), torch.linspace(0, 1.0, 32))
	ys = outer(torch.linspace(0, 1.0, 32), ones(32))
	xs = torch.pow(xs - x, 2.0) / 0.003
	ys = torch.pow(ys - y, 2.0) / 0.003
	r = torch.exp(-0.5*(xs + ys))
	return r

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

#for i in range(10):
	#l1e = make_stim(torch.rand(1), torch.rand(1))
	#plt.imshow(l1e.cpu().numpy())
	#plt.show()
QQ = 32
P = 100
l11 = ones((1024,))
lp1 = ones((P,))
lpa = ones((P,))
lpvar = ones((P,))
eta = 0.1
eta2 = 0.05
keys = zeros((P, 1024))
di = zeros(QQ*10, QQ*10)
ri = zeros(QQ*10, QQ*10)
gi = zeros(QQ*10, QQ*10)
reys = torch.zeros((1024, P))
gate = torch.ones((1024, P))

plot_rows = 2
plot_cols = 3
plt.ion()
fig, axs = plt.subplots(plot_rows, plot_cols, figsize=(35, 16))
im = [ [0]*plot_cols for i in range(plot_rows)]

for i in range(10000):
	x = torch.rand(1)
	y = torch.rand(1)
	# l1e = make_stim(x, y)
	l1e, latents = make_circle()
	l1e = reshape(l1e, (1024,))
	lp = keys @ l1e / tsum(l1e)  # weighted average.  signed.
	lp = lp + torch.randn(P) * 0.12
	lp = clamp(lp, -1.0, 10.0)
	lp = lp - torch.mean(lp)
	lpa = lp * 0.01 + lpa * 0.99
	lp = lp - lpa
	lpvar = lpvar + eta*(lp**2 - lpvar)

	# l2 distance metric from reys
	l1i = (outer(l11, lp) - reys) * gate
	l1i = tsum(l1i**2, 1)
	l1i = torch.exp(-0.5 * l1i)
	l1u = l1e - l1i

	# now we have an error, but updating the keys is not quite so straightforward as w a linear projection, b/c reys is not monotonic
	# the old logic was to do hebbian learning: change keys to minimize error.
	# this works here bc that part is linear.
	# keys = keys + eta*outer(lp, l1u) # might need 'advanced' Hebbiain.
	keys = keys + eta*(outer(lp, l11) - keys)*outer(lp1, l1e)
	# constraint: activity cant grow too large : gain control.
	ds = torch.exp((clamp(lp, 2.5, 1e6) - 2.5) * -0.04)
	keys = keys * outer(ds, l11)
	# constraint: increse gain in the case of low variance.
	ds = torch.exp((0.1 - clamp(lpvar, 0.0, 0.1))* 0.04)
	keys = keys * outer(ds, l11) # this helps a lot

	# the reverse, I think we just have to track lp (?)
	# the learning rate will matter here.
	# "treat as linear but implement with distance fn"
	reys = reys + eta*(outer(l11, lp) - reys)*(outer(l1e, lp1))

	# if error and lp are correlated, increase gate.
	gate = gate + eta2*outer(l1u, lp)
	gate = clamp(gate, 0.0, 2.0)
	# but default decay to 0.5
	# gate = gate + 0.1*eta2*(0.5-gate) negatively impacts performance


	if i == 0:
		data = np.linspace(-1.0, 1.0, (QQ*10 * QQ*10))
		data = np.reshape(data, (QQ*10, QQ*10))
		im[0][0] = axs[0][0].imshow(data)
		im[0][1] = axs[0][1].imshow(data)
		im[1][0] = axs[1][0].imshow(data)

		data = np.linspace(-1.0, 1.0, QQ*QQ)
		data = np.reshape(data, (QQ,QQ))
		im[0][2] = axs[0][2].imshow(data)
		im[1][2] = axs[1][2].imshow(data)

		axs[0][0].set_title('keys forward')
		axs[0][1].set_title('keys reverse')
		axs[1][0].set_title('gate reverse')
		axs[0][2].set_title('l1e')
		axs[1][2].set_title('l1i')

	if i%30 == 29:

		axs[1,1].clear()
		axs[1,1].plot(lp.cpu().numpy())

		# reformat so that each P cell has the 1024 keys organized by space.
		for j in range(100):
			row = (j//10)*QQ
			col = (j%10)*QQ
			v = reshape(keys[j,:], (QQ, QQ))
			di[row:row+QQ, col:col+QQ] = v
			# and the reverse keys are similarly organized
			v = reshape(reys[:,j], (QQ, QQ))
			ri[row:row+QQ, col:col+QQ] = v
			v = reshape(gate[:,j], (QQ, QQ))
			gi[row:row+QQ, col:col+QQ] = v

		im[0][0].set_data(di.cpu().numpy())
		im[0][1].set_data(ri.cpu().numpy())
		im[1][0].set_data(gi.cpu().numpy())

		im[0][2].set_data( (reshape(l1e, (32,32))).cpu().numpy() )
		im[1][2].set_data( (reshape(l1i, (32,32))).cpu().numpy() )

		fig.tight_layout()
		fig.canvas.draw()
		fig.canvas.flush_events()

	print(i)
