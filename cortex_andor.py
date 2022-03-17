import os
import math
import numpy as np
import torch
import torchvision
import random
import pdb
import time
import matplotlib.pyplot as plt
from torch import clamp, matmul, outer, mul, add, sub, ones, zeros, reshape

def make_stim(s, x, y):
	# make a simple on-off vector pattern based on bits s, x, y
	# see notes for a depiction.
	out = zeros(3,3)
	if s:
		# shape 0 is L rotated 180 deg
		out[x, y] = 1
		out[x, y+1] = 1
		out[x+1, y] = 1
	else:
		out[x, y] = 1
		out[x+1, y+1] = 1
	return out

def randbool():
	return torch.randint(0, 2, (1,))

def test_stim():
	for i in range(10):
		s = randbool()
		x = randbool()
		y = randbool()
		o = make_stim(s, x, y)
		plt.imshow(o.numpy())
		plt.show()

# test_stim()
st, sx, sy = (randbool(), randbool(), randbool())
o = make_stim(st, sx, sy)
ov = reshape(o, (9,))
oo = outer(ov, ov) # product of all the boolean variables.
oo = reshape(oo, (81,))
wf = zeros(12, 81)
bi = wf @ oo
l2 = torch.sum(reshape(bi, (3, 4)), 1)
l2p = clamp(1.0 - l2, 0.0, 1.0)
l2b = torch.cat((l2, l2p), 0)
# we need at least three terms in the revese (alas)
# in biology the basal dendrites have far fewer synapses...
k = torch.reshape(outer(torch.reshape(outer(l2b, l2b), (36,)), l2b), (216,))
wb = zeros(9, 216)
l1i = wb @ k
print(l1i)

