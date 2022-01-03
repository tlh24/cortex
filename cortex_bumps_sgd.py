
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
import time

import math
from math import sqrt
from math import cos
import numpy as np
from numpy.random import randn
from numpy.random import rand
import matplotlib.pyplot as plt
import pdb
import numba
from numba import jit
#from multiprocessing import Pool

import torch

from torch import nn, optim
from torch.nn import init
from torch.nn import functional as F
from torch.autograd import Function

batches = 64
nrc = 8 # 
nl2 = 4

torch.set_default_tensor_type('torch.FloatTensor')
# torch.set_default_tensor_type('torch.cuda.FloatTensor')
gdevice = torch.device('cuda')

def check_memory():
	# for tracking down memory leaks.. 
	gc.collect()
	for obj in gc.get_objects():
		try:
			if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
				print(type(obj), obj.size())
		except:
			pass
		
def clamp(num, min_value, max_value):
   return max(min(num, max_value), min_value)

def timing(f):
	def wrap(*args, **kwargs):
		time1 = time.time()
		ret = f(*args, **kwargs)
		time2 = time.time()
		print('{:s} function took {:.3f} ms'.format(f.__name__, (time2-time1)*1000.0))

		return ret
	return wrap


def random_weights3(a, b, c):
	 return torch.clamp(torch.mul(torch.randn(batches, a, b, c), math.sqrt(2.0 / (b * c))), -1.0, 1.0)
	 # this is a guess! 
	 
def random_weights(rows, cols):
	return torch.clamp(torch.mul(torch.randn(batches, rows, cols), math.sqrt(2.0/cols)), -1.0, 1.0)

def imshow_(t, fname):
	t = t.detach().cpu()
	plt.imshow(t)
	plt.savefig(fname)
	plt.close()

class PassGate(nn.Module): 
	def __init__(self, out_dim, in_dim, rc_dim):
		super().__init__()
		self.w_ = torch.nn.Parameter(data = random_weights3(out_dim, in_dim, rc_dim), requires_grad=True)
		self.b_ = torch.nn.Parameter(data = torch.zeros(batches, out_dim), requires_grad=True)
		self.relu = torch.nn.LeakyReLU(0.2)
		
	def update_rc(self, rc):
		self.w_.data = torch.clamp(self.w_, -1.0, 1.0)
		# w_ is out x in x rc; so to do broadcast / batched mm, 
		# need to make rc batch x 1 x rc x 1
		# to yield a batch x out x in x 1
		rc = rc[:, None, :, None] # batch wise reduction
		# adding relu allows RC to 'shut off' segments of dendrites (weights). 
		self.w2 = (torch.squeeze(torch.matmul(self.w_, rc)))
	
	def forward(self, inp):
		inp = inp[:,:,None] # so we do a reduction 
		return torch.squeeze(torch.matmul(self.w2, inp)) + self.b_
	
	
class PGnet(nn.Module):
	def __init__(self):
		super().__init__()
		self.fpg = PassGate(nl2, 41, nrc)
		self.rpg = PassGate(41, nl2, nrc)
		self.pgctrl = torch.nn.Linear(41+nl2+41, nrc)
		self.pgrelu = torch.nn.LeakyReLU(0.2)
		self.pgsoftmax = torch.nn.Softmax(dim=0)
		# self.pgctrl = torch.nn.Parameter(data = random_weights(41+16, nrc), requires_grad=True)
		self.rc = torch.zeros(batches, nrc).to(gdevice)
		self.l2 = torch.zeros(batches, nl2).to(gdevice)
		self.l1i = torch.zeros(batches, 41).to(gdevice)
		self.l2softmax = torch.nn.Softmax(dim=0)
		self.rcblurweight = torch.zeros((batches,nrc,nrc))
		for k in range(nrc):
			self.rcblurweight[:,k,k] = 0.8
		for k in range(nrc-1):
			self.rcblurweight[:,k,k+1] = 0.1
			self.rcblurweight[:,k+1,k] = 0.1
		self.rcblurweight = self.rcblurweight.to(gdevice)
		self.l2blurweight = torch.zeros((batches,nl2,nl2))
		for k in range(nl2):
			self.l2blurweight[:,k,k] = 0.8
		for k in range(nl2-1):
			self.l2blurweight[:,k,k+1] = 0.1
			self.l2blurweight[:,k+1,k] = 0.1
		self.l2blurweight = self.l2blurweight.to(gdevice)
		
	def forward(self, inp):
		# run the network for several steps to reach equilibrium
		# this of course is prone to positive feedback instability! 
		l2old = self.l2
		l1i = self.l1i
		rcold = self.rc
		# with torch.no_grad():
		for k in range(2):
			rcin = torch.cat((inp, l2old, inp-l1i),dim=1)
			rc = self.pgrelu(self.pgctrl(rcin))
			rc = 0.4*rc + 0.6*rcold
			# rc = self.pgsoftmax(rc)
			rc = torch.clamp(rc, -0.2, 1.0)
			# imshow_(rc,'rc_before.png')
			for b in range(3):
				rc = torch.squeeze(torch.matmul(self.rcblurweight, rc[:,:,None]))
			# imshow_(rc,'rc_after.png')
			self.fpg.update_rc(rc) 
			self.rpg.update_rc(rc)
			# l2 = self.l2softmax(self.fpg.forward(inp))
			l2 = self.fpg.forward(inp) * 0.10 + 0.9 * l2old
			l2 = torch.clamp(l2, -0.2, 1.0)
			# imshow_(l2,'l2_before.png')
			for b in range(2):
				l2 = torch.squeeze(torch.matmul(self.l2blurweight, l2[:,:,None]))
			# imshow_(l2,'l2_after.png')
			l1i = self.rpg.forward(l2)
			l1i = torch.clamp(l1i, -0.2, 1.0) 
		
		self.l2 = l2.detach()
		self.l1i = l1i.detach()
		self.rc = rc.detach()
		
		return(l1i, l2.detach(), rc.detach())
	
	def forward_clamp_rc(self, inp, rcin):
		# this assumes a already-calculated one-hot location code 
		# eg if the animal is moving its eyes and has the reafferent
		# determine if this results in quick and stable prediction.
		l2 = self.l2
		l1i = self.l1i
		rc = rcin.detach().to(gdevice)
		for k in range(1):
			self.fpg.update_rc(rc) 
			self.rpg.update_rc(rc)
			# l2 = self.l2softmax(self.fpg.forward(inp))
			l2 = torch.clamp(self.fpg.forward(inp), -0.1, 1.0)
			# pdb.set_trace()
			l2 = torch.squeeze(torch.matmul(self.l2blurweight, l2[:,:,None]))
			l1i = self.rpg.forward(l2)
			l1i = torch.clamp(l1i, 0.0, 1.0)
				
		# now run once to update the weights with the equilibriated activities. 
		#l2 = l2.detach() # might not be necessary..
		#l1i = l1i.detach()
		
		self.fpg.update_rc(rc) 
		self.rpg.update_rc(rc)
		l2 = torch.clamp(self.fpg.forward(inp), -0.1, 1.0)
		# l2 = self.l2softmax(self.fpg.forward(inp))
		l1i = self.rpg.forward(l2)
		l1i = torch.clamp(l1i, 0.0, 1.0)
		
		self.l2 = l2.detach()
		self.l1i = l1i.detach()
		
		return(l1i, l2.detach())
	
	def forward_weights(self):
		return self.fpg.w_
	

# field is a set of sensors, -20 .. 20
# objects are cosine bumps that move through the field 
# with a fixed velocity. 

# this is a real bottleneck.. use JIT
@jit(nopython=True)
def render_batch(input_objs):
	output = np.zeros((batches, 41))
	for i in range(batches):
		position = input_objs[i, 0]
		velocity = input_objs[i, 1]
		width = input_objs[i, 2]
		mod = input_objs[i, 3]
		for j in range(-20,21):
			val = 0.0
			if j >= position - width and j <= position + width:
				val = (np.cos((j-position) / width * 3.1415926) + 1.0)/2.0 
				val = val * ((np.sin((j-position) / mod * 3.1415926) * 0.25) + 0.75)
			output[i,j+20] = val
	return output # torch.from_numpy(output).to(torch.float)

@jit(nopython=True)
def new_obj():
	scl = 1.0
	if randn() < 0.0:
		scl = -1.0
	position = 26.0 * scl
	velocity = (1.75 + randn() * 0.35) * scl * -1.0
	width = 7.0 + randn() * 2.0
	mod = (rand(1)[0] + 1) * 2.25
	return np.asarray([position, velocity, width, mod])

def random_seed(b):
	torch.manual_seed((os.getpid() * int(time.time())) % 123456789 + b)
	return b

@jit(nopython=True)
def move_batch(input_objs):
	for i in range(batches):
		pos = input_objs[i, 0]
		velocity = input_objs[i, 1]
		width = input_objs[i, 2]
		mod = input_objs[i, 3]
		pos += velocity
		# allow the objects to move more fully "off-screen" to avoid edge effects. 
		if pos < -27 or pos > 27:
			input_objs[i,:] = new_obj()
		else:
			input_objs[i,:] = np.array((pos, velocity, width, mod))
	return input_objs

def obj_to_rc(obj_):
	# convert the object position to a one-hot RC activation.
	(pos,vel,width) = obj_
	# make the desired one-hot location. 
	rcdes = torch.zeros(nrc); 
	rci = ((pos + 20) / 40.0) * 13.0
	rci = clamp(rci, 0.0, 13.001)
	rcil = int(math.floor(rci))
	res = rci - rcil
	rcdes[rcil] = 1.0 - res
	if rci < 13:
		rcdes[rcil+1] = res
	return rcdes

def obj_to_rc_batch(input_objs):
	output = torch.zeros(batches, nrc)
	for ind,res in enumerate(map(obj_to_rc, input_objs)):
		output[ind, :] = res
	return output

k = 0
slowloss = 0.0
last_update = time.time()

# pool = Pool(batches) #defaults to number of available CPU's

pgn = PGnet()
pgn.to(device=gdevice)
optimizer = optim.AdamW(pgn.parameters(), lr = 1e-4, betas=(0.9, 0.99), weight_decay=2e-3)
mseloss = torch.nn.MSELoss()
random_seed(123043)
obj_ = np.zeros((batches, 4))
for k in range(batches):
	obj_[k, :] = new_obj()
monitor_every = 4e6
monitor_view = 80

while True:
	t1 = time.time()
	vis = render_batch(obj_)
	vis = torch.from_numpy(vis).to(torch.float)
	t1a = time.time()
	obj_ = move_batch(obj_)
	t1b = time.time()
	vis = vis.to(device=gdevice)
	t2 = time.time()
	pgn.zero_grad()
	t3 = time.time()
	# rc = obj_to_rc_batch(obj_)
	(pred, l2, rc) = pgn.forward(vis)
	t4 = time.time()
	loss = mseloss(pred, vis)
	t5 = time.time()
	loss.backward()
	t6 = time.time()
	optimizer.step()
	t7 = time.time()
	
	pred2 = pred.detach(); 
	
	slowloss = 0.99 * slowloss + 0.01 * loss.detach()
		# be careful to detach the loss!  otherwise the gradient graph grows endlessly! 
	if k % 200 == 0 : 
		upd = time.time() - last_update
		print(f'{k} loss: {loss:.3e}; slowloss {slowloss:.3e} time {upd} / {upd/200}')
		print(f'\trender {(t1a-t1):.3e}; move {(t1b-t1a):.3e}; to_device {(t2-t1b):.3e}; zero_grad {(t3-t2):.3e}; forward {(t4-t3):.3e}; loss {(t5-t4):.3e}; backward {(t6-t5):.3e}; optimizer {(t7-t6):.3e}')
		last_update = time.time()
		gc.collect()
		
	if k % monitor_every < monitor_view and k % monitor_every >= 0: 
		plt.plot(range(41), vis[0,:].cpu().detach().numpy(), 'b')
		plt.plot(range(41), pred[0,:].cpu().detach().numpy(), 'r')
		plt.plot(range(nl2), l2[0,:].cpu().numpy(), 'k')
		plt.plot(range(nrc), rc[0,:].cpu().numpy(), 'g')
		plt.show()
	if k % monitor_every == monitor_view:
		print(pgn.forward_weights())
		
	k += 1

# break the problem down: 
# see if the rc network can generate a one-hot output from l1 and l2. 
rcnet = nn.Sequential(
	nn.Linear(41+nl2, nrc), 
	nn.LeakyReLU(0.2))

optimizer = optim.AdamW(rcnet.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=2e-4)
lossfunc = torch.nn.SmoothL1Loss()

while True: 
	l1 = render(obj_)
	(pos,vel,width) = obj_
	obj2 = (0.0, vel, width) #centered 
	l2 = render(obj2)
	l2 = l2[12:28]
	# make the desired one-hot location. 
	rcdes = obj_to_rc(obj_)
		
	rcnet.zero_grad()
	pred = rcnet(torch.cat([l1, l2]))
	loss = lossfunc(pred, rcdes)
	loss.backward()
	optimizer.step()
	slowloss = 0.99*slowloss + 0.01 * loss.detach()
	if k % 200 == 0 : 
		upd = time.time() - last_update
		print(f'{k} loss: {loss}; slowloss {slowloss} time {upd}')
		last_update = time.time()
	if k % 100000 == 0:
		plt.plot(range(41), l1.cpu().detach().numpy(), 'b')
		plt.plot(range(nl2), l2.cpu().detach().numpy(), 'k')
		plt.plot(range(nrc), rcdes.cpu().numpy(), 'r')
		plt.plot(range(nrc), pred.detach().cpu().numpy(), 'g')
		plt.show()
		
	obj_ = move(obj_)
	if torch.sum(l1) == 0.0 : 
		obj_ = new_obj()
	k += 1
	
	# yes!  the one-layer network can absolutely output one-hot location activations. 
	

