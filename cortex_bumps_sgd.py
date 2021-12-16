
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
import numpy as np
import matplotlib.pyplot as plt
import pdb
from multiprocessing import Pool

import torch

from torch import nn, optim
from torch.nn import init
from torch.nn import functional as F
from torch.autograd import Function

batches = 16

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


def random_weights3(a, b, c):
	 return torch.clamp(torch.mul(torch.randn(batches, a, b, c), math.sqrt(2.0 / (b * c))), -1.0, 1.0)
	 # this is a guess! 
	 
def random_weights(rows, cols):
	return torch.clamp(torch.mul(torch.randn(batches, rows, cols), math.sqrt(2.0/cols)), -1.0, 1.0)

class PassGate(nn.Module): 
	def __init__(self, out_dim, in_dim, rc_dim):
		super().__init__()
		self.w_ = torch.nn.Parameter(data = random_weights3(out_dim, in_dim, rc_dim), requires_grad=True)
		self.b_ = torch.nn.Parameter(data = torch.zeros(batches, out_dim), requires_grad=True)
		
	def update_rc(self, rc):
		self.w_.data = torch.clamp(self.w_, -1.0, 1.0)
		rc = rc[:, None, :, None] # batch wise reduction
		self.w2 = torch.squeeze(torch.matmul(self.w_, rc))
	
	def forward(self, inp):
		inp = inp[:,:,None] # so we do a reduction 
		return torch.squeeze(torch.matmul(self.w2, inp)) + self.b_
	
	
class PGnet(nn.Module):
	def __init__(self):
		super().__init__()
		self.fpg = PassGate(16, 41, 14)
		self.rpg = PassGate(41, 16, 14)
		self.pgctrl = torch.nn.Linear(41+16+41, 14)
		self.pgrelu = torch.nn.LeakyReLU(0.2)
		self.pgsoftmax = torch.nn.Softmax(dim=0)
		# self.pgctrl = torch.nn.Parameter(data = random_weights(41+16, 14), requires_grad=True)
		self.l2 = torch.zeros(batches, 16)
		self.l1i = torch.zeros(batches, 41)
		self.l2softmax = torch.nn.Softmax(dim=0)
		self.rcblurweight = torch.zeros((batches,14,14))
		for k in range(14):
			self.rcblurweight[:,k,k] = 0.85
		for k in range(13):
			self.rcblurweight[:,k,k+1] = 0.075
			self.rcblurweight[:,k+1,k] = 0.075
		self.rcblurweight = self.rcblurweight.to(gdevice)
		self.l2blurweight = torch.zeros((batches,16,16))
		for k in range(16):
			self.l2blurweight[:,k,k] = 0.85
		for k in range(15):
			self.l2blurweight[:,k,k+1] = 0.075
			self.l2blurweight[:,k+1,k] = 0.075
		self.l2blurweight = self.l2blurweight.to(gdevice)
		
	def forward(self, inp):
		# run the network for several steps to reach equilibrium
		# this of course is prone to positive feedback instability! 
		l2 = self.l2
		l1i = self.l1i
		with torch.no_grad():
			for k in range(5):
				rc = self.pgrelu(self.pgctrl(torch.cat([inp, l2, inp-l1i])))
				# rc = self.pgsoftmax(rc)
				rc = torch.clamp(rc, -0.1, 1.0)
				rc = torch.matmul(self.rcblurweight, rc)
				self.fpg.update_rc(rc) 
				self.rpg.update_rc(rc)
				# l2 = self.l2softmax(self.fpg.forward(inp))
				l2 = torch.clamp(self.fpg.forward(inp), -0.1, 1.0)
				l2 = torch.matmul(self.l2blurweight, l2)
				l1i = self.rpg.forward(l2)
				l1i = torch.clamp(l1i, 0.0, 1.0) 
				
		# now run once to update the weights with the equilibriated activities. 
		l2 = l2.detach() # might not be necessary..
		l1i = l1i.detach()
		
		rc = self.pgrelu(self.pgctrl(torch.cat([inp, l2, inp-l1i])))
		rc = self.pgsoftmax(rc)
		# rc = torch.clamp(rc, -0.1, 1.0)
		self.fpg.update_rc(rc) 
		self.rpg.update_rc(rc)
		l2 = torch.clamp(self.fpg.forward(inp), -0.1, 1.0)
		# l2 = self.l2softmax(self.fpg.forward(inp))
		l1i = self.rpg.forward(l2)
		l1i = torch.clamp(l1i, 0.0, 1.0)
		
		self.l2 = l2.detach()
		self.l1i = l1i.detach()
		
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

obj_ = (-10.0, 1.0, 5.0) # position, veclocity, width.

def render(input_obj):
	vis = np.zeros(41); 
	(position, velocity, width) = input_obj; 
	for j in range(-20,21):
		val = 0.0
		if j >= position - width and j <= position + width:
			val = (math.cos((j-position) / width * math.pi) + 1.0)/2.0 
		vis[j+20] = val
	return torch.from_numpy(vis).to(torch.float)

def render_batch(input_objs):
	output = torch.zeros(batches, 41)
	for ind,res in enumerate(pool.map(render, input_objs)):
		output[ind, :] = res
	return output

def move(input_obj):
	(position, velocity, width) = input_obj; 
	# can kinda do whatever we want here, 
	# but let's keep it simple for now. 
	return (position + velocity, velocity, width)

def new_obj(ignore):
	scl = 1.0
	if torch.randn(1) < 0.0:
		scl = -1.0
	position = 20.5 * scl
	velocity = (1.75 + torch.randn(1) * 0.35) * scl * -1.0
	width = 5.0 + torch.randn(1) * 2.0
	return(position, velocity, width)

def move_batch(input_objs):
	input_objs = pool.map(move, input_objs)
	for ind,ob in enumerate(input_objs):
		(pos, vel, w) = ob
		if pos < -22 or pos > 22:
			input_objs[ind] = new_obj(0)
	return input_objs

def obj_to_rc(obj_):
	# convert the object position to a one-hot RC activation.
	(pos,vel,width) = obj_
	# make the desired one-hot location. 
	rcdes = torch.zeros(14); 
	rci = ((pos + 20) / 40.0) * 13.0
	rci = clamp(rci, 0.0, 13.001)
	rcil = int(math.floor(rci))
	res = rci - rcil
	rcdes[rcil] = 1.0 - res
	if rci < 13:
		rcdes[rcil+1] = res
	return rcdes

def obj_to_rc_batch(input_objs):
	output = torch.zeros(batches, 14)
	for ind,res in enumerate(pool.map(obj_to_rc, input_objs)):
		output[ind, :] = res
	return output

k = 0
slowloss = 0.0
last_update = time.time()

pool = Pool(batches) #defaults to number of available CPU's

pgn = PGnet()
pgn.to(device=gdevice)
optimizer = optim.AdamW(pgn.parameters(), lr = 2e-4, betas=(0.9, 0.99), weight_decay=1e-3)
mseloss = torch.nn.MSELoss()
obj_ = pool.map(new_obj, range(batches))
monitor_every = 1e4
monitor_view = 10

while True:
	vis = render_batch(obj_)
	obj_ = move_batch(obj_)
	vis = vis.to(device=gdevice)
	pgn.zero_grad()
	rc = obj_to_rc_batch(obj_)
	(pred,l2) = pgn.forward_clamp_rc(vis, rc)
	loss = mseloss(pred, vis)
	loss.backward()
	optimizer.step()
	
	pred2 = pred.detach(); 
	
	slowloss = 0.99 * slowloss + 0.01 * loss.detach()
		# be careful to detach the loss!  otherwise the gradient graph grows endlessly! 
	if k % 200 == 0 : 
		upd = time.time() - last_update
		print(f'{k} loss: {loss}; slowloss {slowloss} time {upd}')
		last_update = time.time()
		
	if k % monitor_every < monitor_view and k % monitor_every >= 0: 
		plt.plot(range(41), vis[0,:].cpu().detach().numpy(), 'b')
		plt.plot(range(41), pred[0,:].cpu().detach().numpy(), 'r')
		plt.plot(range(16), l2[0,:].cpu().numpy(), 'k')
		plt.plot(range(14), rc[0,:].cpu().numpy(), 'g')
		plt.show()
	if k % monitor_every == monitor_view:
		print(pgn.forward_weights())
		
	k += 1

# break the problem down: 
# see if the rc network can generate a one-hot output from l1 and l2. 
rcnet = nn.Sequential(
	nn.Linear(41+16, 14), 
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
		plt.plot(range(16), l2.cpu().detach().numpy(), 'k')
		plt.plot(range(14), rcdes.cpu().numpy(), 'r')
		plt.plot(range(14), pred.detach().cpu().numpy(), 'g')
		plt.show()
		
	obj_ = move(obj_)
	if torch.sum(l1) == 0.0 : 
		obj_ = new_obj()
	k += 1
	
	# yes!  the one-layer network can absolutely output one-hot location activations. 
	
# next test would be to see if, given both l2 and rcdes, the pg net can output l1i. 
# Need to do that above.. 

# yes, that seems to work as well, to a very high fidelity 
# -- can predict l1 reliably with rc one-hot encoding. 
# however however, it does not seem to set l2 to a static representation. 
# will need to penalize (?) to get this behavior ? 

# Dec 16 2021 
# let's try one update step instead of five for improving continuity / smoothness in L2 representation.
