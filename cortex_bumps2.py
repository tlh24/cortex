import os
import math
import numpy as np
import torch
import random
import pdb
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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

def move(input_obj):
	(position, velocity, width) = input_obj; 
	# can kinda do whatever we want here, 
	# but let's keep it simple for now. 
	return (position + velocity, velocity, width)

def new_obj():
	scl = 1.0
	if torch.randn(1) < 0.0:
		scl = -1.0
	position = 20.5 * scl
	velocity = (1.75 + torch.randn(1) * 0.35) * scl * -1.0
	width = 5.0 + torch.randn(1) * 2.0
	return(position, velocity, width)

# need to set up some weight matrices. 
# assume that all vectors are vertical (row-vectors)
# hence out = W*in. 
# fan-in is 'cols'

def random_weights(rows, cols):
	return torch.clamp(torch.mul(torch.randn(rows, cols), math.sqrt(2.0/cols)), 0.0, 1.0)

def random_weights3(a, b, c):
	 return torch.clamp(torch.mul(torch.randn(a, b, c), math.sqrt(2.0 / (b * c))), 0.0, 1.0)
	 # this is a guess! 

# pass-gates: 
# output = rc * in
# Weight = 16 x 41 x 14 (out, in, rc)
# to update the weights, want an outer product space
# Weight * rc = weight2
# d_weight2 = pad_right(in) * pad_left(rc)   (outer product)
# -- modulo nonlinearities. 
# d_weight3 = reshape(d_weight2, 41 * 14)
# d_weight4 = pad_right(out) * pad_left(d_weight3)   (outer product)
# d_weight = reshape(d_weight4, (16, 41, 14))

def pass_gate(w_, inp, rc):
	w2 = torch.matmul(w_, rc)
	return torch.matmul(w2, inp)

def convallis(v):
	# atan(8*x.^3 - 4*1.5*x.^2)
	# this has a root around 0.74
	p = 8*torch.pow(v, 3.0) - 6*torch.pow(v, 2.0)
	return torch.atan(p)
 
def passgate_update(w_, inp, outp, rc, lr):
	dw2 = torch.outer(inp, rc) # TBD really should check matrix ops here too
	dw3 = torch.reshape( dw2, (inp.size(0)*rc.size(0), ) )
	dw4 = torch.outer(outp, dw3)
	dw4 = torch.reshape( dw4, (outp.size(0), inp.size(0), rc.size(0)) )
	# this reshape seems to be in the right order 4/22/21
	dw4 = torch.mul(dw4, lr)
	return torch.clamp(torch.add(w_, dw4), -1.0, 1.0) # idk? 

def wrc_update(w_, rc, l1l2, lr):
	dw = torch.outer(rc, l1l2)
	dw = torch.mul(dw, lr)
	return torch.clamp(torch.add(w_, dw), -1.0, 1.0)

def inhib_update(w_, l, li, lr):
	# this function approximates competitive inhibition: 
	# if A and B are both active, but A > B, then A inhibits B, 
	# but B does not inhibit A. 
	# this function also scales the weight matrix exponetially
	# if any of the inputs are > 2.0. 
	# inputs: w_, inhibitory weights
	# l, the current layer e+i activation. 
	# li, layer i (for scaling)
	# lr, learning rate. 
	dw = torch.outer(l, l) # this doesn't break symmetry
	dw2 = torch.outer(torch.pow(l, 2.0), torch.ones(l.size(0))) # this does!
	# it also forces the diagonal to zero. 
	dw2 = torch.clamp(dw - dw2, 0.0, 1.0); # only positive weight updates
	dw2 = torch.mul(dw2, lr)
	w_ = torch.add(w_, dw2)
	scale = torch.clamp(li, 2.0, 1e6)
	scale = torch.sub(scale, 2.0)
	scale = torch.exp(torch.mul(scale, -0.04))
	one = torch.ones(w_.size(0))
	dw = torch.outer(scale, one)
	return torch.mul(w_, dw)

def test_inhib_update():
	w = torch.zeros(3, 3)
	a = torch.tensor([0.4, 0.2, 0.1])
	print(a)
	ai = torch.zeros(3); 
	wp = inhib_update(w, a, ai, 1.0)
	print(wp)
	inhib = torch.matmul(wp, a)
	print(inhib)

def passgate_homeostasis(w_, outp, outpavg):
	scale = torch.clamp(outp, 1.5, 1e6)
	scale = torch.sub(scale, 1.5)
	scale = torch.exp(torch.mul(scale, -0.04)) # so many guesses
	scale2 = torch.clamp(outpavg, 0.0, 0.06)
	scale2 = torch.sub(0.06, scale2)
	scale2 = torch.exp(torch.mul(scale2, 0.2))
	scale = scale * scale2 
	one = torch.ones(w_.size(1) * w_.size(2))
	dw = torch.outer(scale, one) # TBD plz check this looks wrong
	dw = torch.reshape(dw, (w_.size(0), w_.size(1), w_.size(2)))
	return(torch.mul(w_, dw))

def rc_homeostasis(w_, rc, rcavg):
	# if anything is > 2, scale down all synaptic weights. 
	scale = torch.clamp(rc, 1.5, 1e6)
	scale = torch.sub(scale, 1.5)
	scale = torch.exp(torch.mul(scale, -0.04)) # more guesses
	scale2 = torch.clamp(rcavg, 0.0, 0.03)
	scale2 = torch.sub(0.03, scale2)
	scale2 = torch.exp(torch.mul(scale2, 0.1))
	scale = scale * scale2
	dw = torch.outer(scale, torch.ones(w_.size(1)))
	return(torch.mul(w_, dw))
 
def softmax(v):
	# softmax after thresholding for positive activity
	# vp = torch.mul(torch.exp(v * 1.2), torch.sign(torch.clamp(v, 0.0, 1e6)))
	vp = torch.exp(v * 1.2)
	denom = torch.sum(vp) / 3.0
	return torch.div(vp, denom)

# weight from thalamus (vision neurons) to l1
w_vis_l1 = torch.eye(41)
w_fpg = random_weights3(16, 41, 14)
w_rpg = random_weights3(41, 16, 14)
w_rc = random_weights(14, (41+41))
w_rci = (torch.ones(14,14) - torch.eye(14)) * (2.0/14.0)
w_l2i = (torch.ones(16,16) - torch.eye(16)) * (2.0/16.0)

def step():
	obj_ = (-10.0, 1.0, 5.0)
	# second layer will hold state through time. 
	l2 = torch.zeros(16)
	l2avg = torch.ones(16) * 0.15
	l1i = torch.zeros(41)
	l1iavg = torch.ones(41) * 0.15 # don't update -- no time sparsity prior
	rc = torch.zeros(14)
	rcavg = torch.ones(14) * 0.15
	global w_rc
	global w_fpg
	global w_rpg
	global w_rci
	global w_l2i
	j = 0
	learn = 0.01
	while True: 
		vis = render(obj_)
		for k in range(10):
			l1e = torch.matmul(w_vis_l1, vis) # just a copy. excitatory
			l1 = l1e - l1i # subtract previous timestep inhibition
			
			l1vis = torch.cat((l1, vis))
			rce = torch.matmul(w_rc, l1vis)
			rce = torch.clamp(rce, 0.0, 2.5)
			rci = torch.matmul(w_rci, rc) # for spatial sparsity
			rcu = rce - rci; 
			rcu = torch.clamp(rcu, 0.0, 2.5)
			rc = 0.85 * rc + 0.15 * rcu # time continuity
			# 0.9^7 ~= 0.48, so each step there is (potentially) 50% hold-over.
			rcavg = 0.997 * rcavg + 0.003 * rc
			w_rc = rc_homeostasis(w_rc, rce, rcavg) 
			if k >= 8:
				# update only when the RC neurons have settled. 
				w_rc = wrc_update(w_rc, rc, l1vis, learn*0.4)
				w_rci = inhib_update(w_rci, rcu, rci, learn*0.1)
			
			l2e = pass_gate(w_fpg, vis, rc) ### TBD
			l2e = torch.clamp(l2e, 0.0, 2.5)
			l2i = torch.matmul(w_l2i, l2)
			#l2u = torch.clamp(l2e - l2i, 0.0, 2.5)
			l2 = 0.97 * l2 + 0.03 * l2e
			# 0.97^7 ~= 0.8.  Decent holdover between frames.
			l2avg = 0.997 * l2avg + 0.003 * l2
			w_fpg = passgate_homeostasis(w_fpg, l2e, l2avg)
			if k >= 8:
				# what if we make this not depend on l1 or l2? 
				w_fpg = passgate_update(w_fpg, l1, l2, rc, learn) ### TBD
				#w_l2i = inhib_update(w_l2i, l2, l2i, learn*0.1)
			
			l1i = pass_gate(w_rpg, l2, rc)
			l1i = torch.clamp(l1i, 0.0, 2.5)
			w_rpg = passgate_homeostasis(w_rpg, l1i, l1iavg)
			l1 = l1e - l1i
			if k >= 8:
				w_rpg = passgate_update(w_rpg, l2, l1, rc, learn)
				# so if l1 is around zero, there should be no weight updates.
				
			if (j % 10000) < 160:
				yield (l1, l1e, l1i, rc, rce, rci, rcavg, l2, l2e, l2i, l2avg, w_rci)
			else:
				if j % 200 == 9:
					yield (l1, l1e, l1i, rc, rce, rci, rcavg, l2, l2e, l2i, l2avg, w_rci)
			j = j+1
		
		obj_ = move(obj_)
		vis = render(obj_)
		if torch.sum(vis) == 0.0 : 
			obj_ = new_obj()
		
def animate(data, axs):
	(l1, l1e, l1i, rc, rce, rci, rcavg, l2, l2e, l2i, l2avg, w_rci) = data
	axs[0,0].clear()
	axs[0,0].set_title('black = l1, blue = l1e (thalm), red = l1i')
	axs[0,0].plot(range(-20, 21), l1, 'k'); 
	axs[0,0].plot(range(-20, 21), l1e, 'b');
	axs[0,0].plot(range(-20, 21), l1i, 'r'); 
	axs[0,0].set(xlim=(-20, 20), ylim = (-0.2, 1.0))
	axs[0,1].clear()
	axs[0,1].set_title('black = rc, blue = rce, red = rci, green = rcavg')
	axs[0,1].plot(range(14), rc, 'k')
	axs[0,1].plot(range(14), rce, 'b')
	axs[0,1].plot(range(14), rci, 'r')
	axs[0,1].plot(range(14), rcavg, 'g')
	axs[0,1].set(xlim=(0, 13), ylim = (0, 1.0))
	axs[1,0].clear()
	axs[1,0].set_title('black = l2, blue = l2e, red = l2i, green = l2avg')
	axs[1,0].plot(range(16), l2, 'k')
	axs[1,0].plot(range(16), l2e, 'b')
	axs[1,0].plot(range(16), l2i, 'r')
	axs[1,0].plot(range(16), l2avg, 'g')
	axs[1,0].set(xlim=(0, 15), ylim = (0, 1.0))
	axs[1,1].clear()
	axs[1,1].set_title('w_rci')
	pos = axs[1,1].imshow(w_rci)
	fig.tight_layout()

#test_inhib_update()

vis = render(obj_)
fig,axs = plt.subplots(2,2, figsize=(18,12))
ani = animation.FuncAnimation(fig, animate, step, interval=10, repeat=True, fargs=(axs,))
plt.show()

#def threshold_LTP(v):
	#v = torch.sub(v, 1.0) 
	#act = torch.mul(torch.exp(-0.5 * v), v) # hermite fn [1]
	#ltp = torch.gt(act, 0.0) # act > 0
	#ltd = torch.lt(act, 0.0) # act < 0

#def process(vis):
	## vis is a 41x1 column vector, output from 'render'. 
	#l1 = torch.matmul(w_vis_l1, vis)
	## l2 = torch.mul(l2, 0.5); # activity delay. 
	
	#rc = torch.matmul(w_l12_rc, torch.concat(l1, l2)); 
	#rc = softmax(rc) # lateral inhibition & gain normalization
	
	#pgf = torch.mul(w_rc_pgf, rc) # broadcast the rc activations.  correct semantics? 
	#pgf = torch.sumdim(pgf, 1); # determine if the pass-gates are open. 
	#pgf = torch.reshape(pgf, 16, 41); 
	#l2_t = torch.matmul(w_l1_l2, l1); 
	#l2 = torch.matmul(pgf, l2_t); 
	
	## adjust the forward weights in a Convalis-Hebbian way. 
	#[l1_act, l1_ltp, l1_ltd] = threshold_LTP(l1); 
	#[l2_act, l2_ltp, l2_ltd] = threshold_LTP(l2); 
	#[rc_act, rc_ltp, rc_ltd] = threshold_LTP(rc);
	
	## (guess) to update any weight, all must be in LTP region. 
	#ltp = torch.repmat(l1_act, (1, 

# Add this to the repo: 
# Towards and integration of deep learning and neuroscience
