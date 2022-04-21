# a reimplementaton of Wolfgang Maass's probabalistic skeleton
# -- only with learning rules! and better tasks! 
import jax
import math
import numpy as np
from numpy.random import randn
from numpy.random import rand
import torch
import matplotlib.pyplot as plt
import jax.numpy as jnp
import ipdb
import time
import functools
import pdb

print(jax.devices())
print(jax.devices()[0].platform, jax.devices()[0].device_kind)

seed = 17013
key = jax.random.PRNGKey(seed)

# Dimensions
SYNAPSES = 8 # synapses per dendrite (constant constraint here seems somewhat reasonable)
DENDRITES = 4 # dendrites per neuron
INDIM = 9
OUTDIM = 4
NEURTYPE = 5 # including 0:input, 1:output, and 2:error.
NEURONS = (INDIM + OUTDIM) * NEURTYPE # neurons per organism
POP = 1 # number of organisms in a population

CONN_DIST_SIGMA = 0.35
CONN_DIST_SIGMA2 = CONN_DIST_SIGMA * CONN_DIST_SIGMA
START_WEIGHT = 0.01

def make_stim(s, x, y):
	# make a simple on-off vector pattern based on bits s, x, y
	# see notes for a depiction.
	out = torch.zeros((3,3))
	if s:
		# shape 0 is L rotated 180 deg
		out[x, y] = 1
		out[x, y+1] = 1
		out[x+1, y] = 1
	else:
		out[x, y] = 1
		out[x+1, y+1] = 1
	return torch.reshape(out, (9,))


# active variables
q = '''
act_source = torch.zeros((NEURONS, DENDRITES, SYNAPSES), dtype=torch.int32)
act_weight = torch.zeros((NEURONS, DENDRITES, SYNAPSES)) '''
# dest is implicit in the structure. 
act_dend = jnp.zeros((NEURONS, DENDRITES)) # store the dendritic activities
act_neur = jnp.zeros((NEURONS)) # store the neuronal activities

# developmental variables
# maybe create them in torch and convert to jax? 
dev_index = jnp.arange(0, NEURONS)
dev_location = torch.zeros(NEURONS, 2)
dev_ntype = torch.zeros((NEURONS,), dtype=torch.int32)
e = 0
for i in range(INDIM):
	for j in range(NEURTYPE):
		dev_ntype[e] = j
		dev_location[e, 0] = float(i) / INDIM
		dev_location[e, 1] = 0.0
		e = e+1

for i in range(OUTDIM):
	for j in range(NEURTYPE):
		dev_ntype[e] = j
		dev_location[e, 0] = float(i) / INDIM
		dev_location[e, 1] = 1.0
		e = e+1

def to_jax(torchvar):
	return jnp.asarray(torchvar.numpy())

dev_location = to_jax(dev_location)
dev_ntype = to_jax(dev_ntype)

key, subkey = jax.random.split(key)
genome = jax.random.uniform(subkey, (2, NEURTYPE, NEURTYPE)) / 100.0 
# fill this out with connection probabilities & initial weights. 
# later we can parameterize this via MLPs (or something). 
# indexed as [layer, input, output] connection probabilities. 

# learning rules -- hebbian / anti-hebbian with linear and quadratic terms. 
# since this seemed to work alright for the xor task. 
key, subkey = jax.random.split(key)
learning_rules = jax.random.uniform( subkey, (2, NEURTYPE)) / 1000.0

# the genome specifies a probability of connection. 
# since we are doing a gather-type operation, for each neuron need to sample from all the possible inputs, 
# add up the total probability, normalize, then sample that. (likely a better way..?)
key, subkey = jax.random.split(key)
rand_numbz = jax.random.uniform(subkey, (NEURONS,DENDRITES,SYNAPSES))
#for i in range(NEURONS):
	#ntype = dev_ntype[i]
	#locx = dev_location[i, 0]
	#locz = dev_location[i, 1]
	#for j in range(DENDRITES):
		#for k in range(SYNAPSES):
			#sumprob = 0.0
			## now need to iterate over all the potential inputs. 
			#for n in range(NEURONS):
				#probv = jnp.zeros((NEURONS,))
				#if n != i: # no self connections.
					#pntype = dev_ntype[n]
					#plocx = dev_location[n, 0]
					#plocz = dev_location[n, 1]
					#distx = math.exp( -0.5* (locx - plocx)*(locx - plocx) / (CONN_DIST_SIGMA*CONN_DIST_SIGMA) )
					#distz = plocz - locz # specifies connections between layers. 
					#nz = 0 if distz < 0.5 else 1
					#prob = distx * genome[nz, pntype, ntype]
					#probv[n] = prob
			#sumprob = jnp.sum(probv)
			#x = rand_numbz[i, j, k] * sumprob
			#sp = 0.0
			#for n in range(NEURONS):
				#p = probv[n]
				#if x >= sp and x < sp + p:
					#act_source[i, j, k] = n
					#act_weight[i, j, k] = START_WEIGHT
					## really need to make this a vmap / pure functional form for speed.  But first, make it work!
				#sp = sp + p 

# convert a given destination and source to a probability of connection.
def select_input(dn, dntype, dloc, sn, sntype, sloc):
	# given a destination neuron  number, type and location,
	# compute the probabilities of connection from all other neurons.
	nonself = jnp.where( dn != sn, 1.0, 0.0)
	dlocx = dloc[0]
	dlocz = dloc[1]
	slocx = sloc[0]
	slocz = sloc[1]
	distx = jnp.exp(-0.5* (dlocx-slocx)*(dlocx-slocx) / CONN_DIST_SIGMA2 )
	distz = dlocz - slocz # specifies connections between layers.
	nz = jnp.array(jnp.round(abs(distz)), int)
	prob = nonself * distx * genome[nz, dntype, sntype]
	return distx

# for a given destination neuron, select a source neuron based on the genome.
def make_synapse(dn, dntype, dloc, drandu):
	pvec = jax.vmap(select_input, (None, None, None, 0, 0, 0), 0)\
		(dn, dntype, dloc, dev_index, dev_ntype, dev_location)
	cdf = jnp.cumsum(pvec)
	totp = cdf[-1]
	x = drandu * totp
	proximity = -1.0 * ( (cdf - x)**2.0 )
	(prox, indx) = jax.lax.top_k(proximity, 1)
	return indx

# for a given destination neuron, make one dendritic branch.
def make_dendrites(dn, dntype, dloc, drandu):
	return jax.vmap(make_synapse, (None, None, None, 0), 0) \
		(dn, dntype, dloc, drandu)

# likewise, make a neuron
def make_neuron(dn, dntype, dloc, drandu):
	return jax.vmap(make_dendrites, (None, None, None, 0), 0) \
		(dn, dntype, dloc, drandu)

def make_brain(dn, dntype, dloc, drandu):
	return jax.vmap(make_neuron, (0,0,0,0), 0)(dn, dntype, dloc, drandu)

select_input(dev_index[0], dev_ntype[0], dev_location[0], dev_index[1], dev_ntype[1], dev_location[1])

q = make_synapse(dev_index[0], dev_ntype[0], dev_location[0], rand_numbz[0,0,0])
print("make_synapse")
print(q.shape) # (1,)

q = make_dendrites(dev_index[0], dev_ntype[0], dev_location[0], rand_numbz[0,0])
print("make_dendrites")
print(q.shape)

q = make_neuron(dev_index[0], dev_ntype[0], dev_location[0], rand_numbz[0])
print("make_neuron")
print(q.shape)

act_source = make_brain(dev_index, dev_ntype, dev_location, rand_numbz)
# for some reason there is a trailing 1-dim .. ? 
act_source = jnp.squeeze(act_source)
print("act_source:")
print(act_source.shape)
act_weight = jnp.ones((NEURONS, DENDRITES, SYNAPSES)) * START_WEIGHT

# ok! that should have set up the source and the weight. 
# now we can try to simulate it.  
# really need to batch this in the future ...
q = 0
st, sx, sy = (int(q/4)%2, q%2, int(q/2)%2)
# st, sx, sy = (randbool(), randbool(), randbool())
l1e = to_jax(make_stim(st, sx, sy))
lr = 0.001

def step(neur, src, w):
    return neur[src] * w

def update_synapse(qsrc, qdst, dst_n, win, learnrules):
    typ = dev_ntype[dst_n]; # learning rules are indexed by the destination neuron.
    # qsrc = presynaptic
    # qdst = postsynaptic
    lrn = learning_rules[:, typ]
    scl = 1.0 - jnp.exp((1.0 - jnp.clip(qdst, 1.0, 1e6)) * -0.04)
    downscale = win * scl
    # dw = lr*( lrn[0]* pre * post + lrn[1] * math.pow(pre*post, 3) )
    dw = qsrc * qdst - downscale
    return dw

def update_dendrite(qsrc, qdst, dst_n, win, learnrules):
	return jax.vmap(update_synapse, (0, None, None, 0, None), 0)\
		(qsrc, qdst, dst_n, win, learnrules)

def update_neuron(qsrc, qdst, dst_n, win, learnrules):
	return jax.vmap(update_dendrite, (0, 0, None, 0, None), 0)\
		(qsrc, qdst, dst_n, win, learnrules)

def update_brain(qsrc, qdst, dst_n, win, learnrules):
	return jax.vmap(update_neuron, (0, 0, 0, 0, None), 0)\
		(qsrc, qdst, dst_n, win, learnrules)


def sim_step():
	act_dend = jnp.zeros((NEURONS, DENDRITES)) # store the dendritic activities
	act_neur = jnp.zeros((NEURONS)) # store the neuronal activities
	for u in range(5):
		# hold the input constant.
		for i in range(INDIM):
			act_neur.at[i * NEURTYPE].set(l1e[i])
		act_syn = jax.vmap(step, (None, 2, 2), 2)(act_neur, act_source, act_weight)
		# just sum the dendrites for now
		act_dend = jnp.sum(act_syn, 2)
		act_neur = jnp.sum(act_dend, 1)
		act_neur = jnp.clip(act_neur, 0.0, 2.5)

	act_w = update_brain(act_source, act_dend, dev_index, act_weight, learning_rules)

sim_step_jit = jax.jit(sim_step)

N = 5000000
start = time.time()
for i in range(N):
	sim_step_jit()
end = time.time()

print((end - start) / float(N))

#print("act_w")
#print(act_w.shape)
