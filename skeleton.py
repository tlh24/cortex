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
SYNAPSES = 3 # synapses per dendrite (constant constraint here seems somewhat reasonable)
DENDRITES = 4 # dendrites per neuron
INDIM = 2 # number of mini-columns
OUTDIM = 2
NEURTYPE = 5 # including 0:input, 1:output, and 2:error.
NEURONS = (INDIM + OUTDIM) * NEURTYPE # neurons per organism
POP = 1 # number of organisms in a population

CONN_DIST_SIGMA = 0.0005
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
# create them in torch and convert to jax -- they only need to be created once.
dev_index = jnp.arange(0, NEURONS)
dev_location = torch.zeros(NEURONS, 2)
dev_ntype = torch.zeros((NEURONS,), dtype=torch.int32)
e = 0
for i in range(INDIM):
	for j in range(NEURTYPE):
		dev_ntype[e] = j
		dev_location[e, 0] = float(i) / (INDIM - 1)
		dev_location[e, 1] = 0.0
		e = e+1

for i in range(OUTDIM):
	for j in range(NEURTYPE):
		dev_ntype[e] = j
		dev_location[e, 0] = float(i) / (INDIM - 1)
		dev_location[e, 1] = 1.0
		e = e+1

def to_jax(torchvar):
	return jnp.asarray(torchvar.numpy())

dev_location = to_jax(dev_location)
dev_ntype = to_jax(dev_ntype)
print('dev_location', dev_location.shape)
print(dev_location)
print('dev_ntype', dev_ntype.shape)
print(dev_ntype)

key, subkey = jax.random.split(key)
genome = jax.random.uniform(subkey, (2, NEURTYPE, NEURTYPE)) / 100.0 
# fill this out with connection probabilities & initial weights. 
# later we can parameterize this via MLPs (or something). 
# indexed as [layer, src, dst] connection probabilities.

# for testing, we need something that is fully structured:
# neuron type 0 connects to 1
# 2 to 3 etc.  connection matrix is one-off-diagonal.
genome = torch.zeros(3, NEURTYPE, NEURTYPE) # (layer, dest, source)
for i in range(NEURTYPE):
	j = (i + 1) % NEURTYPE
	genome[1, i, j] = 1.0
	# up one layer, off set by two.
	j = (i + 2) % NEURTYPE
	genome[2, i, j] = 0.0
	# down one layer, offset by three.
	j = (i + 3) % NEURTYPE
	genome[0, i, j] = 0.0
genome = to_jax(genome)
print('genome', genome.shape)
print(genome)

# learning rules -- hebbian / anti-hebbian with linear and quadratic terms. 
# since this seemed to work alright for the xor task. 
key, subkey = jax.random.split(key)
learning_rules = jax.random.uniform( subkey, (2, NEURTYPE)) / 1000.0
learning_rules = jnp.ones((2, NEURTYPE))

# the genome specifies a probability of connection. 
# since we are doing a gather-type operation, for each neuron need to sample from all the possible inputs, 
# add up the total probability, normalize, then sample that. (likely a better way..?)
key, subkey = jax.random.split(key)
rand_numbz = jax.random.uniform(subkey, (NEURONS,DENDRITES,SYNAPSES))

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
	nz = jnp.array(jax.lax.clamp(jnp.round(distz), -1.0, 1.0), int) + 1
	# print(dntype, sntype, nz, genome[nz, dntype, sntype])
	prob = nonself * distx * genome[nz, dntype, sntype]
	return prob

# for a given destination neuron, select a source neuron based on the genome.
def make_synapse(dn, dntype, dloc, drandu):
	pvec = jax.vmap(select_input, (None, None, None, 0, 0, 0), 0)\
		(dn, dntype, dloc, dev_index, dev_ntype, dev_location)
	cdf = jnp.cumsum(pvec)
	totp = cdf[-1]
	x = drandu * totp
	proximity = cdf - x # need to select the smallest positive number here
	proximity = jnp.where(proximity < 0.0, 1e6, proximity) # ignore negative distances
	proximity = -1.0 * proximity # smallest positive -> smallest negative
	# print(pvec, cdf, proximity)
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

def test_development():
	print('select_input [0] [1]')
	q = select_input(dev_index[0], dev_ntype[0], dev_location[0], dev_index[1], dev_ntype[1], dev_location[1])
	print(q)
	print('select_input [0] [2]')
	q = select_input(dev_index[0], dev_ntype[0], dev_location[0], dev_index[2], dev_ntype[2], dev_location[2])
	print(q)
	print('select_input [0] [11]')
	q = select_input(dev_index[0], dev_ntype[0], dev_location[0], dev_index[11], dev_ntype[11], dev_location[11])
	print(q)
	print('select_input [11] [0]')
	q = select_input(dev_index[11], dev_ntype[11], dev_location[11], dev_index[0], dev_ntype[0], dev_location[0])
	print(q)

	print("make_synapse [0] rand [0]")
	q = make_synapse(dev_index[0], dev_ntype[0], dev_location[0], rand_numbz[0,0,0])
	print(q.shape)
	print(q) # (1,)

	print("make_synapse [0] rand [1]")
	q = make_synapse(dev_index[0], dev_ntype[0], dev_location[0], rand_numbz[0,0,1])
	print(q.shape)
	print(q) # (1,)

	q = make_dendrites(dev_index[0], dev_ntype[0], dev_location[0], rand_numbz[0,0])
	print("make_dendrites", q.shape)
	print(q)

	q = make_neuron(dev_index[0], dev_ntype[0], dev_location[0], rand_numbz[0])
	print("make_neuron [0]", q.shape)
	print(q)

act_source = make_brain(dev_index, dev_ntype, dev_location, rand_numbz)
# for some reason there is a trailing 1-dim .. ? 
act_source = jnp.squeeze(act_source)
print("act_source:", act_source.shape)
print(act_source)
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

def update_synapse(src_n, qdst, dst_n, win, act, learnrules):
    typ = dev_ntype[dst_n]; # learning rules are indexed by the destination neuron.
    qsrc = act[src_n]
    # qsrc = presynaptic
    # qdst = postsynaptic
    lrn = learning_rules[:, typ]
    scl = 1.0 - jnp.exp((jnp.clip(qdst, 1.0, 1e6) - 1.0) * -0.04)
    downscale = win * scl
    dw = lr*( lrn[0] * qsrc*qdst + lrn[1] * jax.lax.pow(qsrc*qdst, 3.0) )
    dw = dw - downscale
    # we can put all sorts of stuff here..
    # dw = qsrc * qdst - downscale
    return dw

def update_dendrite(qsrc, qdst, dst_n, win, act, learnrules):
	return jax.vmap(update_synapse, (0, None, None, 0, None, None), 0)\
		(qsrc, qdst, dst_n, win, act, learnrules)

def update_neuron(qsrc, qdst, dst_n, win, act, learnrules):
	return jax.vmap(update_dendrite, (0, 0, None, 0, None, None), 0)\
		(qsrc, qdst, dst_n, win, act, learnrules)

def update_brain(qsrc, qdst, dst_n, win, act, learnrules):
	return jax.vmap(update_neuron, (0, 0, 0, 0, None, None), 0)\
		(qsrc, qdst, dst_n, win, act, learnrules)

# need to test the weight update.
act_dend = torch.zeros((NEURONS, DENDRITES))
act_neur = torch.zeros(NEURONS)
act_dend[0,0] = 1.0
act_dend[0,2] = 1.0
act_neur[1] = 1.0
act_dend[2,0] = 1.0
act_dend[2,2] = 1.0
act_neur[3] = 1.0
act_dend[4,0] = 2.0
act_dend = to_jax(act_dend)
act_neur = to_jax(act_neur)
act_weight = jnp.zeros((NEURONS, DENDRITES, SYNAPSES))

learning_rules = torch.zeros((2, NEURTYPE))
learning_rules[0,2] = 1.0 # so only synapses on type 2 neurons are on.
learning_rules = to_jax(learning_rules)

print('update_synapse [1] -> [0,0]')
q = update_synapse(act_source[0,0,0], act_dend[0,0], 0, 0.0, act_neur, learning_rules)
print(q)

print('update_synapse [3] -> [2,2]')
q = update_synapse(act_source[2,2,0], act_dend[2,2], 2, 0.0, act_neur, learning_rules)
print(q)

print('update_synapse [0] -> [4,0] downscale w=1.0')
q = update_synapse(act_source[4,0,0], act_dend[4,0], 4, 1.0, act_neur, learning_rules)
print(q)

print('update_dendrite src[0]')
q = update_dendrite(act_source[0,0], act_dend[0,0], 0, act_weight[0,0], act_neur, learning_rules)
print(q)

print('update_dendrite src[1] (should be 0, act_neur[2] = 0)')
q = update_dendrite(act_source[1,0], act_dend[1,0], 0, act_weight[1,0], act_neur, learning_rules)
print(q)

print('update_brain')
q = update_brain(act_source, act_dend, dev_index, act_weight, act_neur, learning_rules)
print(q)


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

	act_w = update_brain(act_source, act_dend, dev_index, act_weight, act_neur, learning_rules)

sim_step_jit = jax.jit(sim_step)

N = 50000
start = time.time()
for i in range(N):
	sim_step_jit()
end = time.time()

print((end - start) / float(N))

#print("act_w")
#print(act_w.shape)
