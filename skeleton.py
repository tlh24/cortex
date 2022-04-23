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

TEST = False

# Dimensions
SYNAPSES = 3 # synapses per dendrite (constant constraint here seems somewhat reasonable)
DENDRITES = 4 # dendrites per neuron
INDIM = 9 # number of mini-columns
OUTDIM = 4
NEURTYPE = 6 # including 0:input, 1:output, and 2:error.
NEURONS = (INDIM + OUTDIM) * NEURTYPE # neurons per organism
POP = 128 # number of organisms in a population

CONN_DIST_SIGMA = 0.5
CONN_DIST_SIGMA2 = CONN_DIST_SIGMA * CONN_DIST_SIGMA
START_WEIGHT = 0.01

q = '''
g_source = torch.zeros((POP, NEURONS, DENDRITES, SYNAPSES), dtype=torch.int32)
g_weight = torch.zeros((POP, NEURONS, DENDRITES, SYNAPSES)) '''
# dest is implicit in the structure. 

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

def to_torch(jaxvar):
	# jax variables are read-only, so we need to make a copy.
	return torch.from_numpy(np.copy(np.asarray(jaxvar)))

dev_location = to_jax(dev_location)
dev_ntype = to_jax(dev_ntype)
print('dev_location', dev_location.shape)
print(dev_location)
print('dev_ntype', dev_ntype.shape)
print(dev_ntype)

key, subkey = jax.random.split(key)
g_genome = jax.random.uniform(subkey, (POP, 3, NEURTYPE, NEURTYPE)) / 100.0
# fill this out with connection probabilities & initial weights. 
# later we can parameterize this via MLPs (or something). 
# indexed as [layer, src, dst] connection probabilities.

# for testing, we need something that is fully structured:
# neuron type 0 connects to 1
# 2 to 3 etc.  connection matrix is one-off-diagonal.
if TEST:
	g_genome = torch.zeros(POP, 3, NEURTYPE, NEURTYPE) # (layer, dest, source)
	for p in range(POP):
		for i in range(NEURTYPE):
			j = (i + 1) % NEURTYPE
			g_genome[p, 1, i, j] = 1.0
			## up one layer, off set by two.
			j = (i + 2) % NEURTYPE
			g_genome[p, 2, i, j] = 0.0
			# down one layer, offset by three.
			j = (i + 3) % NEURTYPE
			g_genome[p, 0, i, j] = 0.0
	g_genome = to_jax(g_genome)
print('g_genome', g_genome.shape)
print(g_genome)

# learning rules -- hebbian / anti-hebbian with linear and quadratic terms. 
# since this seemed to work alright for the xor task. 
key, subkey = jax.random.split(key)
learning_rules = jax.random.uniform( subkey, (POP, 2, NEURTYPE)) / 100.0
if TEST:
	learning_rules = jnp.ones((POP, 2, NEURTYPE))

# the genome specifies a probability of connection. 
# since we are doing a gather-type operation, for each neuron need to sample from all the possible inputs, 
# add up the total probability, normalize, then sample that. (likely a better way..?)


# convert a given destination and source to a probability of connection.
def select_input(dn, dntype, dloc, sn, sntype, sloc, genome):
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
def make_synapse(dn, dntype, dloc, drandu, genome):
	# iterate over all possible input neurons, emit probability of connection.
	pvec = jax.vmap(select_input, (None, None, None, 0, 0, 0, None), 0)\
		(dn, dntype, dloc, dev_index, dev_ntype, dev_location, genome)
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
def make_dendrite(dn, dntype, dloc, drandu, genome):
	return jax.vmap(make_synapse, (None, None, None, 0, None), 0) \
		(dn, dntype, dloc, drandu, genome)

# likewise, make a neuron
def make_neuron(dn, dntype, dloc, drandu, genome):
	return jax.vmap(make_dendrite, (None, None, None, 0, None), 0) \
		(dn, dntype, dloc, drandu, genome)

def make_brain(dn, dntype, dloc, drandu, genome):
	return jax.vmap(make_neuron, (0,0,0,0, None), 0)\
		(dn, dntype, dloc, drandu, genome)

def make_pop(dn, dntype, dloc, drandu, genome):
	return jax.vmap(make_brain, (None,None,None,0, 0), 0)\
		(dn, dntype, dloc, drandu, genome)

def test_development():
	global key
	key, subkey = jax.random.split(key)
	rand_numbz = jax.random.uniform(subkey, (POP,NEURONS,DENDRITES,SYNAPSES))

	print('select_input [0] [1]')
	q = select_input(dev_index[0], dev_ntype[0], dev_location[0], dev_index[1], dev_ntype[1], dev_location[1], g_genome[0])
	print(q)
	print('select_input [0] [2]')
	q = select_input(dev_index[0], dev_ntype[0], dev_location[0], dev_index[2], dev_ntype[2], dev_location[2], g_genome[0])
	print(q)
	print('select_input [0] [11]')
	q = select_input(dev_index[0], dev_ntype[0], dev_location[0], dev_index[11], dev_ntype[11], dev_location[11], g_genome[0])
	print(q)
	print('select_input [11] [0]')
	q = select_input(dev_index[11], dev_ntype[11], dev_location[11], dev_index[0], dev_ntype[0], dev_location[0], g_genome[0])
	print(q)

	print("make_synapse [0] rand [0]")
	q = make_synapse(dev_index[0], dev_ntype[0], dev_location[0], rand_numbz[0,0,0,0], g_genome[0])
	print(q.shape)
	print(q) # (1,)

	print("make_synapse [0] rand [1]")
	q = make_synapse(dev_index[0], dev_ntype[0], dev_location[0], rand_numbz[0,0,0,1], g_genome[0])
	print(q.shape)
	print(q) # (1,)

	q = make_dendrite(dev_index[0], dev_ntype[0], dev_location[0], rand_numbz[0,0,0], g_genome[0])
	print("make_dendrite [0,0,0]", q.shape)
	print(q)

	q = make_neuron(dev_index[0], dev_ntype[0], dev_location[0], rand_numbz[0,0], g_genome[0])
	print("make_neuron [0,0]", q.shape)
	print(q)

	q = make_brain(dev_index, dev_ntype, dev_location, rand_numbz[0], g_genome[0])
	print("make_brain [0]", q.shape)
	print(q.shape)

#test_development()

key, subkey = jax.random.split(key)
rand_numbz = jax.random.uniform(subkey, (POP,NEURONS,DENDRITES,SYNAPSES))

g_source = make_pop(dev_index, dev_ntype, dev_location, rand_numbz, g_genome)
# for some reason there is a trailing 1-dim .. ? 
g_source = jnp.squeeze(g_source)
print("g_source:", g_source.shape)
print(g_source)
g_weight = jnp.ones((POP, NEURONS, DENDRITES, SYNAPSES)) * START_WEIGHT
print('g_weight')
print(g_weight.shape)

lr = 0.001

def step(neur, src, w):
    return neur[src] * w

def step_brain(neur, src, w):
	return jax.vmap(step, (None, 2, 2), 2)(neur, src, w)

def update_synapse(src_n, qdst, dst_n, win, act, learnrules):
    typ = dev_ntype[dst_n]; # learning rules are indexed by the destination neuron.
    qsrc = act[src_n]
    # qsrc = presynaptic
    # qdst = postsynaptic
    lrn = learnrules[:, typ]
    scl = 1.0 - jnp.exp((jnp.clip(qdst, 1.0, 1e6) - 1.0) * -0.04)
    downscale = win * scl
    dw = lr*( lrn[0] * qsrc*qdst + lrn[1] * jax.lax.pow(qsrc*qdst, 3.0) )
    dw = dw - downscale
    # we can put all sorts of stuff here..
    # dw = qsrc * qdst - downscale
    return dw

def update_dendrite(src_n, qdst, dst_n, win, act, learnrules):
	return jax.vmap(update_synapse, (0, None, None, 0, None, None), 0)\
		(src_n, qdst, dst_n, win, act, learnrules)

def update_neuron(src_n, qdst, dst_n, win, act, learnrules):
	return jax.vmap(update_dendrite, (0, 0, None, 0, None, None), 0)\
		(src_n, qdst, dst_n, win, act, learnrules)

def update_brain(src_n, qdst, dst_n, win, act, learnrules):
	return jax.vmap(update_neuron, (0, 0, 0, 0, None, None), 0)\
		(src_n, qdst, dst_n, win, act, learnrules)

def update_pop(src_n, qdst, dst_n, win, act, learnrules):
	return jax.vmap(update_brain, (0, 0, None, 0, 0, 0), 0)\
		(src_n, qdst, dst_n, win, act, learnrules)
		# each individual has (potentially) different learning rules.

def test_weight_update():
	# need to test the weight update.
	act_dend = torch.zeros((POP, NEURONS, DENDRITES))
	act_neur = torch.zeros(POP, NEURONS)
	act_dend[0,0,0] = 1.0
	act_dend[0,0,2] = 1.0
	print('g_source[0,0,0,0]', g_source[0,0,0,0])
	act_neur[0,17] = 1.0
	act_dend[0,2,0] = 1.0
	act_dend[0,2,2] = 1.0
	act_neur[0,3] = 1.0
	act_dend[0,4,0] = 2.0
	act_dend = to_jax(act_dend)
	act_neur = to_jax(act_neur)
	weight = jnp.zeros((POP, NEURONS, DENDRITES, SYNAPSES))

	learning_rules = torch.zeros((POP, 2, NEURTYPE))
	learning_rules[0,0,0] = 1.0 # synapses on type 0 neurons are on.
	learning_rules[0,0,2] = 1.0 # synapses on type 2 neurons are on.
	learning_rules = to_jax(learning_rules)

	print('update_synapse [1] -> [0,0]')
	q = update_synapse(g_source[0,0,0,0], act_dend[0,0,0], 0, 0.0, act_neur[0], learning_rules[0])
	print(q)
	print(q.shape)

	print('update_synapse [3] -> [2,2]')
	q = update_synapse(g_source[0,2,2,0], act_dend[0,2,2], 2, 0.0, act_neur[0], learning_rules[0])
	print(q)

	print('update_synapse [0] -> [4,0] downscale w=1.0')
	q = update_synapse(g_source[0,4,0,0], act_dend[0,4,0], 4, 1.0, act_neur[0], learning_rules[0])
	print(q)

	print('update_dendrite src[0]')
	q = update_dendrite(g_source[0,0,0], act_dend[0,0,0], 0, weight[0,0,0], act_neur[0], learning_rules[0])
	print(q)

	print('update_dendrite src[1] (should be 0, act_neur[2] = 0)')
	q = update_dendrite(g_source[0,1,0], act_dend[0,1,0], 0, weight[0,1,0], act_neur[0], learning_rules[0])
	print(q)

	print('update_brain')
	q = update_brain(g_source[0], act_dend[0], dev_index, weight[0], act_neur[0], learning_rules[0])
	print(q)

#test_weight_update()

# make all stimuli
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

stimuli = torch.zeros(8, 9)
for i in range(8):
	st, sx, sy = (int(i/4)%2, i%2, int(i/2)%2)
	stimuli[i,:] = make_stim(st, sx, sy)
print('stimuli')
print(stimuli)
# expand this so we can blit easily using 'where'
stimuli_expanded = torch.zeros(8,POP,NEURONS)
stimuli_mask = torch.zeros(POP, NEURONS)
for j in range(8):
	for i in range(POP):
		for k in range(9):
			stimuli_expanded[j,i, k*NEURTYPE] = stimuli[j,k]
			stimuli_mask[i, k*NEURTYPE] = 1.0
stimuli_expanded = to_jax(stimuli_expanded)
stimuli_mask = to_jax(stimuli_mask)
print('stimuli_expanded', stimuli_expanded.shape)
print(jnp.squeeze(stimuli_expanded[:, 0, :]))
print('stimuli_mask', stimuli_mask.shape)
print(stimuli_mask)

def sim_step(stim_indx):
	global g_weight # updated in this function.
	global act_dend
	global act_neur
	global dev_ntype
	act_dend = jnp.zeros((POP,NEURONS,DENDRITES)) # store the dendritic activities
	act_neur = jnp.zeros((POP,NEURONS)) # store the neuronal activities
	for u in range(5):
		# hold the input constant.
		act_neur = jnp.where(stimuli_mask > 0, \
			stimuli_expanded[stim_indx], act_neur)
		q = jnp.sum(act_neur)
		# print('act_neur at start sum=', q, act_neur.shape)
		# print(act_neur)
		# half the neuron types are inhibitory.
		def make_inhibitory(ntype, neur):
			return jnp.where(ntype >= int(NEURTYPE / 2), -1.0*neur, neur)
		# print(dev_ntype)
		act_neur = jax.vmap(make_inhibitory, (None, 0), 0)(dev_ntype, act_neur)
		act_syn = jax.vmap(step_brain, (0, 0, 0), 0)(act_neur, g_source, g_weight)
		# print("act_syn", act_syn.shape)
		# just sum the dendrites for now
		act_dend = jnp.sum(act_syn, 3)
		act_neur = jnp.sum(act_dend, 2)
		act_neur = jnp.clip(act_neur, 0.0, 2.5)
		# print("act_neur", act_neur.shape)

	dw = update_pop(g_source, act_dend, dev_index, g_weight, act_neur, learning_rules)
	g_weight = g_weight + dw
	g_weight = jnp.clip(g_weight, 0.0, 0.5)
	# print('dw', dw.shape)
	# print(dw)
	return jnp.mean(act_neur)

sim_step_jit = jax.jit(sim_step)

def train():
	global key
	# run the population for 200 steps?
	key, subkey = jax.random.split(key)
	stim_indxs = jax.random.randint(subkey, (200,), 0, 8)
	mean_activity = jax.vmap(sim_step_jit, (0), 0)(stim_indxs)
	#plt.plot(mean_activity)
	#plt.show()

# sim_step(0)
N = 100000
start = time.time()
for i in range(N):
	train()
end = time.time()

print("train time:", (end - start) / float(N))

# alright, so we can simulate these dang networks (not with high confidence..)
# now need to evaluate them
# three metrics
# 1 - Can it detect (predict) in-domain data?
# 2 - Can it detect OOD errors in the input?
# 3 - Does it form a working index (or addressing scheme) on the data?

# need some sort of plot to see what's going on ... right?

#print("act_w")
#print(act_w.shape)
