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
import cma
import traceback

print(jax.devices())
print(jax.devices()[0].platform, jax.devices()[0].device_kind)

seed = 17016
key = jax.random.PRNGKey(seed)

TEST = False
DEBUG_ITER = False

if TEST:
	SYNAPSES = 3 # synapses per dendrite (constant constraint here seems somewhat reasonable)
	DENDRITES = 4 # dendrites per neuron
	INDIM = 4 # number of mini-columns
	OUTDIM = 4
	NEURTYPE = 6 # including 0:input, 1:output, and 2:error.
	NEURONS = (INDIM + OUTDIM) * NEURTYPE # neurons per organism
	POP = 1 # number of organisms in a population
	lr = 0.001
	CONN_DIST_SIGMA = 0.005
else:
	# Dimensions
	SYNAPSES = 4 # synapses per dendrite (constant constraint here seems somewhat reasonable)
	DENDRITES = 4 # dendrites per neuron
	INDIM = 9 # number of mini-columns
	OUTDIM = 4
	NEURTYPE = 8 # including 0:input, 1:output, and 2:error.
	NEURONS = (INDIM + OUTDIM) * NEURTYPE # neurons per organism
	POP = 1024 # number of organisms in a population
	lr = 0.1
	CONN_DIST_SIGMA = 0.25

DTYPE = dtype=jnp.float32 # this only applies to simulation, not development.
# float32: 1.23 ms / eval. (RTX 3080 laptop)
# float16: 1.5 ms / eval. (RTX 3080 laptop)
# bfloat16: 2.4 ms /eval. (RTX 3080 laptop)

CONN_DIST_SIGMA2 = CONN_DIST_SIGMA * CONN_DIST_SIGMA
START_WEIGHT = 0.0

q = '''
source = torch.zeros((POP, NEURONS, DENDRITES, SYNAPSES), dtype=torch.int32)
weight = torch.zeros((POP, NEURONS, DENDRITES, SYNAPSES)) '''
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
		dev_location[e, 0] = float(i) / (OUTDIM - 1)
		dev_location[e, 1] = 1.0
		e = e+1

# dev_location = torch.zeros(NEURONS, 2)

def to_jax(torchvar, dtype=DTYPE):
	return jnp.asarray(torchvar.numpy(), dtype=dtype)

def to_torch(jaxvar):
	# jax variables are read-only, so we need to make a copy.
	return torch.from_numpy(np.copy(np.asarray(jaxvar)))

dev_location = to_jax(dev_location)
dev_ntype = to_jax(dev_ntype, dtype=jnp.int32)
#print('dev_location', dev_location.shape)
#print(dev_location)
#print('dev_ntype', dev_ntype.shape)
#print(dev_ntype)
#quit()

key, subkey = jax.random.split(key)
g_genome = jax.random.uniform(subkey, (POP, 3, NEURTYPE, NEURTYPE), dtype=DTYPE) / 100.0
# fill this out with connection probabilities & initial weights. 
# later we can parameterize this via MLPs (or something). 
# indexed as [layer, src, dst] connection probabilities.

# for testing, we need something that is fully structured:
# neuron type 0 connects to 1
# 2 to 3 etc.  connection matrix is one-off-diagonal.
if TEST:
	g_genome = torch.zeros((POP, 3, NEURTYPE, NEURTYPE)) # (layer, dest, source)
	for p in range(POP):
		for i in range(NEURTYPE):
			# same layer
			j = (i + 1) % NEURTYPE
			g_genome[p, 1, i, j] = 1.0
			## from below, offset by two.
			j = (i + 2) % NEURTYPE
			g_genome[p, 2, i, j] = 0.0
			# from above, offset by three.
			j = (i + 3) % NEURTYPE
			g_genome[p, 0, i, j] = 0.0
	g_genome = to_jax(g_genome)
#print('g_genome', g_genome.shape)
#print(g_genome)
#quit()

# learning rules -- hebbian / anti-hebbian with linear and quadratic terms. 
# since this seemed to work alright for the xor task. 
key, subkey = jax.random.split(key)
learning_rules = jax.random.uniform( subkey, (POP, NEURTYPE, 3), dtype=DTYPE) / 100.0

# the genome specifies a probability of connection. 
# since we are doing a gather-type operation, for each neuron need to sample from all the possible inputs, 
# add up the total probability, normalize, then sample that. (likely a better way..?)


# convert a given destination and source to a probability of connection.
def select_input(dn, dntype, dloc, sn, sntype, sloc, genome, rules):
	# given a destination neuron  number, type and location,
	# compute the probabilities of connection from all other neurons.
	nonself = jnp.where( dn != sn, 1.0, 0.0)
	dlocx = dloc[0] # numerical stability ??
	dlocz = dloc[1]
	slocx = sloc[0]
	slocz = sloc[1]
	sigma = rules[dntype, 2]
	distx = jnp.exp(-0.5* ((dlocx-slocx)**2) / CONN_DIST_SIGMA2 ) # this doesnt work, ??!!
	# distx = 1.0 - jax.lax.clamp( (dlocx-slocx)**2, 0.01, 0.99)
	# distx = jax.lax.clamp(jnp.abs((dlocx - slocx) * CONN_DIST_SIGMA), 0.01, 0.99)
	distz = dlocz - slocz # specifies connections between layers.
	nz = jnp.array(jax.lax.clamp(\
		jnp.round(distz), -1.0, 1.0), int) + 1
	# print('-- dntype.dtype', dntype.dtype)
	# print(dntype, sntype, dloc, sloc, nz, genome[nz, dntype, sntype])
	# distx seems to be causing some problems here...!!
	prob = nonself * distx * genome[nz, dntype, sntype]
	return prob

# for a given destination neuron, select a source neuron based on the genome.
def make_synapse(dn, dntype, dloc, drandu, genome, rules):
	# iterate over all possible input neurons, emit probability of connection.
	pvec = jax.vmap(select_input, (None, None, None, 0, 0, 0, None, None), 0)\
		(dn, dntype, dloc, dev_index, dev_ntype, dev_location, genome, rules)
	cdf = jnp.cumsum(pvec)
	totp = cdf[-1]
	x = drandu * totp
	proximity = jnp.where(cdf >= x, cdf, 1e6)
	proximity = -1.0 * proximity # smallest positive -> smallest negative
	# print(pvec, cdf, proximity)
	(prox, indx) = jax.lax.top_k(proximity, 1)
	return indx

# for a given destination neuron, make one dendritic branch.
def make_dendrite(dn, dntype, dloc, drandu, genome, rules):
	return jax.vmap(make_synapse, (None, None, None, 0, None, None), 0) \
		(dn, dntype, dloc, drandu, genome, rules)

# likewise, make a neuron
def make_neuron(dn, dntype, dloc, drandu, genome, rules):
	return jax.vmap(make_dendrite, (None, None, None, 0, None, None), 0) \
		(dn, dntype, dloc, drandu, genome, rules)

def make_brain(dn, dntype, dloc, drandu, genome, rules):
	return jnp.squeeze(jax.vmap(make_neuron, (0,0,0,0, None, None), 0)\
		(dn, dntype, dloc, drandu, genome, rules))

def make_pop(dn, dntype, dloc, drandu, genome, rules):
	return jax.vmap(make_brain, (None,None,None,0,0,0), 0)\
		(dn, dntype, dloc, drandu, genome, rules)

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
	print(q)

#test_development()
#quit()

print('make_pop_jit')
make_pop_jit = jax.jit(make_pop)


key, subkey = jax.random.split(key)
rand_numbz = jax.random.uniform(subkey, (POP,NEURONS,DENDRITES,SYNAPSES))
# pdb.set_trace()
g_source = make_pop(dev_index, dev_ntype, dev_location, rand_numbz, g_genome, learning_rules)
## for some reason there is a trailing 1-dim .. ?
#g_source = jnp.squeeze(g_source)
#print("g_source:", g_source.shape)
#print(g_source)
#g_weight = jnp.ones((POP, NEURONS, DENDRITES, SYNAPSES), dtype=DTYPE) * START_WEIGHT
#print('g_weight')
#print(g_weight.shape)
#print(g_weight.dtype)

def step(neur, src, w):
    return neur[src] * w

def step_brain(neur, src, w):
	return jax.vmap(step, (None, 2, 2), 2)(neur, src, w)
	# src at this point is (NEURONS, DENDRITES, SYNAPSES), as is w.

def update_synapse(src_n, qdst, dst_n, win, act, rules):
    typ = dev_ntype[dst_n]; # learning rules are only indexed by the destination neuron.
    qsrc = act[src_n]
    # qsrc = presynaptic
    # qdst = postsynaptic
    lrn = rules[typ]
    scl = 1.0 - jnp.exp((jnp.clip(jnp.abs(qdst), 1.0, 1e6) - 1.0) * -0.04)
    downscale = win * scl # this should still work for negative weights.
    #
    dw = lr*( lrn[0] * qsrc*qdst + lrn[1] * jax.lax.pow(3.0*qsrc*qdst, 3.0) )
    dw = dw - downscale
    #dw = 0.1* (lrn[0] - win)
    return dw

def update_dendrite(src_n, qdst, dst_n, win, act, rules):
	return jax.vmap(update_synapse, (0, None, None, 0, None, None), 0)\
		(src_n, qdst, dst_n, win, act, rules)

def update_neuron(src_n, qdst, dst_n, win, act, rules):
	return jax.vmap(update_dendrite, (0, 0, None, 0, None, None), 0)\
		(src_n, qdst, dst_n, win, act, rules)

def update_brain(src_n, qdst, dst_n, win, act, rules):
	return jax.vmap(update_neuron, (0, 0, 0, 0, None, None), 0)\
		(src_n, qdst, dst_n, win, act, rules)

def update_pop(src_n, qdst, dst_n, win, act, rules):
	return jax.vmap(update_brain, (0, 0, None, 0, 0, 0), 0)\
		(src_n, qdst, dst_n, win, act, rules)
		# each individual has (potentially) different learning rules.

def test_weight_update(key):
	key, subkey = jax.random.split(key)
	rand_numbz = jax.random.uniform(subkey, (POP,NEURONS,DENDRITES,SYNAPSES))
	g_source = make_pop_jit(dev_index, dev_ntype, dev_location, rand_numbz, g_genome)

	print('g_source[0,0,0,0]', g_source[0,0,0,0])
	print('g_source[0,0,0,1]', g_source[0,0,0,1])
	print('g_source[0,0,1,0]', g_source[0,0,1,0])
	print('g_source[0,1,0,0]', g_source[0,1,0,0])
	print('g_source[0,2,0,0]', g_source[0,2,0,0])

	act_dend = torch.zeros((POP, NEURONS, DENDRITES))
	act_neur = torch.zeros(POP, NEURONS)
	act_dend[0,0,0] = 1.0
	act_dend[0,0,2] = 1.0
	act_neur[0,1] = 1.0
	act_dend[0,2,0] = 1.0
	act_dend[0,2,2] = 1.0
	act_neur[0,3] = 1.0
	act_dend[0,4,0] = 2.0
	act_dend = to_jax(act_dend)
	act_neur = to_jax(act_neur)
	weight = jnp.zeros((POP, NEURONS, DENDRITES, SYNAPSES), dtype=DTYPE)

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

#test_weight_update(key)
#quit()

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
def expand_stimuli(indata):
	# expand stimuli so we can blit easily to the whole population using 'where'
	N = indata.shape[0]
	expanded = torch.zeros(N,POP,NEURONS)
	mask = torch.zeros(POP, NEURONS)
	for j in range(N):
		for i in range(POP):
			for k in range(9):
				expanded[j,i, k*NEURTYPE] = indata[j,k]
				mask[i, k*NEURTYPE] = 1.0
	expanded = to_jax(expanded)
	mask = to_jax(mask)
	return expanded, mask

stimuli_expanded, stimuli_mask = expand_stimuli(stimuli)

print('stimuli_expanded', stimuli_expanded.shape)
#print(jnp.squeeze(stimuli_expanded[:, 0, :]))
#print('stimuli_mask', stimuli_mask.shape)
#print(stimuli_mask[0])

def stim_dt(stim_data, stim_indx, act_dend, act_neur, source, weight):
	global dev_ntype
	# hold the input constant.
	act_neur = jnp.where(stimuli_mask > 0, \
		stim_data[stim_indx], act_neur)

	# q = jnp.sum(act_neur)
	# print('act_neur at start sum=', q, act_neur.shape)
	# print(act_neur)
	# half the neuron types are inhibitory.
	def make_inhibitory(ntype, neur):
		return jnp.where(ntype >= int(NEURTYPE / 2), -1.0*neur, neur)
	# vmap over the population
	act_neur = jax.vmap(make_inhibitory, (None, 0), 0)(dev_ntype, act_neur)

	#fig, axs = plt.subplots(1, 2, figsize=(18, 8))
	#im = axs[0].imshow(jnp.reshape(act_neur[0, 0:INDIM*NEURTYPE], (INDIM,NEURTYPE)))
	#plt.colorbar(im, ax=axs[0])
	#axs[0].set_title('act_neur within sim_dt')
	#im = axs[1].imshow(jnp.transpose(jnp.reshape(weight[0], (NEURONS,DENDRITES*SYNAPSES))))
	#plt.colorbar(im, ax=axs[1])
	#axs[1].set_title('weight')
	#plt.show()

	# again, step over the whole population.
	act_syn = jax.vmap(step_brain, (0, 0, 0), 0)(act_neur, source, weight)
	# print("act_syn", act_syn.shape)
	# just sum the dendrites for now
	act_dend = jnp.sum(act_syn, 3)
	act_neur = jnp.sum(act_dend, 2)
	act_neur = jnp.clip(act_neur, 0.0, 2.5)

	# re-force the input to be constant (will be overwritten by above)
	act_neur = jnp.where(stimuli_mask > 0, \
		stim_data[stim_indx], act_neur)

	# add in per-subtype biases.
	def get_bias2(rules, ntype):
		def get_bias(rules, ntype):
			return rules[ntype, 2]
		return jax.vmap(get_bias, (None, 0), 0)(rules, ntype)
	bias = jax.vmap(get_bias2, (0, None), 0)(rules, dev_ntype)
	#print('bias', bias.shape, bias[0])
	act_neur = act_neur + bias*4

	return act_dend, act_neur

def sim_step_update(key, stim_data, stim_indx, source, weight, rules):
	global dev_ntype
	act_dend = jnp.zeros((POP,NEURONS,DENDRITES), dtype=DTYPE) # store the dendritic activities
	act_neur = jnp.zeros((POP,NEURONS), dtype=DTYPE) # store the neuronal activities
	for u in range(5):
		act_dend, act_neur = stim_dt(stim_data, stim_indx, act_dend, act_neur, source, weight)

	key, subkey = jax.random.split(key)
	noiz = jnp.clip(jax.random.normal(subkey, act_dend.shape) * 0.03, 0.0, 0.5)
	act_dend = act_dend + noiz

	dw = update_pop(source, act_dend, dev_index, weight, act_neur, rules)
	weight = weight + dw
	weight = jnp.clip(weight, 0.0, 0.5)

	#if DEBUG_ITER:
		#fig, axs = plt.subplots(1, 2, figsize=(18, 8))
		#im = axs[0].imshow(jnp.reshape(act_neur[0, 0:INDIM*NEURTYPE], (INDIM,NEURTYPE)))
		#plt.colorbar(im, ax=axs[0])
		#axs[0].set_title('act_neur within sim_step_update')
		#im = axs[1].imshow(jnp.transpose(jnp.reshape(dw[0], (NEURONS,DENDRITES*SYNAPSES))))
		#plt.colorbar(im, ax=axs[1])
		#axs[1].set_title('dw')
		#plt.show()

	# print('dw', dw.shape)
	# print(dw)
	return key, act_dend, act_neur, weight, dw
	# return act_neur

def sim_step_eval(stim_data, stim_indx, source, weight):
	global dev_ntype
	act_dend = jnp.zeros((POP,NEURONS,DENDRITES), dtype=DTYPE) # store the dendritic activities
	act_neur = jnp.zeros((POP,NEURONS), dtype=DTYPE) # store the neuronal activities
	for u in range(5):
		act_dend, act_neur = stim_dt(stim_data, stim_indx, act_dend, act_neur, source, weight)
	return act_neur

sim_step_update_jit = jax.jit(sim_step_update)
sim_step_eval_jit = jax.jit(sim_step_eval)

def train(key, source, rules):
	global stimuli_expanded
	weight = jnp.ones((POP, NEURONS, DENDRITES, SYNAPSES), dtype=DTYPE) * START_WEIGHT
	# run the population for 500 steps?
	key, subkey = jax.random.split(key)
	stim_indxs = jax.random.randint(subkey, (500,), 0, 8)
	act_neur = jnp.zeros((POP, NEURONS))
	# we cannot vmap this -- they must be run in serial.
	#def scan_inner(carry, indx):
		#stimuli_expanded, act_neur, source, weight, rules = carry
		#act_dend, act_neur, weight = sim_step_update_jit\
			#(stimuli_expanded, indx, source, weight, rules)
		#return (stimuli_expanded, act_neur, source, weight, rules), 0.0

	#carry, nul = jax.lax.scan(scan_inner, \
		#(stimuli_expanded, act_neur, source, weight, rules), stim_indxs)
	#(stimuli_expanded, act_neur, source, weight, rules) = carry
	if DEBUG_ITER:
		for i in range(500):
			key, act_dend, act_neur, weight, dw = sim_step_update\
				(key, stimuli_expanded, stim_indxs[i], source, weight, rules)
	else:
		for i in range(500):
			key, act_dend, act_neur, weight, dw = sim_step_update_jit\
				(key, stimuli_expanded, stim_indxs[i], source, weight, rules)
	# plt.imshow(jnp.reshape(act_neur[0, 1:NEURTYPE:INDIM*NEURTYPE+1], (3,3)))
	#plt.imshow(jnp.reshape(act_neur[0, 0:INDIM*NEURTYPE], (INDIM,NEURTYPE)))
	#plt.colorbar()
	#plt.show()
	return key, act_dend, act_neur, weight, dw

#print('jit(train())')
#train_jit = jax.jit(train)

# sim_step(0)
def test_train_speed(key):
	#key, subkey = jax.random.split(key)
	#genome = jax.random.uniform(subkey, (POP, 3, NEURTYPE, NEURTYPE), dtype=DTYPE) / 100.0
	# test!
	genome = torch.zeros((POP, 3, NEURTYPE, NEURTYPE))
	for j in range(1, NEURTYPE):
		genome[:, 1, j, 0] = 1.0
		# everything needs to be connected to something!
	genome[:, 1, 0, NEURTYPE-1] = 1.0
	genome = to_jax(genome)

	key, subkey = jax.random.split(key)
	rules = jax.random.uniform( subkey, (POP, NEURTYPE, 3), dtype=DTYPE) / 100.0
	key, subkey = jax.random.split(key)
	rand_numbz = jax.random.uniform(subkey, (POP,NEURONS,DENDRITES,SYNAPSES))
	source = make_pop_jit(dev_index, dev_ntype, dev_location, rand_numbz, genome, rules)

	print('source in test_train_speed', source.shape)
	print(jnp.squeeze(source[POP-1]))
	print(jnp.squeeze(source[POP-1]) % NEURTYPE)
	print('checksum:', jnp.sum(jnp.squeeze(source[POP-1]) % NEURTYPE))
	#quit()
	N = 50
	start = time.time()
	for i in range(N):
		train(key, source, rules)
	end = time.time()
	print("train time per iter:", (end - start) / float(N))
	return key

#print('testing training speed')
#key = test_train_speed(key)
#quit();
# training time with jit & jax.lax.scan(): 0.738 sec
# training time with native python for loop: 0.761 sec
# (3% slower -- but much easier to understand!)


# alright, so we can simulate these dang networks (not with high confidence..)
# now need to evaluate them
# three metrics
# 1 - Can it detect (predict) in-domain data?
# 2 - Can it detect OOD errors in the input?
# 3 - Does it form a working index (or addressing scheme) on the data?

# need to map neuron type / layer to prediction
# assume this is inhibitory (??)
predict_select = torch.zeros(INDIM, dtype=torch.int32) # 'get' indexes
error_select = torch.zeros(INDIM, dtype=torch.int32)
for i in range(INDIM):
	predict_select[i] = i*NEURTYPE + 1
	error_select[i] = i*NEURTYPE + 2
predict_select = to_jax(predict_select, dtype=jnp.int32)
error_select = to_jax(error_select, dtype=jnp.int32)

# need to make a corrupted version of the stimuli.
corrupt_error = torch.zeros(9*8, 9)
corrupt_correct = torch.zeros(9*8, 9)
for i in range(9*8):
	corrupt_correct[i, :] = stimuli[i%8, :]
	corrupt_error[i, int(i/8)] = 0.01

corrupt_stim = corrupt_correct + corrupt_error

corrupt_expanded, corrupt_mask = expand_stimuli(corrupt_stim)
# we actually don't need the mask -- it doesn't change.

#for i in range(8):
	#plt.imshow(torch.reshape(corrupt_correct[i, :], (3,3)))
	#plt.show()
#for i in range(24):
	#plt.imshow(torch.reshape(corrupt_stim[i, :], (3,3)))
	#plt.colorbar()
	#plt.show()
corrupt_stim = to_jax(corrupt_stim)
corrupt_error = to_jax(corrupt_error)
corrupt_correct = to_jax(corrupt_correct)

def eval_(source, weight):
	indx = jnp.arange(0, 9*8)
	act_data = jax.vmap(sim_step_eval_jit, (None, 0, None, None), 1)\
		(corrupt_expanded, indx, source, weight)
	# dimensions [POP, 72, NEURONS]
	#for i in range(8):
		#fig, axs = plt.subplots(1, 2, figsize=(12, 8))
		#im = axs[0].imshow(jnp.reshape(act_data[0, i, predict_select], (3,3)))
		#plt.colorbar(im, ax=axs[0])
		#axs[0].set_title('act_data[pred_select]')
		#im = axs[1].imshow(jnp.reshape(corrupt_correct[i,:], (3,3)))
		#plt.colorbar(im, ax=axs[1])
		#axs[1].set_title('corrupt_correct')
		#plt.show()
	def dif(a, b):
		return (a-b)**2
	pred_correct = jax.vmap(dif, (0, None), 0)(act_data[:,:,predict_select], corrupt_correct)
	pred_err = jax.vmap(dif, (0, None), 0)(act_data[:,:,error_select], corrupt_error)
	#print('pred_correct', pred_correct.shape)
	#print('pred_err', pred_err.shape)
	#pred_correct = jnp.sum(pred_correct, (1,2))
	#pred_err = jnp.sum(pred_err, (1,2))
	#fig, axs = plt.subplots(1, 2, figsize=(12, 8))
	#axs[0].plot(pred_correct)
	#axs[0].set_title('pred_correct')
	#axs[1].plot(pred_err)
	#axs[1].set_title('pred_err')
	#plt.show()
	print('mean pred_correct', jnp.mean(pred_correct), 'mean pred_err', jnp.mean(pred_err), ' / ', jnp.mean(pred_correct + pred_err))
	return pred_correct

def eval_gar(key, genome, rules):
	global dev_index
	global dev_ntype
	global dev_location

	genome = jnp.clip(genome, 0.0, 1.0)
	rules = jnp.clip(rules, -1.0, 2.0)

	key, subkey = jax.random.split(key)
	rand_numbz = jax.random.uniform(subkey, (POP,NEURONS,DENDRITES,SYNAPSES))
	source = make_pop_jit(dev_index, dev_ntype, dev_location, rand_numbz, genome, rules)

	key, act_dend, act_neur, weight, dw = train(key, source, rules)
	pop_eval = eval_(source, weight)
	pop_eval = jnp.sum(pop_eval, (1,2))
	return pop_eval, key, act_dend, act_neur, weight, dw

animate = True
plot_rows = 2
plot_cols = 3
figsize = (19, 11)
if animate:
	plt.ion()
	fig, axs = plt.subplots(plot_rows, plot_cols, figsize=figsize)
initialized = False
im = [ [0]*plot_cols for i in range(plot_rows)]
cbar = [ [0]*plot_cols for i in range(plot_rows)]


def plot_tensor(r, c, v, name, lo, hi):
	global initialized
	if not initialized:
		# seed with random data so we get the range right
		cmap_name = 'PuRd' # purple-red
		if lo == -1*hi:
			cmap_name = 'seismic'
		data = np.linspace(lo, hi, v.shape[0] * v.shape[1])
		data = np.reshape(data, (v.shape[0], v.shape[1]))
		im[r][c] = axs[r,c].imshow(data, cmap = cmap_name)
		cbar[r][c] = plt.colorbar(im[r][c], ax=axs[r,c])
	else:
		im[r][c].set_data(v)
		cbar[r][c].update_normal(im[r][c]) # probably does nothing
		axs[r,c].set_title(name)


SIZE_GENOME = 3*NEURTYPE*NEURTYPE
SIZE_RULES = 3*NEURTYPE
SIZE_ALL = SIZE_GENOME + SIZE_RULES

def iterate_evo(key, genome, rules, do_plot):
	global initialized
	pop_eval, key, act_dend, act_neur, weight, dw = eval_gar(key, genome, rules)
	#print('pop_eval', pop_eval.shape, pop_eval)

	if do_plot:
		# things to plot: act_dend, act_neur, weight,
		# genome, rules
		plot_tensor(0, 0, jnp.reshape(act_dend[0], (int(NEURONS/NEURTYPE), NEURTYPE*DENDRITES)), 'act_dend', 0.0, 1.0)
		plot_tensor(0, 1, jnp.reshape(act_neur[0], (int(NEURONS/NEURTYPE), NEURTYPE)), 'act_neur', 0.0, 2.5)
		plot_tensor(0, 2, jnp.reshape(weight[0], (NEURONS, DENDRITES*SYNAPSES)), 'weight', 0.0, 0.25)

		plot_tensor(1, 0, jnp.reshape(genome[0,1], (NEURTYPE, NEURTYPE)), 'genome', 0.0, 0.2)
		plot_tensor(1, 1, jnp.reshape(rules[0], (NEURTYPE, 3)), 'rules', -0.25, 0.25)
		plot_tensor(1, 2, jnp.reshape(dw[0], (NEURONS, DENDRITES*SYNAPSES)), 'dw', -1/500.0, 1/500.0)
		#print('act_dend[10]', jnp.reshape(act_dend[10], (int(NEURONS/NEURTYPE), NEURTYPE*DENDRITES)))
		#print('rules[10]', rules[10])
		#print('dw[1]', jnp.reshape(dw[10], (NEURONS, DENDRITES*SYNAPSES)))

		fig.tight_layout()
		fig.canvas.draw()
		fig.canvas.flush_events()
		if animate:
			initialized = True

	indx = jnp.argsort(pop_eval) # ascending order.
	# so we want the first elements (smallest error)

	old_genome = jnp.reshape(genome, (POP, SIZE_GENOME))
	old_rules = jnp.reshape(rules, (POP, SIZE_RULES))
	old_ = jnp.concatenate((old_genome, old_rules), 1)
	indx_ = jnp.arange(0, SIZE_ALL)

	def make_kid(mother, father, noise_mask, noise_mutation):
		crossover_genome = np.random.randint(0, SIZE_ALL)
		nu = jnp.where(indx_ < crossover_genome, old_[mother, :], old_[father, :])
		# add in noise: noise_mask is uniform, noise_mutation
		nu = jnp.where(noise_mask > 0.99, noise_mutation + nu, nu)
		return nu

	key, subkey = jax.random.split(key)
	mother = indx[jax.random.randint(subkey, (POP,), 0, int(POP/2) )]
	key, subkey = jax.random.split(key)
	father = indx[jax.random.randint(subkey, (POP,), 0, int(POP/2) )]

	#print('mother', mother.shape, mother)
	#print('father', father.shape, father)

	key, subkey = jax.random.split(key)
	noise_mask = jax.random.uniform(subkey, (POP, SIZE_ALL))
	key, subkey = jax.random.split(key)
	noise_mutation = jax.random.normal(subkey, (POP, SIZE_ALL)) * 0.001 # guess!

	new_ = jax.vmap(make_kid, (0, 0, 0, 0), 0)(mother, father, noise_mask, noise_mutation)

	genome = jnp.reshape(new_[:, 0:SIZE_GENOME], (POP,3,NEURTYPE,NEURTYPE))
	rules = jnp.reshape(new_[:,SIZE_GENOME:SIZE_ALL], (POP,NEURTYPE,3))

	# allow for slight negatives (disconnect)
	genome = jnp.clip(genome, -0.01, 1.0)
	rules = jnp.clip(rules, -1.0, 1.0)

	# print('source', source[0,0,0,:])

	return key, genome, rules

try:
	loaded = torch.load('skeleton_genome_rules.pt')
	genome = loaded['genome']
	rules = loaded['rules']
	genome = to_jax(genome)
	rules = to_jax(rules)
	print('loaded a saved genome / rules file!')
except Exception as e:
	print(traceback.format_exc())
	print('couldnt load model weights files.  making new random ones.')
	key, subkey = jax.random.split(key)
	genome = jax.random.uniform(subkey, (POP, 3, NEURTYPE, NEURTYPE), dtype=DTYPE) / 100.0
	key, subkey = jax.random.split(key)
	rules = jax.random.uniform( subkey, (POP, NEURTYPE, 3), dtype=DTYPE) / 100.0

# test!
#genome = torch.zeros((POP, 3, NEURTYPE, NEURTYPE))
#for j in range(1, NEURTYPE):
	#genome[:, 1, j, 0] = 1.0
	## everything needs to be connected to something!
	## this connectivity pattern should perfectly solve the problem..
#genome[:, 1, 0, NEURTYPE-1] = 1.0
#genome = to_jax(genome)

for k in range(20000):
	key, genome, rules = iterate_evo(key, genome, rules, k%4==3)
	if k % 100 == 99:
		genome_t = to_torch(genome)
		rules_t = to_torch(rules)
		m = {'genome': genome_t, 'rules': rules_t}
		torch.save(m, 'skeleton_genome_rules.pt')
		print('saving skeleton_genome_rules.pt')

genome_ = jnp.reshape(genome[0, :, :, :], (SIZE_GENOME,))
rules_ = jnp.reshape(rules[0, :, :], (SIZE_RULES,))
print('genome_', rules_.shape)
print('rules_', rules_.shape)
gar = jnp.concatenate((genome_, rules_), 0)
cma.s.pprint(cma.CMAOptions(''))
es = cma.CMAEvolutionStrategy(jnp.asarray(gar), 0.002, {'popsize':POP})
while not es.stop():
	solutions = es.ask()
	genome = torch.zeros(POP, SIZE_GENOME)
	rules = torch.zeros(POP, SIZE_RULES)
	i = 0
	for s in solutions:
		genome[i, :] = torch.tensor(s[0:SIZE_GENOME])
		rules[i, :] = torch.tensor(s[SIZE_GENOME : SIZE_ALL])
		i = i + 1
	genome = torch.reshape(genome, (POP, 3, NEURTYPE, NEURTYPE))
	rules = torch.reshape(rules, (POP, NEURTYPE, 3))
	genome = to_jax(genome)
	rules = to_jax(rules)
	evl, key = eval_gar(key, genome, rules)
	# evl = 300.0 - evl
	# print('evl', evl.shape, evl)
	es.tell(solutions, np.asarray(evl))
	es.logger.add()
	es.disp()
