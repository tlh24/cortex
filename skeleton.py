# a reimplementaton of Wolfgang Maass's probabalistic skeleton
# -- only with learning rules! and better tasks! 
# CUDA_VISIBLE_DEVICES=0,1
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
# import cma
import traceback
plt.rcParams['figure.dpi'] = 170

print(jax.devices())
print("jax", jax.devices()[0].platform, jax.devices()[0].device_kind)

# torch.set_default_tensor_type('torch.cuda.FloatTensor')
# ^^ slows things down!
torch_device = 1
print("torch cuda devices", torch.cuda.device_count())
print("torch device", torch.cuda.get_device_name(torch_device))

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
	lr = 1.0
	CONN_DIST_SIGMA = 1.0
else:
	# Dimensions
	SYNAPSES = 4 # synapses per dendrite (constant constraint here seems somewhat reasonable)
	DENDRITES = 4 # dendrites per neuron
	INDIM = 9 # number of mini-columns
	OUTDIM = 4
	NEURTYPE = 8 # including 0:input, 1:output, and 2:error.
	NEURONS = (INDIM + OUTDIM) * NEURTYPE # neurons per organism
	POP = 8*1024 # number of organisms in a population
	#POP = 7680
	lr = 0.25
	CONN_DIST_SIGMA = 0.025
	NUMRULES = 4 # hebbian, cube_hebb, bias, sigma.

DTYPE = jnp.float16 # this only applies to simulation, not development.
DTYPE32 = jnp.float32
def dtype(a):
	return jnp.float16(a)

CONN_DIST_SIGMA2 = CONN_DIST_SIGMA * CONN_DIST_SIGMA
START_WEIGHT = 0.0

SIZE_GENOME = 3*NEURTYPE*NEURTYPE
SIZE_RULES = NUMRULES*NEURTYPE
SIZE_ALL = SIZE_GENOME + SIZE_RULES
print('search dimension (SIZE_ALL):', SIZE_ALL, 'POP:', POP, 'dtype', DTYPE)

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
	return jnp.asarray(torchvar.cpu().numpy(), dtype=dtype)

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
	sigma = rules[dntype, 3] + 0.0013456
	distx = jnp.exp(-0.5* ((dlocx-slocx)**2) / (sigma*sigma) ) # this doesnt work, ??!!
	# distx = 1.0 - jax.lax.clamp( (dlocx-slocx)**2, 0.01, 0.99)
	# distx = jax.lax.clamp(jnp.abs((dlocx - slocx) * CONN_DIST_SIGMA), 0.01, 0.99)
	distz = dlocz - slocz # specifies connections between layers.
	nz = jnp.array(jnp.clip(\
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

#print('make_pop_jit')
make_pop_jit = jax.jit(make_pop)


def step(neur, src, w):
    return neur[src] * w

def step_brain(neur, src, w):
	return jax.vmap(step, (None, 2, 2), 2)(neur, src, w)
	# src at this point is (NEURONS, DENDRITES, SYNAPSES), as is w.

def step_pop(act_neur, source, weight):
	return jax.vmap(step_brain, (0, 0, 0), 0)(act_neur, source, weight)

def update_synapse(src_n, act_dend, act_neur, dst_n, win, act, rules):
    typ = dev_ntype[dst_n]; # learning rules are only indexed by the destination neuron.
    qsrc = act[src_n]
    # qsrc = presynaptic
    # act_dend = postsynaptic
    lrn = rules[typ]
    scl = 1.0 - jnp.exp((jnp.clip(jnp.abs(act_neur), 1.0, 1e6) - 1.0) * -0.04)
    downscale = win * scl # this should still work for negative weights.
    #
    dw = lr*( lrn[0] * qsrc*act_dend + lrn[1] * jax.lax.pow(3.0*qsrc*act_dend, dtype(3.0)) )
    dw = dw - downscale
    #dw = 0.1* (lrn[0] - win)
    return dw

def update_dendrite(src_n, act_dend, act_neur, dst_n, win, act, rules):
	return jax.vmap(update_synapse, (0, None, None, None, 0, None, None), 0)\
		(src_n, act_dend, act_neur, dst_n, win, act, rules)

def update_neuron(src_n, act_dend, act_neur, dst_n, win, act, rules):
	return jax.vmap(update_dendrite, (0, 0, None, None, 0, None, None), 0)\
		(src_n, act_dend, act_neur, dst_n, win, act, rules)

def update_brain(src_n, act_dend, act_neur, dst_n, win, act, rules):
	return jax.vmap(update_neuron, (0, 0, 0, 0, 0, None, None), 0)\
		(src_n, act_dend, act_neur, dst_n, win, act, rules)

def update_pop(src_n, act_dend, act_neur, dst_n, win, act, rules):
	return jax.vmap(update_brain, (0, 0, 0, None, 0, 0, 0), 0)\
		(src_n, act_dend, act_neur, dst_n, win, act, rules)
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

	learning_rules = torch.zeros((POP, NEURTYPE, 4))
	learning_rules[0,1,1] = 1.0 # synapses on type 0 neurons are on.
	learning_rules[0,2,1] = 1.0 # synapses on type 2 neurons are on.
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

def check_inf(a, name):
	s = jnp.sum(jnp.isinf(a))
	jax.lax.cond( s>0, lambda b: print(f'%s has infs!\n'%(name)), lambda b: None, s )

def test_weight_update2(key):
	print('test_weight_update2()')
	genome = torch.zeros((POP, 3, NEURTYPE, NEURTYPE))
	for j in range(1, NEURTYPE):
		genome[:, 1, j, j-1] = 1.0
		# everything needs to be connected to something!
	genome[:, 1, 0, NEURTYPE-1] = 1.0
	genome = to_jax(genome, dtype=DTYPE32)

	rules = torch.zeros((POP, NEURTYPE, NUMRULES))
	rules[0,1,0] = 1.0 # synapses on type 1 cubic on
	rules[0,2,0] = 1.0 # synapses on type 2 cubic on
	rules = to_jax(rules)

	key, subkey = jax.random.split(key)
	rand_numbz = jax.random.uniform(subkey, (POP,NEURONS,DENDRITES,SYNAPSES))
	source = make_pop_jit(dev_index, dev_ntype, dev_location, rand_numbz, genome, rules)

	weight = torch.zeros((POP, NEURONS, DENDRITES, SYNAPSES))
	weight[0, 1, :, :] = 0.1 # input weights on neuron 1 all 0.1
	weight[0, NEURTYPE+2, :, :] = 0.1 # input weights on neuron 1 all 0.1
	weight[0, 2, :, :] = 1.0 # for checking downscaling.
	weight = to_jax(weight)

	act_neur = torch.zeros((POP, NEURONS))
	act_neur[0, 0] = 1.0
	act_neur[0, NEURTYPE+1] = 1.0
	act_neur[0, 2] = 2.0
	act_neur = to_jax(act_neur)

	act_syn = step_pop(act_neur, source, weight)
	act_dend = jnp.sum(act_syn, 3)
	# don't update act_neur (for the purposes of checking)

	dw = update_pop(source, act_dend, act_neur, dev_index, weight, act_neur, rules)

	# -------
	# need to check outer product, too!
	key, subkey = jax.random.split(key)
	rand_numbz = jax.random.uniform(subkey, (POP,NEURONS,DENDRITES,SYNAPSES))
	CONN_DIST_SIGMA = 1.0
	source = make_pop_jit(dev_index, dev_ntype, dev_location, rand_numbz, genome, rules)

	act_neur = torch.zeros((POP, NEURONS))
	act_dend = torch.zeros((POP, NEURONS, DENDRITES))
	act_neur[0,0] = 0.5
	act_neur[0,NEURTYPE] = 1.0
	act_neur[0,2*NEURTYPE] = 0.75
	act_neur[0,3*NEURTYPE] = 0.25
	act_dend[0,1,0] = 0.33
	act_dend[0,1,1] = 0.67
	act_dend[0,1,2] = 1.0
	act_dend[0,1,3] = 0.25
	act_neur = to_jax(act_neur)
	act_dend = to_jax(act_dend)

	dw = update_pop(source, act_dend, act_neur, dev_index, weight, act_neur, rules)
	# yes, it seems that dw is accurately reflecting the (gather) outer-product.

	return key

#key = test_weight_update2(key)
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
#print('stimuli')
#print(stimuli)
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

#print('stimuli_expanded', stimuli_expanded.shape)
#print(jnp.squeeze(stimuli_expanded[:, 0, :]))
#print('stimuli_mask', stimuli_mask.shape)
#print(stimuli_mask[0])

def sim_dt(stim_data, stim_indx, act_dend, act_neur, source, weight, rules):
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
	act_syn = step_pop(act_neur, source, weight)
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
		return jax.vmap(get_bias, (None, 0), 0)(rules, ntype) # over ntypes
	bias = jax.vmap(get_bias2, (0, None), 0)(rules, dev_ntype) # over pop
	#print('bias', bias.shape, bias[0])
	act_neur = act_neur + bias*4

	return act_dend, act_neur

def sim_step_update(key, k, stim_data, stim_indx, source, weight, rules):
	global dev_ntype
	act_dend = jnp.zeros((POP,NEURONS,DENDRITES), dtype=DTYPE) # store the dendritic activities
	act_neur = jnp.zeros((POP,NEURONS), dtype=DTYPE) # store the neuronal activities
	for u in range(5):
		act_dend, act_neur = sim_dt(stim_data, stim_indx, act_dend, act_neur, source, weight, rules)

	key, subkey = jax.random.split(key)
	noiz_scl = 0.022 * (jnp.cos(3.1415926 * k / 500.0) + 1.0)
	noiz = jnp.clip(jax.random.normal(subkey, act_dend.shape, dtype=DTYPE) * noiz_scl, 0.0, 0.8)
	act_dend = act_dend + noiz

	dw = update_pop(source, act_dend, act_neur, dev_index, weight, act_neur, rules)
	weight = weight + dw
	weight = jnp.clip(weight, 0.0, 1.0)

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
	return key, k+1, act_dend, act_neur, weight, dw
	# return act_neur

def sim_step_eval(stim_data, stim_indx, source, weight, rules):
	global dev_ntype
	act_dend = jnp.zeros((POP,NEURONS,DENDRITES), dtype=DTYPE) # store the dendritic activities
	act_neur = jnp.zeros((POP,NEURONS), dtype=DTYPE) # store the neuronal activities
	for u in range(5):
		act_dend, act_neur = sim_dt(stim_data, stim_indx, act_dend, act_neur, source, weight, rules)
	return act_neur

sim_step_update_jit = jax.jit(sim_step_update)
sim_step_eval_jit = jax.jit(sim_step_eval)

def train(key, source, rules):
	global stimuli_expanded
	weight = jnp.ones((POP, NEURONS, DENDRITES, SYNAPSES), dtype=DTYPE) * START_WEIGHT
	# run the population for 500 steps?
	key, subkey = jax.random.split(key)
	stim_indxs = jax.random.randint(subkey, (500,), 0, 8)
	# we cannot vmap this -- they must be run in serial.
	# we can scan it though!
	act_neur = jnp.zeros((POP, NEURONS), dtype=DTYPE)
	act_dend = jnp.zeros((POP, NEURONS, DENDRITES), dtype=DTYPE)
	def scan_inner(carry, indx):
		key, k, act_dend, act_neur, dw, source, weight, rules = carry
		key, k, act_dend, act_neur, weight, dw = \
			sim_step_update\
			(key, k, stimuli_expanded, indx, source, weight, rules)
		return (key, k, act_dend, act_neur, dw, source, weight, rules), 0.0

	carry, nul = jax.lax.scan(scan_inner, \
		(key, 0, act_dend, act_neur, weight, source, weight, rules), stim_indxs)
	(key, k, act_dend, act_neur, dw, source, weight, rules) = carry

	#if DEBUG_ITER:
		#for i in range(500):
			#key, act_dend, act_neur, weight, dw = sim_step_update\
				#(key, stim_indxs[i], source, weight, rules)
	#else:
		#for i in range(500):
			#key, act_dend, act_neur, weight, dw = sim_step_update_jit\
				#(key, stim_indxs[i], source, weight, rules)
	# plt.imshow(jnp.reshape(act_neur[0, 1:NEURTYPE:INDIM*NEURTYPE+1], (3,3)))
	#plt.imshow(jnp.reshape(act_neur[0, 0:INDIM*NEURTYPE], (INDIM,NEURTYPE)))
	#plt.colorbar()
	#plt.show()
	return key, act_dend, act_neur, weight, dw

#print('jit(train())')
#train_jit = jax.jit(train)


# alright, so we can simulate these dang networks (not with high confidence..)
# now need to evaluate them
# three metrics
# 1 - Can it detect (predict) in-domain data?
# 2 - Can it detect OOD errors in the input?
# 3 - Does it form a working index (or addressing scheme) on the data?

# need to map neuron type / layer to prediction
# assume this is inhibitory (??)
select_predict = torch.zeros(INDIM, dtype=torch.int32) # 'get' indexes
select_error = torch.zeros(INDIM, dtype=torch.int32)
select_out = torch.zeros(OUTDIM, dtype=torch.int32)
for i in range(INDIM):
	select_predict[i] = i*NEURTYPE + 1
	select_error[i] = i*NEURTYPE + 2
for i in range(OUTDIM):
	select_out[i] = (i+INDIM)*NEURTYPE
select_predict = to_jax(select_predict, dtype=jnp.int32)
select_error = to_jax(select_error, dtype=jnp.int32)
select_out = to_jax(select_out, dtype=jnp.int32)

# need to make a corrupted version of the stimuli.
corrupt_error = torch.zeros(9*8, 9)
corrupt_correct = torch.zeros(9*8, 9)
out_correct = torch.zeros(9*8, 3)
for i in range(9*8):
	corrupt_correct[i, :] = stimuli[i%8, :]
	corrupt_error[i, int(i/8)] = 0.5 #FIXME
	out_correct[i, 0] = float(i%2)
	out_correct[i, 1] = float(int(i/2)%2)
	out_correct[i, 2] = float(int(i/4)%2)

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
corrupt_stim = to_jax(corrupt_stim, dtype=DTYPE32)
corrupt_error = to_jax(corrupt_error, dtype=DTYPE32)
corrupt_correct = to_jax(corrupt_correct, dtype=DTYPE32)
out_correct = to_jax(out_correct, dtype=DTYPE32)

def behavioral_novelty(x, siz):
	# act_data = jnp.reshape(act_data, (POP, 72*NEURONS))
	# from a given dataset, want to maximize the minimum distance between
	# any behavior and all other behaviors.
	# annoyingly, does not seem to work well in jax. keep running out of memory!!
	# have to use pytorch.
	#x = to_torch(act_data)
	#x = x.cuda(torch_device)
	x = torch.clip(x, 0.0, 1.5) # apply the ~same criteria as 'reasonable'
	# otherwise the distance benefits from going outside..
	dist = torch.cdist(x, x)
	#dist = to_jax(dist)
	#@jax.jit
	#def l2_dist2(m, ai, bi):
		#return jax.lax.fori_loop(0, siz, \
			#lambda i,c: c+(m[ai,i]-m[bi,i])**2, 0.0)
	#def l2_dist_cond(m, ai, bi):
		#return jax.lax.cond(bi > ai, l2_dist2, lambda a,b,c: 0.0, m,ai,bi)
	#def l2_dist3(m, ind, vi):
		#return jax.vmap(l2_dist_cond, (None, 0, None), 0)(m, ind, vi)

	#ind1 = jnp.arange(0,siz)
	#ind2 = jnp.arange(0,siz)
	#dist = jax.vmap(l2_dist3, (None, None, 0), 0)(act_data, ind1, ind2)

	# this is lower-triangular,
	# e.g [1,0] is the distance from 1 to 0 (which is symmetric, hence
	# copy it for minimum calculation..)
	dist = dist + torch.eye(siz, device=torch_device) * 1e6
	return dist

def reasonable_activity(act_data):
	# quadratically penalize solutions that vary too much.
	return jnp.sum((jnp.clip(act_data, 2.0, 1e6)-2.0), 1)

def eval_(source, weight, rules, novelty_hist, do_novelty, lb_eval):
	indx = jnp.arange(0, 9*8)
	act_data = jax.vmap(sim_step_eval_jit, (None, 0, None, None, None), 1)\
		(corrupt_expanded, indx, source, weight, rules)
	# dimensions [POP, 72, NEURONS]

	# !!! note this math needs to be float32, otherwise it overflows !!!
	act_data = jnp.array(act_data, dtype=DTYPE32)

	if do_novelty:
		act_data = jnp.reshape(act_data, (POP, 72*NEURONS))
		reasonable = reasonable_activity(act_data)
		indx = jnp.arange(0, 72*NEURONS, 3)
		act_data = act_data[:, indx] # downsample both space and time, speed up the distance calc.
		adt = to_torch(act_data).cuda(torch_device)
		novelty_hist = torch.cat((novelty_hist, adt), 0)
		dist = behavioral_novelty(novelty_hist, POP*2)
		dist,_ = torch.min(dist, 1) # axis dosn't matter, it's symmetric.
		# select the largest ones (most distant points) to continue
		indx = torch.argsort(dist)
		indx = indx[POP:POP*2]
		novelty_hist = novelty_hist[indx]
		#need to return the pseudo-objective
		# minimize = maximize negative.
		novelty = -1.0 * dist[POP:POP*2]

		novelty = to_jax(novelty, dtype=DTYPE32)
		print('novelty', jnp.mean(novelty), 'reasonable', jnp.mean(reasonable))
		return novelty_hist, novelty + 0.03*reasonable

	else:
		def dif(a, b):
			return (a-b)**2
		pred_correct = jax.vmap(dif, (0, None), 0)\
			(act_data[:,:,select_predict], corrupt_correct)
		pred_err = jax.vmap(dif, (0, None), 0)\
			(act_data[:,:,select_error], corrupt_error)

		pred_out = act_data[:,:,select_out] # dimensions (POP, 72, 3)
		#def regress(test, true):
			## this function slows things down considerably...
			## Could make faster by implementing manual fast least squares.
			## also, it needs to be float32 for numerical stability.
			#(w,resid,rank,sing) = jnp.linalg.lstsq(test, true) # test * w = true
			#msk = jnp.isnan(w) # linear regression might not work!
			#w = jnp.where(jnp.isnan(w), 1.0, w)
			#pred = test @ w # (72, 3)
			#err = jnp.sum((pred-true)**2)
			#return err
		#try:
			#pred_out = jax.vmap(regress, (0, None), 0)\
				#(pred_out, out_correct)
		#except Exception as e:
			#print(traceback.format_exc())
			#pred_out = jnp.ones((POP,), dtype=DTYPE32) # all the same prediction error then

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
		pred_correct = jnp.sum(pred_correct, (1,2))
		pred_err = jnp.sum(pred_err, (1,2))

		objective = pred_correct + 1.5*pred_err #+ 2.0*pred_out
		print('correct', jnp.mean(pred_correct), 'err', jnp.mean(pred_err), 'out', jnp.mean(pred_out), 'LB', lb_eval[0], '/', jnp.mean(objective), jnp.min(objective))

		# objective = pred_correct
		return novelty_hist, objective
		# probably should put an energy term in there too....

def eval_gar(key, genome, rules, novelty_hist, do_novelty, lb_eval):
	global dev_index
	global dev_ntype
	global dev_location

	genome = jnp.clip(genome, 0.0, 1.0)
	rules = jnp.clip(rules, -1.0, 2.0)

	key, subkey = jax.random.split(key)
	rand_numbz = jax.random.uniform(subkey, (POP,NEURONS,DENDRITES,SYNAPSES), dtype=DTYPE)
	source = make_pop_jit(dev_index, dev_ntype, dev_location, rand_numbz, genome, rules)

	key, act_dend, act_neur, weight, dw = train(key, source, rules)
	novelty_hist, pop_eval = eval_(source, weight, rules, novelty_hist, do_novelty, lb_eval)
	return pop_eval, key, act_dend, act_neur, weight, dw, novelty_hist

#eval_gar_jit = jax.jit(eval_gar)


def test_speed(key):
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
	rules = jax.random.uniform( subkey, (POP, NEURTYPE, NUMRULES), dtype=DTYPE) / 100.0
	key, subkey = jax.random.split(key)
	rand_numbz = jax.random.uniform(subkey, (POP,NEURONS,DENDRITES,SYNAPSES))
	source = make_pop_jit(dev_index, dev_ntype, dev_location, rand_numbz, genome, rules)

	print('source in test_train_speed', source.shape)
	#print(jnp.squeeze(source[POP-1]))
	#print(jnp.squeeze(source[POP-1]) % NEURTYPE)
	print('checksum:', jnp.sum(jnp.squeeze(source[POP-1]) % NEURTYPE))

	N = 10
	start = time.time()
	for i in range(N):
		train(key, source, rules)
	end = time.time()
	print("train time per iter:", (end - start) / float(N))

	N = 10
	start = time.time()
	for i in range(N):
		make_pop_jit(dev_index, dev_ntype, dev_location, rand_numbz, genome, rules)
	end = time.time()
	print("make_pop time per iter:", (end - start) / float(N))

	novelty_hist = jnp.zeros((POP, 72*NEURONS))
	N = 10
	start = time.time()
	for i in range(N):
		eval_gar(key, genome, rules, novelty_hist, False)
	end = time.time()
	print("eval_gar time per iter:", (end - start) / float(N))
	return key

#print('testing training & eval speed')
#key = test_speed(key)
#quit();

animate = True
plot_rows = 3
plot_cols = 3
figsize = (19, 10)
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
		# cbar[r][c].update_normal(im[r][c]) # probably does nothing
		axs[r,c].set_title(name)


def iterate_evo(key, genome, rules, do_plot, novelty_hist, do_novelty, iter_k, leaderboard, leaderboard_eval):
	global initialized
	pop_eval, key, act_dend, act_neur, weight, dw, novelty_hist = \
		eval_gar(key, genome, rules, novelty_hist, do_novelty, leaderboard_eval)

	if do_plot:
		# things to plot: act_dend, act_neur, weight,
		# genome, rules
		plot_tensor(0, 0, jnp.reshape(jnp.average(act_dend, 0), (int(NEURONS/NEURTYPE), NEURTYPE*DENDRITES)), 'mean act_dend', 0.0, 1.0)
		plot_tensor(0, 1, jnp.reshape(jnp.average(act_neur, 0), (int(NEURONS/NEURTYPE), NEURTYPE)), 'mean act_neur', 0.0, 2.5)
		plot_tensor(0, 2, jnp.reshape(jnp.average(weight, 0), (NEURONS, DENDRITES*SYNAPSES)), 'mean weight', 0.0, 0.5)

		plot_tensor(1, 0, jnp.reshape(jnp.average(genome[:,1,:,:], 0), (NEURTYPE, NEURTYPE)), 'mean genome', 0.0, 0.2)
		plot_tensor(1, 1, jnp.reshape(jnp.average(rules, 0), (NEURTYPE, NUMRULES)), 'mean rules', -0.25, 0.25)
		plot_tensor(1, 2, jnp.reshape(jnp.average(dw, 0), (NEURONS, DENDRITES*SYNAPSES)), 'mean dw', -1/40.0, 1/40.0)

		# need to get standard deviations in here too
		plot_tensor(2, 0, jnp.reshape(jnp.std(genome[:,1,:,:], 0), (NEURTYPE, NEURTYPE)), 'std genome', 0.0, 0.2)
		plot_tensor(2, 1, jnp.reshape(jnp.std(rules, 0), (NEURTYPE, NUMRULES)), 'std rules', -0.25, 0.25)
		plot_tensor(2, 2, jnp.reshape(jnp.std(dw, 0), (NEURONS, DENDRITES*SYNAPSES)), 'std dw', -1/40.0, 1/40.0)

		#print('act_dend[10]', jnp.reshape(act_dend[10], (int(NEURONS/NEURTYPE), NEURTYPE*DENDRITES)))
		#print('rules[10]', rules[10])
		#print('dw[1]', jnp.reshape(dw[10], (NEURONS, DENDRITES*SYNAPSES)))

		fig.tight_layout()
		fig.canvas.draw()
		fig.canvas.flush_events()
		if animate:
			initialized = True
		plt.savefig(f'images/gar_%d.png'%iter_k)

	indx = jnp.argsort(pop_eval) # ascending order.
	# so we want the first elements (smallest error)

	old_genome = jnp.reshape(genome, (POP, SIZE_GENOME))
	old_rules = jnp.reshape(rules, (POP, SIZE_RULES))
	old_ = jnp.concatenate((old_genome, old_rules), 1)

	if not do_novelty and jnp.sum(jnp.isnan(pop_eval) + jnp.isinf(pop_eval)) == 0:
		leaderboard = jnp.concatenate((leaderboard, old_), 0)
		leaderboard_eval = jnp.concatenate((leaderboard_eval, pop_eval), 0)
		lb_indx = jnp.argsort(leaderboard_eval) # again, ascending.
		lb_indx = lb_indx[0:int(POP)]
		leaderboard = leaderboard[lb_indx]
		leaderboard_eval = leaderboard_eval[lb_indx]
		#print('leaderboard best so far', leaderboard_eval[0])
		# printed in eval_()

	indx_ = jnp.arange(0, SIZE_ALL)
	def make_kid(mother, father, noise_mask, noise_mutation):
		crossover_genome = np.random.randint(0, SIZE_ALL)
		nu = jnp.where(indx_ < crossover_genome, old_[mother, :], old_[father, :])
		# nu = old_[mother, :]
		# add in noise: noise_mask is uniform, noise_mutation
		mutation_thresh = 1.0 - (3.0/SIZE_ALL)
		nu = jnp.where(noise_mask > mutation_thresh, noise_mutation + nu, nu)
		# and add in depreciation / nonsense mutation
		dep_thresh = 1.0 - 1.0/(SIZE_ALL)
		nu = jnp.where(noise_mask < dep_thresh, nu, nu * 0.33)
		return nu

	key, subkey = jax.random.split(key)
	mother = indx[jax.random.randint(subkey, (POP,), 0, int(POP/2) )]
	key, subkey = jax.random.split(key)
	father = indx[jax.random.randint(subkey, (POP,), 0, int(POP/2) )]

	#print('mother', mother.shape, mother)
	#print('father', father.shape, father)

	key, subkey = jax.random.split(key)
	noise_mask = jax.random.uniform(subkey, (POP, SIZE_ALL), dtype=DTYPE)
	key, subkey = jax.random.split(key)
	noise_mutation = jax.random.normal(subkey, (POP, SIZE_ALL), dtype=DTYPE) * 0.025 # guess!

	new_ = jax.vmap(make_kid, (0, 0, 0, 0), 0)(mother, father, noise_mask, noise_mutation)

	genome = jnp.reshape(new_[:, 0:SIZE_GENOME], (POP,3,NEURTYPE,NEURTYPE))
	rules = jnp.reshape(new_[:,SIZE_GENOME:SIZE_ALL], (POP,NEURTYPE,NUMRULES))
	# promote this to float32 for mutation etc.

	# allow for slight negatives (disconnect)
	genome = jnp.clip(genome, -0.01, 2.0)
	rules = jnp.clip(rules, -0.5, 0.5)

	# need to add in rules normalization -- if some neurons are not active,
	# bump their rules up a lil bit. act_neur is (POP, NEURONS)
	act_neur = jnp.reshape(act_neur, (POP * int(NEURONS / NEURTYPE), NEURTYPE))
	activity = jnp.mean(act_neur, 0)

	key, subkey = jax.random.split(key)
	noise = jax.random.uniform(subkey, (POP,), dtype=DTYPE)
	up = jnp.outer(noise, activity < 0.05)
	key, subkey = jax.random.split(key)
	noise = jax.random.uniform(subkey, (NUMRULES,), dtype=DTYPE) *\
		jnp.array((1,1,1,0), dtype=DTYPE)
	# jnp's outer compresses inputs to 1d; will need to vmap.
	def mull(a, b):
		return a * b * 0.005
	up = jax.vmap(mull, (None, 0), -1)(up, noise)

	key, subkey = jax.random.split(key)
	noise = jax.random.uniform(subkey, (POP,), dtype=DTYPE)
	down = jnp.outer(noise, activity > 0.7)
	key, subkey = jax.random.split(key)
	noise = jax.random.uniform(subkey, (NUMRULES,), dtype=DTYPE) *\
		jnp.array((1,1,1,0), dtype=DTYPE)
	down = jax.vmap(mull, (None, 0), -1)(down, noise)
	rules = rules + up - down
	rules = jnp.array(rules, dtype=DTYPE)

	return key, genome, rules, novelty_hist, leaderboard, leaderboard_eval

try:
	loaded = torch.load('gar_noout.pt')
	genome = loaded['genome']
	rules = loaded['rules']
	novelty_hist = loaded['novelty_hist'].cuda(device=torch_device)
	leaderboard = loaded['leaderboard']
	leaderboard_eval = loaded['leaderboard_eval']

	genome = to_jax(genome)
	rules = to_jax(rules)
	leaderboard = to_jax(leaderboard)
	leaderboard_eval = to_jax(leaderboard_eval)

	if False:
		genome = jnp.reshape(leaderboard[:, 0:SIZE_GENOME], (POP,3,NEURTYPE,NEURTYPE))
		rules = jnp.reshape(leaderboard[:,SIZE_GENOME:SIZE_ALL], (POP,NEURTYPE,NUMRULES))

	print('loaded a saved genome / rules file!')
except Exception as e:
	print(traceback.format_exc())
	print('couldnt load model genome / rules files.  making new random ones.')
	key, subkey = jax.random.split(key)
	genome = jax.random.uniform(subkey, (POP, 3, NEURTYPE, NEURTYPE), dtype=DTYPE) / 100.0
	key, subkey = jax.random.split(key)
	rules = jax.random.uniform( subkey, (POP, NEURTYPE, NUMRULES), dtype=DTYPE) / 100.0

#novelty_hist = torch.zeros((POP, 24*NEURONS), device=torch_device)
#leaderboard = jnp.zeros((POP, SIZE_ALL), dtype=DTYPE32)
#leaderboard_eval = jnp.ones((POP,), dtype=DTYPE32) * 1e6

do_novelty = False

for k in range(200000):
	#rules = torch.zeros((POP, NEURTYPE, 3))
	#rules[:, 1, 1] = 0.5 # just the cube learning rule.  'should work'
	#rules = to_jax(rules)
	key, genome, rules, novelty_hist, leaderboard, leaderboard_eval = iterate_evo(\
		key, genome, rules, k%4==3, novelty_hist, do_novelty, k, leaderboard, leaderboard_eval)
	if k % 50 == 49:
		genome_t = to_torch(genome)
		rules_t = to_torch(rules)
		leaderboard_t = to_torch(leaderboard)
		leaderboard_eval_t = to_torch(leaderboard_eval)
		m = {'genome': genome_t, 'rules': rules_t, 'leaderboard':leaderboard_t, 'leaderboard_eval':leaderboard_eval_t, 'novelty_hist':novelty_hist}
		torch.save(m, 'gar_noout.pt')
		print('saving gar_noout.pt')
	if k % 70 >= 60:
		do_novelty = True
	else:
		do_novelty = False

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
