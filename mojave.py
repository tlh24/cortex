import math
import numpy as np
from numpy.random import randn
from numpy.random import rand
import torch
import matplotlib.pyplot as plt
import pdb

"""
Stimulus function
parameters: 
dim, outside dimensions (e.g. 5)
latent, number of latents (columns or rows 'on') 
output: [ nbatch x dim x dim ]  tensor
""" 
# @jit(nopython=True) # numba decorator
def make_stimulus(nbatch, dim, nlatent):
	s = np.zeros((nbatch,dim,dim)) 
	for b in range(nbatch):
		# random draw of colums with replacement. 
		cols = np.floor(rand(nlatent) * dim).astype(int)
		for c in cols:
			for k in range(0,dim):
				s[b,k,c] = 1.0
		rows = np.floor(rand(nlatent) * dim).astype(int)
		for r in rows:
			for k in range(0,dim):
				s[b,r,k] = 1.0
	return s
	
def test_make_stimulus():
	nbatch = 5
	s = make_stimulus(nbatch, 8, 3)
	for b in range(nbatch):
		im = plt.imshow(s[b,:,:])
		plt.colorbar(im)
		plt.show()
		
		
class Neuron_vector:
  def __init__(self, output_number): # nn is number of output neurons
    self.output_number = output_number
    self.input_number = 0 # must be updated.
    self.input_ = [] # input is a concatenation of other Neuron_vectors
    self.w_ = torch.zeros(1) # change this later during development
    self.output_function = lambda x:x
    self.internal_state = torch.zeros(5) # for dopamine, serotonin, etc... 

  def forward(self, all_neurons):
    # maths, e.g. matrix mul. 
    soma = torch.matmul(self.w_, torch.cat(input_, dim=0))
    # output function
    return output_function(soma)

  def add_inputs(self, input_neuron_vec):
    self.input_ += [input_neuron_vec]

  def grow_synapses(self):
    ### call after instantiating and connecting all neurons to make synapses
    k = 0
    for nv in input_:
      k += len(nv)
    self.input_number = k
    self.w_ = torch.mul(torch.randn(output_number, k), 1/math.sqrt(k))
    # may want to change this later, 'smarter'
		
test_make_stimulus()
