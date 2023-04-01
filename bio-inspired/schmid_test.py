# want to test horizontal and vertical bars source separation 
# from Schmidhuber 1999

import math
import numpy as np
from numpy.random import rand
from numpy.random import randn
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision.utils import save_image

from numba import jit
import pdb
import os
import time

## being lazy ... these should be arguments.  
batch_size = 64 
log_interval = 200
dim = 8
dim2 = dim*dim
nlatent = 20
nhid = 96

epochs = 40
nepoch = 5000

@jit(nopython=True)
def make_stimulus_np(nbatch):
	s = np.zeros((nbatch,dim,dim))
	for b in range(nbatch):
		for v in range(0,dim):
			if rand() < (1.0 / dim):
				for k in range(0,dim):
					s[b,k,v] = 1.0
		for h in range(0,dim):
			if rand() < (1.0 / dim):
				for k in range(0,dim):
					s[b,h,k] = 1.0
	return s

# use truly gaussian distributed vertical and horizontal random variables. 
# this doesn't work ..
@jit(nopython=True)
def make_stimulus_np_gauss(nbatch):
	s = np.zeros((nbatch,dim,dim))
	def make_randn():
		q = (randn() + 1.0) / (2*dim)
		return max(q, 0.0)
	for b in range(nbatch):
		for v in range(0,dim):
			q = make_randn()
			for k in range(0,dim):
				s[b,k,v] += q
		for h in range(0,dim):
			q = make_randn()
			for k in range(0,dim):
				s[b,h,k] += q
	return s

def make_stimulus(nbatch):
	return torch.from_numpy(make_stimulus_np(nbatch)).to(torch.float)

def check_stimulus():
	for i in range(10):
		s = make_stimulus(1)
		s = torch.squeeze(s); 
		im = plt.imshow(s.numpy())
		plt.colorbar(im)
		plt.show()

device = torch.device("cuda")

class VAE(nn.Module):
	def __init__(self):
		super(VAE, self).__init__()

	# adding two hidden layers doesn't improve things
	# possibly because of initialization -- ?
		self.fc1 = nn.Linear(dim2, nhid)
		# self.fc12 = nn.Linear(nhid, nhid)
		self.fc21 = nn.Linear(nhid, nlatent)
		self.fc22 = nn.Linear(nhid, nlatent)
		self.fc3 = nn.Linear(nlatent, nhid)
		# self.fc31 = nn.Linear(nhid, nhid)
		self.fc4 = nn.Linear(nhid, dim2)

	def encode(self, x):
		h1 = F.leaky_relu(self.fc1(x), negative_slope=0.02)
		# h2 = F.leaky_relu(self.fc12(h1), negative_slope=0.02)
		return self.fc21(h1), self.fc22(h1)

	def reparameterize(self, mu, logvar):
		std = torch.exp(0.5*logvar)
		eps = torch.randn_like(std)
		return mu + eps*std

	def decode(self, z):
		h3 = F.leaky_relu(self.fc3(z), negative_slope=0.02)
		# h4 = F.leaky_relu(self.fc31(h3), negative_slope=0.02)
		return torch.sigmoid(self.fc4(h3))

	def forward(self, x):
		mu, logvar = self.encode(x.view(-1, dim2))
		z = self.reparameterize(mu, logvar)
		return self.decode(z), mu, logvar
	
	def get_l1_w(self):
		return self.fc1.weight.detach()
	
	def get_l2_w(self):
		return self.fc21.weight.detach()

torch.manual_seed((os.getpid() * int(time.time())) % 123456789)
model = VAE().to(device)
# optimizer = optim.AdamW(model.parameters(), lr=1e-3)
optimizer = optim.AdamW(model.parameters(), lr = 5e-5, 
								betas=(0.9, 0.95), weight_decay=5e-3)
model.get_l1_w()

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar, beta):
	BCE = F.binary_cross_entropy(recon_x, x.view(-1, dim2), reduction='sum')

	# see Appendix B from VAE paper:
	# Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
	# https://arxiv.org/abs/1312.6114
	# 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
	KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

	return BCE + beta * KLD

def train(epoch):
	model.train()
	train_loss = 0
	save_w1 = torch.zeros(nepoch, 10*dim2)
	save_w2 = torch.zeros(nepoch, 10*nlatent)
	beta = (epoch + 0.01) / (2.0*epochs) + 0.25
	for k in range(nepoch):
		data = make_stimulus(batch_size).view(-1, dim2)
		data = data.to(device)
		optimizer.zero_grad()
		recon_batch, mu, logvar = model(data)
		loss = loss_function(recon_batch, data, mu, logvar, beta)
		loss.backward()
		train_loss += loss.item()
		optimizer.step()
		save_w1[k] = model.get_l1_w()[:10,:].reshape(10*dim2)
		save_w2[k] = model.get_l2_w()[:,:10].reshape(10*nlatent)
		if k % log_interval == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
					epoch, k, nepoch,
					100. * k / nepoch,
					loss.item() / len(data)))
	print('====> Epoch: {} Average loss: {:.4f}'.format(
			epoch, train_loss / (len(data) * nepoch)))
	# plot some weight trajectories. 
	#fig,axs = plt.subplots(1, 2, figsize=(16,9))
	#for k in range(20):
		#j = int(math.floor(rand() * 10*dim2))
		#axs[0].plot(range(nepoch), torch.squeeze(save_w1[:,j]).numpy())
		#j = int(math.floor(rand() * 10*nlatent))
		#axs[1].plot(range(nepoch), torch.squeeze(save_w2[:,j]).numpy())
	#plt.show()
	 
def test(epoch):
	model.eval()
	test_loss = 0
	mu_save = torch.zeros(batch_size*1000, nlatent)
	with torch.no_grad():
		for i in range(1000):
			data = make_stimulus(batch_size).view(-1, dim2)
			data = data.to(device)
			recon_batch, mu, logvar = model(data)
			mu_save[i*batch_size:(i+1)*batch_size, :] = mu
			test_loss += loss_function(recon_batch, data, mu, logvar, 1.0).item()
			if i == 0:
				n = min(data.size(0), 12)
				data = data[:n].view(n, 1, dim, dim)
				recon_batch = recon_batch[:n].view(n, 1, dim, dim)
				comparison = torch.cat((data, recon_batch))
				save_image(comparison.cpu(),
							'reconstruction_' + str(epoch) + '.png', nrow=n)
	test_loss /= 1000 * len(data)
	print('====> Test set loss: {:.4f}'.format(test_loss))
	
	# make the latent variable mu covariance matrix 
	mu_cov = torch.cov(torch.transpose(mu_save, 0, 1))
	save_image(mu_cov.cpu(), 'mu_cov'+str(epoch)+'.png')
	
	# look at the latent variable representations. 
	latent_rep = torch.zeros(nlatent, dim2).to(device)
	for i in range(nlatent):
		z = torch.zeros(nlatent).to(device); 
		z[i] = mu_cov[i,i] * 2.0
		latent_rep[i] = model.decode(z)
	latent_rep = torch.reshape(latent_rep, (nlatent, 1, dim, dim))
	save_image(latent_rep.cpu(), 'latent_rep'+str(epoch)+'.png')


if __name__ == "__main__":
	os.chdir("results")
	os.system("rm *.png")
	for epoch in range(1, epochs + 1):
		train(epoch)
		test(epoch)
		with torch.no_grad():
			sample = torch.randn(64, nlatent).to(device)
			sample = model.decode(sample).cpu()
			save_image(sample.view(64, 1, dim, dim),
							'sample_' + str(epoch) + '.png')
	# make the images easier to view.
	os.system("mogrify -scale 2000% *.png")
		
