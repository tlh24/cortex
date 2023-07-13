# test out vector arithmetic in a transformer! 
import torch as th
from torch import nn, optim
import matplotlib.pyplot as plt
import clip_model
import pdb

batch_size = 64
n_ctx = 16
n_symb = 10
n_layers = 2
n_heads = 4
n_test = 2
n_iters = 10000
learning_rate = 2e-4
clip_grad = 0.1

doplot = False
doperturb = False
ringers = False

torch_device = 0
print("torch cuda devices", th.cuda.device_count())
print("torch device", th.cuda.get_device_name(torch_device))
th.cuda.set_device(torch_device)
th.set_default_tensor_type('torch.cuda.FloatTensor')
th.set_float32_matmul_precision('high') # desktop.

if doperturb: 
	perturb = th.randn(n_symb)
else:
	perturb = th.zeros(n_symb)

def gen_data(): 
	# task: select the two symbols that match
	# subtract the difference between their position encodings
	# background symbol encodings -- random [0, 0.5)
	x = th.zeros(batch_size, n_ctx, n_symb+5)
	x[:,:,0:n_symb] = th.randn(batch_size, n_ctx, n_symb) / 2
	# select matches
	i = th.randint(n_ctx, [batch_size])
	j = th.randint(n_ctx-1, [batch_size]) + 1
	j = th.remainder(i + j, n_ctx)
	k = th.arange(batch_size)
	# set the i = j symbol match
	x[k,i,:] = x[k,j,:]
	# set 'special' symbols: sinusoids with random phase.
	if(ringers):
		ss = th.sin(th.arange(0, n_symb).unsqueeze(0).expand(batch_size,n_symb) + th.rand(batch_size).unsqueeze(-1).expand(batch_size,n_symb)*6.28)*1
		cc = th.cos(th.arange(0, n_symb).unsqueeze(0).expand(batch_size,n_symb) + th.rand(batch_size).unsqueeze(-1).expand(batch_size,n_symb)*6.28)*1
		s2 = ss.clone()
		s2[:,:n_symb] = s2[:,:n_symb] + perturb
		x[k,i,0:n_symb] = ss[k,:]
		x[k,j,0:n_symb] = s2[k,:]
	# pdb.set_trace()
	# positions
	x[:,:,-5] = th.randint(128, [batch_size, n_ctx]) / 16.0
	x[:,:,-4] = x[:,:,-5] / 4
	x[:,:,-3] = x[:,:,-4] / 4
	x[:,:,-2] = x[:,:,-3] / 4
	# labels for us humans ;)
	x[:,:,-1] = 0
	x[k,i,-1] = 1
	x[k,j,-1] = 1
	y = th.abs(x[k,i,-5] - x[k,j,-5])
	return x,y
	
def test_plot (): 
	x,y = gen_data()
	print(y)
	plt.imshow(x[0,:,:].cpu().numpy())
	plt.clim(0,2)
	plt.colorbar()
	plt.show()
	
test_plot()

slowloss = 1.0

class Racoonizer(nn.Module): 
	def __init__(
		self,
		n_ctx:int,
		indim:int,
		embed_dim:int
		): 
		super().__init__()
		self.n_ctx = n_ctx
		self.indim = indim
		self.embed_dim = embed_dim
		
		self.encoder = nn.Linear(indim, embed_dim)
		self.gelu = clip_model.QuickGELU()
		
		self.trfmr = clip_model.Transformer(
			width = embed_dim, 
			layers = n_layers, 
			heads = n_heads, 
			attn_mask = None)
		
		self.decoder = nn.Linear(embed_dim, 1)
		
	def forward(self, x): 
		x = self.encoder(x)
		x = self.gelu(x)
		x = self.trfmr(x) # [bs, n_ctx, embed_dim]
		x = self.decoder(x) # [bs, n_ctx, 1]
		return x[:,:,0].sum(1)

testloss = th.zeros(n_test)

for j in range(n_test): 
	# reinit the model each time. 
	model = Racoonizer(n_ctx = n_ctx, 
						 indim = n_symb+4,
						 embed_dim = 64)

	lossfunc_mse = nn.MSELoss(reduction='mean')
	optimizer = optim.Adam(model.parameters(), lr=learning_rate)

	for i in range(n_iters): 
		x,targ = gen_data()
		
		if(doplot):
			plt.imshow(x[0,:,:].cpu().numpy())
			plt.clim(0,2)
			plt.colorbar()
			plt.show()
		
		xp = x[:,:,:-1] # remove the human annotations
		y = model(xp)
		loss = lossfunc_mse(y,targ)
		lossflat = th.sum(loss)
		lossflat.backward()
		th.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
		optimizer.step() 
		lossflat.detach()
		slowloss = 0.99*slowloss + 0.01 * lossflat.item()
		print(slowloss)
		
		# need to check that we can visually solve the problem! 
		
		if(doplot):
			# dotproduct between symbols
			w = th.einsum("btc,bsc->bts", x[:,:,0:n_symb], x[:,:,0:n_symb])
			plt.imshow(w[0,:,:].cpu().numpy())
			plt.colorbar()
			plt.show()
		# get the best matches
		# pdb.set_trace()

	testloss[j] = slowloss
	
print(testloss)
