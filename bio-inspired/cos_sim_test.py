import torch as th
import numpy as np

n_iters = 2000
eps = 1e-3

# # try gaussian random numbers (hypersphere)
# 
# for d in range(2, 512, 4): 
# 
# 	c = th.zeros(n_iters)
# 
# 	for i in range(n_iters): 
# 		x = th.randn(d)
# 		y = th.randn(d)
# 		lx = th.dot(x,x).sqrt()
# 		ly = th.dot(y,y).sqrt()
# 		cos_sim = th.dot(x,y) / (lx * ly)
# 		c[i] = cos_sim
# 	
# 	orth = th.logical_and(x < eps, x > -1*eps)
# 	print(d, orth.sum(), c.abs().mean())
# 	
# # try uniform random numbers (e.g. a hypercube)
# 
# for d in range(2, 512, 4): 
# 
# 	c = th.zeros(n_iters)
# 
# 	for i in range(n_iters): 
# 		x = (th.rand(d) - 0.5) * 2
# 		y = (th.rand(d) - 0.5) * 2
# 		lx = th.dot(x,x).sqrt()
# 		ly = th.dot(y,y).sqrt()
# 		cos_sim = th.dot(x,y) / (lx * ly)
# 		c[i] = cos_sim
# 	
# 	orth = th.logical_and(x < eps, x > -1*eps)
# 	print(d, orth.sum(), c.abs().mean())

torch_device = 0
print("torch cuda devices", th.cuda.device_count())
print("torch device", th.cuda.get_device_name(torch_device))
th.cuda.set_device(torch_device)
th.set_default_tensor_type('torch.cuda.FloatTensor')
th.set_float32_matmul_precision('high') # desktop.

n = 32000
c = th.zeros(512)
for d in range(2, 512, 1): 
	x = th.randn(n,d)
	y = th.randn(n,d)
	lx = th.einsum("bc,bc->b", x, x).sqrt()
	ly = th.einsum("bc,bc->b", y, y).sqrt()
	cos_sim = th.einsum("bc,bc->b", x, y) / (lx * ly)
	print(d, cos_sim.abs().mean())
	c[d] = cos_sim.abs().mean()
	
import matplotlib.pyplot as plt
plt.plot(np.arange(0,512), c.cpu().numpy())
plt.show()

# confident that this is correct. 

# how many eps-orthogonal vectors can we fit?
d = 2048
maxn = 20000
basis = th.zeros(maxn, d)
nb = 0
eps = 0.08 # average abs cosine distance
while(nb < maxn):
	y = th.randn(d)
	ly = th.dot(y, y).sqrt()
	ok = True
	if nb >0:
		lb = th.einsum("bc,bc->b",basis[:nb,:], basis[:nb,:]).sqrt()
		cos_sim = th.einsum("bc,c->b", basis[:nb,:], y) / (lb * ly)
		mx = th.max(th.abs(cos_sim))
		if mx > eps: 
			ok = False
	if ok: 
		basis[nb,:] = y
		nb = nb + 1
		print(nb)
		
# it seems that, when eps > mean eps for a given distribution, 
# then, yes, you can get an large number of bases that 
# are discriminable based on dot product metric.
# with eps = 0.15 and d = 512, get ~10k bases.
