# test out vector arithmetic in a transformer! 
import torch
import matplotlib.pyplot as plt


batch_size = 1
n_ctx = 12
n_symb = 10

def gen_data(): 
	# task: select the two symbols that match
	# subtract the difference between their position encodings
	# symbol encodings -- random [0, 1)
	x = torch.rand(batch_size, n_ctx, n_symb+5)
	# matches
	i = torch.randint(n_ctx, [batch_size])
	j = torch.randint(n_ctx-1, [batch_size]) + 1
	j = torch.remainder(i + j, n_ctx)
	x[:,i,:] = x[:,j,:]
	# positions
	x[:,:,-5] = torch.randint(128, [batch_size, n_ctx])
	x[:,:,-4] = x[:,:,-5] / 4
	x[:,:,-3] = x[:,:,-4] / 4
	x[:,:,-2] = x[:,:,-3] / 4
	# labels for us humans ;)
	x[:,:,-1] = 0
	x[:,i,-1] = 1
	x[:,j,-1] = 1
	y = torch.abs(x[:,i,-5] - x[:,j,-5])
	return x,y

x,y = gen_data()
print(y)
plt.imshow(x[0,:,:].numpy())
plt.clim(0,2)
plt.colorbar()
plt.show()
