import torch
import time

nimg = 6*2048

torch_device = 0
print("torch cuda devices", torch.cuda.device_count())
print("torch device", torch.cuda.get_device_name(torch_device))

sta = time.perf_counter()

x = torch.zeros(nimg, 30, 30)

for i in range(nimg): 
	k = torch.ones(30, 30) 
	x[i,:,:] = k
	
	
fin = time.perf_counter()

y = x.cuda()
z = torch.sum(y)

print("elapsed time:", fin-sta)
print(z)
	
