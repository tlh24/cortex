import torch as th
import time

image_count = 5*2048
image_res = 30

torch_device = 0
print("torch cuda devices", th.cuda.device_count())
print("torch device", th.cuda.get_device_name(torch_device))
th.cuda.set_device(torch_device)
th.set_default_tensor_type('torch.cuda.FloatTensor')

dbf = th.ones(image_count, image_res, image_res)

tic = time.perf_counter()

for i in range(100000): 
	a = th.randn(image_res, image_res)
	d = th.sum((dbf - a)**2, (1,2))
	mindex = th.argmin(d)
	dist = d[mindex]

toc = time.perf_counter()
print(f"this took {toc-tic} seconds")
# really long on the CPU
# 27 seconds on the GPU
# 45 sec w ocaml
# 24.85 sec if only do GC ever 10th iter
# 23.84 every 30
# 23.56 every 100
