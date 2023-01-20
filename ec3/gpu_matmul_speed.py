import torch as th
import time

torch_device = 0
print("torch cuda devices", th.cuda.device_count())
print("torch device", th.cuda.get_device_name(torch_device))
th.cuda.set_device(torch_device)
th.set_default_tensor_type('torch.cuda.FloatTensor')

N = 2000

res = th.zeros(5)

for j in range(5): 
	a = th.randn(N,N)
	b = th.randn(N,N)
	d = th.zeros(N,N)
	tic = time.time()
	for i in range(N*4): 
		c = a @ b
		d = d + c
		a = a @ d
	toc = time.time()
	print("elapsed time:", toc-tic)
	res[j] = toc-tic
	time.sleep(19)
	
print("mean:", th.mean(res))

# with the monitor at 4k120: 
# GPU # 0 (3090) at 26-40C: 12.4598 (34% faster / 2080 is 53% slower)
# GPU # 1 (2080) at 60-70C: 19.0269
# GPU # 2 (2080) at 48-65C: 19.5551 (2.5-2.7% slower)

# with the monitor at 4k24: 
# GPU # 1 (2080) 65-72c:  19.0552 (slightly slower, within margin of error)
# with the monitor at 2k30: 
# 18.979, 18.9932
# back to 4k120: 
# 19.0269, 19.1146
# 0.4% slower.  Probably meaningless!
# Conclusion: no sense in adding another GPU to the system.  Seems that driving a monitor does not consume many resources; variance between cards is greater. 
