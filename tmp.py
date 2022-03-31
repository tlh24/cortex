import torch
from torch import clamp, matmul, outer, mul, add, sub, ones, zeros, reshape

def tri3(n):
	return 2.0/3.0 * n * (n-1) * (n-2)

scaler = 1.0

def outerupt3(v):
	# outer product upper-triangle, but for terms with three, two, and one factor
	nin = v.shape[0]
	nout = int( nin * (nin+1) * (nin+2) / 6.0 + \
		nin * (nin+1) * 0.5 + nin )
	# this took me a surprisingly long time to figure out... sigh
	print(nin, nout)
	r = zeros(nout)
	e = 0
	for i in range(nin):
		for j in range(i, nin):
			for k in range(j, nin):
				print(e, i, j, k)
				r[e] = v[i] * v[j] * v[k] * scaler * scaler
				e = e + 1
	for i in range(nin):
		for j in range(i, nin):
			print(e, i, j)
			r[e] = v[i] * v[j] * scaler
			e = e + 1
	for i in range(nin):
		print(e, i)
		r[e] = v[i]
		e = e + 1
	return r

x = torch.arange(0, 5, 1)
y = outerupt3(x)
print(y)
