import math
from math import sin, cos, pi
import torch as th
import torch.nn as nn
import fcntl, os, select, sys
import pdb
import matplotlib.pyplot as plt
import numpy as np
import subprocess 

import logod_pb2
import xf

class EqualLR:
	def __init__(self, name):
		self.name = name

	def compute_weight(self, module):
		weight = getattr(module, self.name + '_orig')
		fan_in = weight.data.size(1) * weight.data[0][0].numel()
		# print("compute_weight ", weight.size())

		return weight * sqrt(2 / fan_in)
	
	# Idea: rather than changing the intialization of the weights, 
	# here they just use N(0, 1), and dynamically scale the weights by 
	# sqrt(2 / fan_in), per He 2015. 
	# updates are applied to the unscaled weights (checked), 
	# which means that the gradient updates are also scaled by sqrt(2/fan_in)

	@staticmethod
	def apply(module, name):
		fn = EqualLR(name)

		weight = getattr(module, name)
		del module._parameters[name]
		module.register_parameter(name + '_orig', nn.Parameter(weight.data))
		module.register_forward_pre_hook(fn)

		return fn

	def __call__(self, module, input):
		weight = self.compute_weight(module)
		setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
	EqualLR.apply(module, name)

	return module

class EqualConv2d(nn.Module):
	def __init__(self, *args, **kwargs):
		super().__init__()

		conv = nn.Conv2d(*args, **kwargs)
		conv.weight.data.normal_()
		conv.bias.data.zero_()
		self.conv = equal_lr(conv)

	def forward(self, input):
		return self.conv(input)


class EqualLinear(nn.Module):
	def __init__(self, in_dim, out_dim):
		super().__init__()

		linear = nn.Linear(in_dim, out_dim)
		linear.weight.data.normal_()
		linear.bias.data.zero_()

		self.linear = equal_lr(linear)

	def forward(self, input):
		return self.linear(input)
	

class Reshape(nn.Module):
	def __init__(self, *args):
		super(Reshape, self).__init__()
		self.shape = args

	def forward(self, x):
		return x.view(self.shape)
	
class ConvBlock(nn.Module):
	def __init__(
		self,
		in_channel,
		out_channel,
		kernel_size,
		padding,
		kernel_size2=None,
		padding2=None,
	):
		super().__init__()

		pad1 = padding
		pad2 = padding
		if padding2 is not None:
			pad2 = padding2

		kernel1 = kernel_size
		kernel2 = kernel_size
		if kernel_size2 is not None:
			kernel2 = kernel_size2

		self.conv1 = nn.Sequential(
			EqualConv2d(in_channel, out_channel, kernel1, padding=pad1),
			nn.LeakyReLU(0.2),
		)
		self.conv2 = nn.Sequential(
			EqualConv2d(out_channel, out_channel, kernel2, padding=pad2),
			nn.AvgPool2d(2), # downsample.
			nn.LeakyReLU(0.2),
		)

	def forward(self, input):
		out = self.conv1(input)
		out = self.conv2(out)
		return out
	
def requires_grad(model, flag=True):
	for p in model.parameters():
		p.requires_grad = flag
		
#Ok, first step is to generate some images. 
ocamlLogoPath = "./_build/default/program.exe"
sp = subprocess.Popen(ocamlLogoPath, shell=False, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=False)

def make_nonblock(fd): 
	fl = fcntl.fcntl(fd, fcntl.F_GETFL)
	fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)
	
make_nonblock(sp.stdout.fileno())
make_nonblock(sp.stderr.fileno())


# write a program to the subprocess

square_prog = """
(
	loop 0 4 (
		move (ul * 2) (ua / 4) )
)
"""
g_lp_count = 10

def ocaml_run(prog_str):
	global g_lp_count
	lp = logod_pb2.Logo_program()
	lp.id = g_lp_count
	g_lp_count = g_lp_count + 1
	lp.prog = prog_str
	lp.resolution = 32
	q = lp.SerializeToString()
	# print(f"sending len {len(q)}", lp)
	sp.stdin.write(q)
	sp.stdin.flush()
	print("\n")

	streams = [ sp.stdout ]
	temp0 = []
	readable, writable, exceptional = select.select(streams, temp0, temp0, 5)
	if len(readable) == 0:
		raise Exception("Timeout of 5 seconds reached!")
	
	buff = bytearray(256)
	nrx = sp.stdout.readinto(buff)
	if nrx <= 0:
		print("No data received on stdout! stderr: ", sp.stderr.peek())
	# convert the bytearray to a protobuf.. 
	result = logod_pb2.Logo_result()
	try: 
		result.ParseFromString(bytes(buff[0:nrx]))
		print(result)
	except Exception as inst: 
		print("ParseFromString; stderr: ", sp.stderr.peek())
		# print("stdout:", bytes(buff))
		# print(f"error! could not parse protobuf; saving {nrx} bytes to bad_buffer.pb")
		# print("debug with cat bad_buffer.pb | protoc --decode Logo_result --proto_path ./_build/default/ logod.proto")
		# fil = open("bad_buffer.pb", "wb")
		# fil.write(bytes(buff[0:nrx]))
		# fil.close()
		# print(inst)
		# # make a comparison
		# lr = logod_pb2.Logo_result()
		# lr.id = lp.id
		# lr.stride = 32
		# lr.width = 32
		# lr.height = 32
		# seg = lr.segs.add()
		# seg.x0 = 0.0
		# seg.y0 = 0.0
		# seg.x1 = 1.0
		# seg.y1 = 0.0
		# lr.cost = 1.0
		# q = lr.SerializeToString()
		# print(f"should be {len(q)}: ", lr.SerializeToString())
		# fil = open("good_buffer.pb", "wb")
		# fil.write(lr.SerializeToString())
		# fil.close()


	buff2 = bytearray(1500)
	n = sp.stderr.readinto(buff2)
	if n != 1024: 
		# print("unexpected number of bytes in image, ", n)
		return False
	else: 
		# do something else with this image..
		# a = np.frombuffer(buff2, dtype=np.uint8, count=n)
		# a = np.reshape(a, (32, 32))
		# plt.imshow(a)
		# plt.colorbar()
		# plt.show()
		return True

ocaml_run(square_prog)
ocaml_run("( move 1 1 )")


tokens = [" ", "(", ")",";","+ ","- ","* ","/ ", 
			 "move ","loop ","v","ua ","ul ", 
			 "0 ","1 ","2 "
			 ] # 'eof'

tktype = [9, 0, 0, 1, 2, 2, 2, 2, 
			 3, 4, 5, 6, 6, 
			 7, 7, 7
			 ] # convert this to a one-hot as well
			# better for MLP encoder
			
		
toklen = len(tokens)
typlen = 10
poslen = 6
indim = toklen + typlen + poslen*2
n_ctx = 32
bs = 1
xfrmr = xf.Transformer(n_ctx, 128, 8, 8)
	# n_ctx, width, layers, heads
encoder = nn.Linear(indim, 128)
	# just a simple linear layer to put into transformer latent space. 
	# how are these weights initialized?? 
gelu = nn.GELU()

# positional encoding. 
posenc = th.zeros(n_ctx, poslen*2)
for i in range(n_ctx): 
	for j in range(poslen):
		posenc[i,j*2+0] = sin((2*pi*i / n_ctx) * (j+1))
		posenc[i,j*2+1] = cos((2*pi*i / n_ctx) * (j+1))
		
posenc = posenc.expand(bs, n_ctx, poslen*2) 

prog = [0 for i in range(n_ctx)]
prog_human = list(map(lambda i : tokens[i], prog))
x = th.zeros(n_ctx, indim)
for i in range(n_ctx): # might be faster way? 
	x[i, prog[i]] = 1.0
	typ = tktype[prog[i]]
	x[i, toklen+typ] = 1.0
	# add positional encoding too. 

x = x.expand(bs, n_ctx, indim)
x[:,:,toklen + typlen : indim] = posenc
x = encoder(x)
x = gelu(x)
x = xfrmr(x)
# that works pretty seamlessly! 
# x.shape is now [1, 32, 128] 
# -- the output of a decoder section
# now need to add a decoder section to output tokens + continuations. 
# and then do search to boostrap..
# no, see cortex_bumps.py; we can start with an editor. 

# next task is to enumerate programs ('templates') then mess 'em up and ask the transformer to fix (or not)
# I guess do n-level depth-first search
logf = open("ec3_log.txt", "w")

def prog_to_string(prog): 
	q = list(map(lambda i : tokens[i], prog))
	q = " ".join(q)
	q = "( " + q + " )"
	return q
	
def check_formed(prog) : 
	q = prog_to_string(prog)
	return ocaml_run(q)
	

def enumerate_programs(n_levels, level, prog): 
	if level == n_levels: 
		if check_formed(prog) : 
			logf.write(prog_to_string(prog))
			logf.write("\n")
			return prog
		else:
			return None
	lst = []
	for i in range(toklen): 
		q = enumerate_programs(n_levels, level+1, prog + [i])
		if q is not None: 
			lst.append(q)
	if len(lst) > 0: 
		return lst
	else : 
		return None

valid = enumerate_programs(5, 0, [])
print(valid)

logf.close()
sp.terminate()
sys.exit(0)
