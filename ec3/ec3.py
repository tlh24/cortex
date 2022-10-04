import math
import torch as th
import torch.nn as nn
import fcntl, os, select, sys
import pdb
import matplotlib.pyplot as plt
import numpy as np

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
import subprocess 
ocamlLogoPath = "./_build/default/program.exe"
sp = subprocess.Popen(ocamlLogoPath, shell=False, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=False)

# write a program to the subprocess
# home_dir = os.path.expanduser("~")
# lib_dir = os.path.join(home_dir, "Dropbox/work/ML/mypaint/")
sys.path.append("./_build/default/")
import logod_pb2
lp = logod_pb2.Logo_program()
lp.id = 1
lp.prog = """
(
	loop 0 4 (
		move (ul * 2) (ua / 4) )
)
"""
lp.resolution = 32
print("sending", lp.__str__())
sp.stdin.write(lp.SerializeToString())
sp.stdin.flush()
print("\n")
#output = p.communicate(input=lp.SerializeToString())[0]
#print(output) communicate closes the pipe


fd = sp.stdout.fileno()
fl = fcntl.fcntl(fd, fcntl.F_GETFL)
fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)

streams = [ sp.stdout ]
temp0 = []
readable, writable, exceptional = select.select(streams, temp0, temp0, 5)
if len(readable) == 0:
    raise Exception("Timeout of 5 seconds reached!")
 
buff = bytearray(4096)
numberOfBytesReceived = sp.stdout.readinto(buff)
if numberOfBytesReceived <= 0:
    raise Exception("No data received!")
# convert the bytearray to a protobuf.. 
result = logod_pb2.Logo_result()
try: 
	result.ParseFromString(buff)
	print(result)
except Exception as inst: 
	print("error! ocaml logo interpreter returned ", buff)


buff2 = bytearray(4096)
n = sp.stderr.readinto(buff2)
if n != 1024: 
	print("unexpected number of bytes in image, ", n)
a = np.frombuffer(buff2, dtype=np.uint8, count=n)
a = np.reshape(a, (32, 32))
plt.imshow(a)
plt.colorbar()
plt.show()
