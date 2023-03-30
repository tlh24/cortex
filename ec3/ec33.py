import math
import mmap
import torch as th
from torch import nn, optim
import torch.cuda.amp
import matplotlib.pyplot as plt
import copy
from ctypes import *
import socket
import time
import clip_model
import argparse
import io
import os
import pdb

# import torch._dynamo as dynamo
# dynamo.config.verbose=True

batch_size = 256*3
image_res = 30
toklen = 30
poslen = 6
p_indim = toklen + 1 + poslen*2 
e_indim = 5 + toklen + poslen*2
p_ctx = 64

patch_size = 5
v_ctx = int((image_res / patch_size) ** 2 + 1)
vision_width = 256
prog_width = 128
vision_heads = 8
vision_layers = 4
prog_heads = 8
prog_layers = 6
embed_dim = 256

train_iters = 100000
learning_rate = 0.001 # maximum learning rate. scheduled.
# learning rate of 0.002 is unstable.  Should figure out why. 
weight_decay = 5e-6
nreplace = 0


parser = argparse.ArgumentParser(description='Transformer-based program synthesizer')
parser.add_argument("-b", "--batch_size", help="Set the batch size", type=int)
parser.add_argument("-d", "--dreaming", help="Set the model to dream", action="store_true")
args = parser.parse_args()
batch_size = args.batch_size
g_dreaming = args.dreaming
g_training = not g_dreaming
print(f"batch_size:{batch_size}")
print(f"dreaming:{g_dreaming}")

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
if g_dreaming: 
	sock.connect(('127.0.0.1', 4341))
else:
	sock.connect(('127.0.0.1', 4340))
sock.sendall(b"update_batch")
data = sock.recv(1024)
print(f"Received {data!r}")

def make_mmf(fname): 
	fd = open(fname, "r+b")
	return mmap.mmap(fd.fileno(), 0)

def read_mmap(mmf, dims): 
	mmf.seek(0)
	mmb = mmf.read()
	siz = len(mmb)
	mmb2 = (c_char * siz).from_buffer_copy(mmb)
	x = th.frombuffer(mmb2, dtype=th.float).clone()
	x = th.reshape(x, dims)
	return x
	
def write_mmap(mmf, data): 
	q = data.detach().cpu().numpy().tobytes()
	mmf.seek(0)
	n = mmf.write(q)
	return n

if g_dreaming: 
	mmapno = 1
else:
	mmapno = 0

edsiz = batch_size * e_indim * 4
os.system(f"fallocate -l {edsiz} editdiff_{mmapno}.mmap")
# the other mmaps are allocated by ocaml.
	
fd_bpro = make_mmf(f"bpro_{mmapno}.mmap")
fd_bimg = make_mmf(f"bimg_{mmapno}.mmap")
fd_bedts = make_mmf(f"bedts_{mmapno}.mmap")
fd_bedtd = make_mmf(f"bedtd_{mmapno}.mmap")
fd_editdiff = make_mmf(f"editdiff_{mmapno}.mmap")
fd_posenc = make_mmf(f"posenc_{mmapno}.mmap")
posenc = read_mmap(fd_posenc, [p_ctx, poslen*2])

torch_device = 0
print("torch cuda devices", th.cuda.device_count())
print("torch device", th.cuda.get_device_name(torch_device))
th.cuda.set_device(torch_device)
th.set_default_tensor_type('torch.cuda.FloatTensor')
# torch.set_float32_matmul_precision('high') # desktop.

def build_attention_mask(v_ctx, p_ctx):
	# allow the model to attend to everything when predicting an edit
	# causalty is enforced by the editing process.
	# see ec31.py for a causal mask.
	ctx = v_ctx + p_ctx
	mask = th.ones(ctx, ctx)
	return mask


class ecTransformer(nn.Module):
	def __init__(self, image_resolution: int, vision_width:int, patch_size:int,  prog_width:int, embed_dim:int, v_ctx:int, p_ctx:int, p_indim:int, e_indim:int): 
		super().__init__()
		self.v_ctx = v_ctx
		self.p_ctx = p_ctx
		self.p_indim = p_indim
		
		self.vit = clip_model.VisionTransformer(
			input_resolution = image_resolution, 
			patch_size = patch_size, 
			width = vision_width, 
			layers = 4, 
			heads = 8, 
			output_dim = embed_dim)

		self.vit_to_prt = nn.Linear(embed_dim, prog_width)
		
		self.encoder = nn.Linear(p_indim, prog_width)
		self.prt = clip_model.Transformer(
			width = prog_width, 
			layers = 6, 
			heads = 8, 
			attn_mask = build_attention_mask(v_ctx, p_ctx))
			
		self.prt_to_edit = nn.Linear(prog_width * (p_ctx + v_ctx), e_indim)
		self.ln_post = clip_model.LayerNorm(e_indim)
		self.gelu = clip_model.QuickGELU()
		self.tok_softmax = nn.Softmax(dim = 1)
	
	def forward(self, u, batch_a, batch_p): 
		# encode the image (we should only need to do this once??)
		q = th.zeros(6) # ! this will be parallelized !
		vx = self.vit(batch_a) # x is size [bs, v_ctx, 256] 
		q[0] = th.std(vx)
		vx = self.vit_to_prt(vx)
		q[1] = th.std(vx)
		# vx = gelu(vx) # ? needed ? 

		px = self.encoder(batch_p)
		q[2] = th.std(px)
		vxpx = th.cat((vx, px), dim = 1)
		q[3] = th.std(vxpx)
		# x = vxpx * mask
		x = self.prt(vxpx) # bs, v_ctx + p_ctx, prog_width
		q[4] = th.std(x)
		x = th.reshape(x, (-1,(v_ctx + p_ctx)*prog_width))
		# batch size will vary with dataparallel
		x = self.prt_to_edit(x)
		q[5] = th.std(x)
		# x = self.ln_post(x) # scale the inputs to softmax
		# x = self.gelu(x)
		x = th.cat((self.tok_softmax(x[:,0:4]),
				  self.tok_softmax(x[:,4:4+toklen]), 
				  x[:,4+toklen:]), dim=1)
		return x,q

model = ecTransformer(image_resolution = image_res, 
							 vision_width = vision_width, 
							 patch_size = patch_size, 
							 prog_width = prog_width, 
							 embed_dim = embed_dim, 
							 v_ctx = v_ctx, 
							 p_ctx = p_ctx, 
							 p_indim = p_indim, 
							 e_indim = e_indim)

from os.path import exists
if exists("ec32.ptx"):
	loaded_dict = torch.load("ec32.ptx")
	model.load_state_dict(loaded_dict)
	# prefix = 'module.'
	# n_clip = len(prefix)
	# adapted_dict = {k[n_clip:]: v for k, v in loaded_dict.items()
	# 					if k.startswith(prefix)}
# except: 
# 	print("could not load model parameters from ec32.ptx")

trainable_params = sum(
	p.numel() for p in model.parameters() if p.requires_grad
)
print(f"Number of model parameters:{trainable_params/1e6}M")

# model = nn.DataParallel(model)

# loss is on the predicted edit: 
# [0:4] is the categorical edit type, sub del ins fin
# [4:toklen] is the categorical character/token.  
# [5+toklen:5+toklen+poslen*2] is the (absolute) position encoding, vectoral.
# not sure why there is a 5 in there.

lossfunc_cel = nn.CrossEntropyLoss(label_smoothing = 0.08, reduction='mean')
lossfunc_mse = nn.MSELoss(reduction='mean')
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

def print_model_params(): 
	print(model.prt_to_tok.weight[0,:])
	print(model.prt.resblocks[0].mlp[0].weight[0,:])
	print(model.vit_to_prt.weight[0,1:20])
	print(model.vit.transformer.resblocks[0].mlp[0].weight[0,1:20])
	print(model.vit.conv1.weight[0,:])
	# it would seem that all the model parameters are changing.

def std_model_params(): 
	q = th.zeros(5)
	q[0] = th.std(model.vit.conv1.weight)
	q[1] = th.std(model.vit.transformer.resblocks[0].mlp[0].weight)
	q[2] = th.std(model.vit_to_prt.weight)
	q[3] = th.std(model.prt.resblocks[0].mlp[0].weight)
	q[4] = th.std(model.prt_to_tok.weight)
	return q
	
scaler = torch.cuda.amp.GradScaler()
slowloss = 1.0
losslog = open("loss_log.txt", "w")
lr = learning_rate
tic = time.time()
if g_training:
	print("training...")
if g_dreaming:
	print("dreaming...")

# compiling this does not seem to work... 
def train(mod, bimg, bpro, bedts): 
	model.zero_grad()
	y,q = model(u, bimg.cuda(), bpro.cuda())
	loss = lossfunc(y, bedts.cuda())
	lossflat = th.sum(loss)
	lossflat.backward()
	th.nn.utils.clip_grad_norm_(model.parameters(), 0.05)
	optimizer.step()
	return y,q
	
# train_opt = th.compile(train, mode="reduce-overhead")

for u in range(train_iters): 
	# keep things synchronous for now. 
	sock.sendall(b"update_batch")
	data = sock.recv(100) # faster? 
	
	bpro = read_mmap(fd_bpro, [batch_size, p_ctx, p_indim])
	bimg = read_mmap(fd_bimg, [batch_size, 3, image_res, image_res])
	bedts = read_mmap(fd_bedts, [batch_size, e_indim])
	
	if th.min(bedts[:,0]) < 0: 
		print("bedts synchronization issue!")
	
	# with th.autocast(device_type='cuda', dtype=torch.float16):
	# y,q = train_opt(model, bimg.cuda(), bpro.cuda(), bedts.cuda())
	model.zero_grad()
	y,q = model(u, bimg.cuda(), bpro.cuda())
	if g_training: 
		targ = bedts.cuda()
		loss = lossfunc_mse(y, targ)
		# loss_typ = lossfunc_cel(y[:,0:4], targ[:,0:4])
		# loss_chr = lossfunc_cel(y[:,4:4+toklen], targ[:,4:4+toklen])
		# loss_pos = lossfunc_mse(y[:,5+toklen:], targ[:,5+toklen:])
		# loss = loss_typ + loss_chr + loss_pos # should be batch_size
		lossflat = th.sum(loss)
		lossflat.backward()
		th.nn.utils.clip_grad_norm_(model.parameters(), 0.05)
		optimizer.step() 
		lossflat.detach()
	else: 
		lossflat = 0.0
		
	slowloss = 0.99*slowloss + 0.01 * lossflat
	# ngpu = th.cuda.device_count()
	# q = th.reshape(q, (ngpu,-1)) 
	# q = th.mean(q, 0) # only for model DP. 
	if g_training: 
		losslog.write(f"{u}\t{slowloss}")
		for i in range(q.shape[0]): 
			losslog.write(f"\t{q[i].cpu().item()}")
		losslog.write(f"\t{nreplace+0.001}")
		losslog.write("\n")
		losslog.flush()
	
	write_mmap(fd_bedtd, y)
	if g_training: 
		write_mmap(fd_editdiff, bedts - y.cpu()) # synchronization.
		sock.sendall(b"decode_edit")
		data = sock.recv(100)
	# scaler.scale(lossflat).backward()
	# th.nn.utils.clip_grad_norm_(model.parameters(), 0.05)
	# scaler.step(optimizer)
	# scaler.update()
	
	if u % 11 == 0 :
		toc = time.time()
		rate = int((batch_size * 11) / (toc - tic))
		tic = toc
		print(f'{u} {lr:.6f} loss: {lossflat:.5f}; slowloss {slowloss:.5f}; {rate} samp/sec')
	
	# change the learning rate. 
	if false: 
		lr = learning_rate
		# ramp up between 1000 and 11000
		if u > 1000:
			lr = lr * (1 + ((u-1000) / 5000))
		lr = min(lr, 0.001) # this seems to be the outright maximum
		# decay from 11k to end
		if u > 11000: 
			lr = lr * math.exp((11000-u) / 50000)
		for g in optimizer.param_groups:
			g['lr'] = lr
				
	if u % 1000 == 999 : 
		if g_training: 
			torch.save(model.state_dict(), "ec32.ptx")
			print("saved ec32.ptx")
		if g_dreaming: 
			loaded_dict = torch.load("ec32.ptx")
			model.load_state_dict(loaded_dict)
			print("dreamer reloaded model parameters.")
	


fd_bpro.close()
fd_bimg.close()
fd_bedts.close()
fd_bedtd.close()
fd_posenc.close()

sock.close()


