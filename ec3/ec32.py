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
import pdb

import torch._dynamo as dynamo
dynamo.config.verbose=True

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
learning_rate = 0.00125 # maximum learning rate. scheduled.
# learning rate of 0.002 is unstable.  Should figure out why. 
weight_decay = 5e-6
nreplace = 0


parser = argparse.ArgumentParser(description='Transformer-based program synthesizer')
parser.add_argument("-b", "--batch_size", help="Set the batch size", type=int)
args = parser.parse_args()
batch_size = args.batch_size
print(f"batch_size:{batch_size}")

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(('127.0.0.1', 4340))
sock.sendall(b"update_batch\n")
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
	
fd_bpro = make_mmf("bpro.mmap")
fd_bimg = make_mmf("bimg.mmap")
fd_bedt = make_mmf("bedt.mmap")
fd_posenc = make_mmf("posenc.mmap")
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
	prefix = 'module.'
	n_clip = len(prefix)
	adapted_dict = {k[n_clip:]: v for k, v in loaded_dict.items()
						if k.startswith(prefix)}
	model.load_state_dict(loaded_dict)
# except: 
# 	print("could not load model parameters from ec32.ptx")

# model = nn.DataParallel(model)

# lossfunc = nn.CrossEntropyLoss(label_smoothing = 0.08, reduction='none')
lossfunc = nn.MSELoss(reduction='mean')
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


def decode_edit(y): 
	typ = th.argmax(y[:,0:4], 1)
	typ = typ.cpu() # type should be th.Long
	bs = y.shape[0]
	ityp = []
	styp = []
	cl = []
	posl = [] # keep native - faster.
	for j in range(bs): 
		ityp.append(typ[j])
		if typ[j] == 0: 
			styp.append('sub')
		if typ[j] == 1: 
			styp.append('del')
		if typ[j] == 2: 
			styp.append('ins')
		if typ[j] == 3: 
			styp.append('fin')
		c = th.argmax(y[j,4:4+toklen]).cpu()
		c = chr(c + ord('0'))
		cl.append(c)
		pos = y[j,5+toklen:]
		z = y[j,5+toklen:]
		z = th.unsqueeze(z,0)
		z = z.expand(p_ctx, -1)
		cos = th.nn.CosineSimilarity(dim=1)
		pos = cos(posenc[:,:], z)
		posl.append(th.argmax(pos).item())
	return (ityp,styp, cl, posl) 

	
def compare_edit(batch_e, y): 
	ityp, styp, c, pos = decode_edit(batch_e)
	print("batch_e: ", styp[0]," ",c[0]," ",pos[0])
	ityp, styp, c, pos = decode_edit(y)
	print("model_y: ", styp[0]," ",c[0]," ",pos[0])
	
	
def hallucinate (): 
	# get a new batch, decode & apply edits. 
	sock.sendall(b"reset_batch:\n")
	data = sock.recv(100)
	
	done = th.zeros(batch_size)
	
	for i in range(16): 
		# range depends on the design spec -- e.g. max 8 edits.
		bpro = read_mmap(fd_bpro, [batch_size, p_ctx, p_indim])
		bimg = read_mmap(fd_bimg, [batch_size, 3, image_res, image_res])
		bedt = read_mmap(fd_bedt, [batch_size, e_indim])
	
		# with th.autocast(device_type='cuda', dtype=torch.float16):
		y,q = model(u, bimg.cuda(), bpro.cuda())
		
		ityp,styp,cl,posl = decode_edit(y.cpu())
		# apply edits in ocaml, to avoid code duplication. 
		# keep the messages small, < 4k. 
		b = bytearray(b"edit_types:")
		for i in range(batch_size): 
			ti = ityp[i]
			if done[i] < 1: 
				b.append(ti + ord('0'))
			else: 
				b.append(3 + ord('0')) # 'fin'
			if ti == 3: 
				done[i] = 2
		b.append(ord('\n'))
		sock.sendall(b)
		data = sock.recv(100)
		# print(f"edit_types received {data!r}")
		
		b = bytearray(b"edit_pos:")
		for pos in posl: 
			# same idea, pos irrelevant if 'fin'.
			b.append(pos + ord('0'))
		b.append(ord('\n'))
		sock.sendall(b)
		data = sock.recv(100)
		# print(f"edit_pos received {data!r}")
		
		b = bytearray(b"edit_chars:")
		for c in cl: 
			# doesn't matter if we inject c when the op is 'fin'
			b.append(ord(c))
		b.append(ord('\n'))
		sock.sendall(b)
		data = sock.recv(100)
		# print(f"edit_chars received {data!r}")
		
		sock.sendall(b"apply_edits:\n")
		data = sock.recv(100)
		# print(f"apply_edits received {data!r}")
		# synchronous; must also update the mmaped files.
		
	sock.sendall(b"print_progenc:\n")
	data = sock.recv(100)
	print("done with hallucination.")
	sock.sendall(b"reset_batch:\n")
	data = sock.recv(100)

scaler = torch.cuda.amp.GradScaler()
slowloss = 1.0
losslog = open("loss_log.txt", "w")
lr = learning_rate
tic = time.time()
print("training...")

# compiling this does not seem to work... 
def train(mod, bimg, bpro, bedt): 
	model.zero_grad()
	y,q = model(u, bimg.cuda(), bpro.cuda())
	loss = lossfunc(y, bedt.cuda())
	lossflat = th.sum(loss)
	lossflat.backward()
	th.nn.utils.clip_grad_norm_(model.parameters(), 0.05)
	optimizer.step()
	
# train_opt = th.compile(train, mode="reduce-overhead")

for u in range(train_iters): 
	# need to set the default tensor type to CPU
	# so we can read from the mmap files (obvi in CPU mem)
	# th.set_default_tensor_type('torch.FloatTensor')
	bpro = read_mmap(fd_bpro, [batch_size, p_ctx, p_indim])
	bimg = read_mmap(fd_bimg, [batch_size, 3, image_res, image_res])
	bedt = read_mmap(fd_bedt, [batch_size, e_indim])
	# set back to GPU 
	# th.set_default_tensor_type('torch.cuda.FloatTensor')
	
	# now that we have a copy, can ask ocaml to update async
	sock.sendall(b"update_batch\n")
	
	# with th.autocast(device_type='cuda', dtype=torch.float16):
	train_opt(model, bimg.cuda(), bpro.cuda(), bedt.cuda())
	# model.zero_grad()
	# y,q = model(u, bimg.cuda(), bpro.cuda())
	# loss = lossfunc(y, bedt.cuda())
	# lossflat = th.sum(loss)
	# lossflat.backward()
	# th.nn.utils.clip_grad_norm_(model.parameters(), 0.05)
	# optimizer.step()
		
	# scaler.scale(lossflat).backward()
	# th.nn.utils.clip_grad_norm_(model.parameters(), 0.05)
	# scaler.step(optimizer)
	# scaler.update()
	
	# do this later (async) to improve gpu utilization
	data = sock.recv(100)
	try:
		nreplace = int(data[3:].decode("utf-8"))
	except: 
		print("didn't get nreplace")
	
	slowloss = 0.99*slowloss + 0.01 * lossflat.detach()
	if u % 7 == 0 :
		toc = time.time()
		rate = int((batch_size * 7) / (toc - tic))
		tic = toc
		print(f'{u} {lr:.6f} loss: {lossflat:.5f}; slowloss {slowloss:.5f}; {rate} samp/sec')
		compare_edit(bedt, y.cpu())
		# print(i, lossflat, loss[0], y_mask[0])
		# print(x[0])
		# print(y[0])
		# print(x[0] - y[0])
	ngpu = th.cuda.device_count()
	q = th.reshape(q, (ngpu,-1))
	q = th.mean(q, 0)
	losslog.write(f"{u}\t{slowloss}")
	for i in range(q.shape[0]): 
		losslog.write(f"\t{q[i].cpu().item()}")
	losslog.write(f"\t{nreplace+0.001}")
	losslog.write("\n")
	losslog.flush()
	
	# change the learning rate. 
	lr = learning_rate
	# # ramp up between 1000 and 11000
	# if u > 1000:
	# 	lr = lr * (1 + ((u-1000) / 5000))
	# lr = min(lr, 0.001) # this seems to be the outright maximum
	# # decay from 11k to end
	# if u > 11000: 
	# 	lr = lr * math.exp((11000-u) / 50000)
	for g in optimizer.param_groups:
		g['lr'] = lr
		
	if u % 99 == 98 and u > 800: 
		hallucinate()
		if u > 4000: 
			hallucinate()
		if u > 6000: 
			hallucinate()
		if u > 8000: 
			hallucinate()
				
	if u % 1000 == 999 : 
		torch.save(model.state_dict(), "ec32.ptx")
		print("saved ec32.ptx")


fd_bpro.close()
fd_bimg.close()
fd_bedt.close()
fd_posenc.close()

sock.close()


