import math
from math import sin, cos, pi
import torch as th
from torch import nn, optim
import fcntl, os, select, sys
import pdb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import subprocess 

import logod_pb2
import xf
import clip_model

torch_device = 0
print("torch cuda devices", th.cuda.device_count())
print("torch device", th.cuda.get_device_name(torch_device))
th.cuda.set_device(torch_device)
th.set_default_tensor_type('torch.cuda.FloatTensor')


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
image_resolution = 30

def ocaml_run(prog_str):
	global g_lp_count
	lp = logod_pb2.Logo_program()
	lp.id = g_lp_count
	g_lp_count = g_lp_count + 1
	lp.prog = prog_str
	lp.resolution = image_resolution
	lp.log_en = False # logging off = 14.2 sec; on = 23.6 sec
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
		# print(result)
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
	if n != result.stride * result.height: 
		# print("unexpected number of bytes in image, ", n)
		return None
	else: 
		# do something else with this image..
		a = np.frombuffer(buff2, dtype=np.uint8, count=n)
		a = np.reshape(a, (result.height, result.stride))
		a = a[:, 0:result.width]
		# plt.imshow(a)
		# plt.colorbar()
		# plt.show()
		return a

ocaml_run(square_prog)
ocaml_run("( move 1 1 )")


tokens = [" ", "(", ")",";","+ ","- ","* ","/ ", 
			 "move ","loop ","v","ua ","ul ", 
			 "0 ","1 ","2 ",
			 "eof"] 

tktype = [0, 1, 1, 2, 3, 3, 3, 3, 
			 4, 5, 6, 7, 7, 
			 8, 8, 8,
			 9] # convert this to a one-hot as well
			# better for MLP encoder
			
		
toklen = len(tokens)
typlen = 10
poslen = 6
p_indim = toklen + typlen + poslen*2
p_ctx = 15
patch_size = 5
v_ctx = int((image_resolution / patch_size) ** 2 + 1)
batch_size = 32
prog_width = 128
xfrmr = xf.Transformer(p_ctx, prog_width, 8, 8)
	# p_ctx, width, layers, heads
encoder = nn.Linear(p_indim, prog_width)
	# just a simple linear layer to put into transformer latent space. 
	# how are these weights initialized?? 
gelu = nn.GELU()

# positional encoding. 
posenc = th.zeros(p_ctx, poslen*2)
for i in range(p_ctx): 
	for j in range(poslen):
		posenc[i,j*2+0] = sin((2*pi*i / p_ctx) * (j+1))
		posenc[i,j*2+1] = cos((2*pi*i / p_ctx) * (j+1))
		
posenc = posenc.expand(batch_size, p_ctx, poslen*2) 

prog = [0 for i in range(p_ctx)]
prog_human = list(map(lambda i : tokens[i], prog))
x = th.zeros(p_ctx, p_indim)
for i in range(p_ctx): # might be faster way? 
	x[i, prog[i]] = 1.0
	typ = tktype[prog[i]]
	x[i, toklen+typ] = 1.0
	# add positional encoding too. 

x = x.expand(batch_size, p_ctx, p_indim).contiguous()
x[:,:,toklen + typlen : p_indim] = posenc
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
	
def prog_to_human(prog): 
	return list(map(lambda i : tokens[i], prog))

def enumerate_programs(n_levels, level, prog): 
	if level == n_levels: 
		a = check_formed(prog)
		if a is not None: 
			# logf.write(prog_to_string(prog))
			# logf.write("\n")
			return (prog, a)
		else:
			return None
	lst = []
	for i in range(toklen - 1): 
		q = enumerate_programs(n_levels, level+1, prog + [i])
		if q is not None: 
			if type(q) is tuple: # leaf call
				lst.append(q)
			if type(q) is list: 
				lst = lst + q # list of (list, np_array). 
	if len(lst) > 0: 
		return lst
	else : 
		return None

valid = enumerate_programs(4, 0, [])
print(valid)
for item in valid: 
	prog, a = item
	print(prog)
print(len(valid))

# how many of these are actually interesting? 
def nonzero_img(b): 
	prog,a = b
	return np.sum(a) > 0
valid_nonzero = list(filter(nonzero_img, valid))
vnf = open("valid_nonzero/list.txt", "w")
for i, item in enumerate(valid_nonzero): 
	prog, a = item
	prog_human = prog_to_human(prog)
	vnf.write(f"{i} : {prog} : {prog_human}")
	vnf.write("\n")
	matplotlib.image.imsave(f'valid_nonzero/out{i}.png', a)
vnf.close()
print(f'number of valid non-zero programs {len(valid_nonzero)}')

def build_attention_mask2(v_ctx, p_ctx):
    # lazily create causal attention mask, with full attention between the vision tokens
    # pytorch uses additive attention mask; fill with -inf
    ctx = v_ctx + p_ctx
    mask = th.empty(ctx, ctx)
    mask.fill_(float("-inf"))
    mask.triu_(1)  # zero out the lower diagonal
    mask[0:v_ctx, 0:v_ctx] = 1.0
    return mask

# alrighty then!  time to train the transformer to create these strings! 
# decoder-only architecture, input is the image (compressed) and previous tokens. 
vision_width = 256
vision_heads = 8
vision_layers = 4
prog_heads = 8
prog_layers = 6
embed_dim = 256

class ecTransformer(nn.Module):
	def __init__(self, image_resolution: int, vision_width:int, patch_size:int,  prog_width:int, embed_dim:int, v_ctx:int, p_ctx:int, p_indim:int): 
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
			attn_mask = build_attention_mask2(v_ctx, p_ctx))
			
		self.prt_to_tok = nn.Linear(prog_width * (p_ctx + v_ctx), toklen + typlen)
		self.ln_post = clip_model.LayerNorm(toklen + typlen)
		self.gelu = clip_model.QuickGELU()
		self.tok_softmax = nn.Softmax(dim = 1)
	
	def forward(self, batch_a, batch_p, mask): 
		# encode the image (we should only need to do this once??)
		vx = self.vit(batch_a) # x is size [16, v_ctx, 256] 
		vx = self.vit_to_prt(vx)
		# vx = gelu(vx) # ? needed ? 

		px = self.encoder(batch_p)
		vxpx = th.cat((vx, px), dim = 1)
		x = vxpx * mask
		x = self.prt(x) # bs, v_ctx * p_ctx, prog_width
		x = th.reshape(x, (batch_size,-1))
		x = self.prt_to_tok(x)
		x = self.ln_post(x) # scale the inputs to softmax
		x = self.gelu(x)
		# x = self.tok_softmax(x)
		return x

model = ecTransformer(image_resolution = image_resolution, 
							 vision_width = vision_width, 
							 patch_size = patch_size, 
							 prog_width = prog_width, 
							 embed_dim = embed_dim, 
							 v_ctx = v_ctx, 
							 p_ctx = p_ctx, 
							 p_indim = p_indim)

lossfunc = nn.CrossEntropyLoss(label_smoothing = 0.08, reduction='none')
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)

def print_model_params(): 
	print(model.prt_to_tok.weight[0,:])
	print(model.prt.resblocks[0].mlp[0].weight[0,:])
	print(model.vit_to_prt.weight[0,1:20])
	print(model.vit.transformer.resblocks[0].mlp[0].weight[0,1:20])
	print(model.vit.conv1.weight[0,:])
	# it would seem that all the model parameters are changing.

def make_batch(): 
	indx = np.random.choice(len(valid_nonzero), size=batch_size, replace=False)
	batch_a = th.zeros((batch_size, image_resolution, image_resolution)) # todo: fp16
	batch_p = th.zeros((batch_size, p_ctx, p_indim))
	batch_pl = np.zeros(batch_size)
	j = 0
	for i in indx: 
		prog, a = valid_nonzero[i] # note! this is a reference. 
		batch_a[j,:,:] = th.tensor(a)
		# encode the program, too
		prog = prog + [toklen-1] # terminate; return new list.
		batch_pl[j] = len(prog)
		# prog_human = list(map(lambda i : tokens[i], prog))
		for k in range(len(prog)): 
			batch_p[j, k, prog[k]] = 1.0
			typ = tktype[prog[k]]
			batch_p[j, k, toklen+typ] = 1.0
		j = j+1
		
	batch_p[:,:,toklen+typlen : p_indim] = posenc
	batch_a = th.unsqueeze(batch_a, dim = 1) # batch, channels, H,
	return (batch_a, batch_p, batch_pl)

def make_mask_y(i, batch_p, batch_pl): 
	y = batch_p[:, i, 0:toklen + typlen]
	y_mask = th.tensor(batch_pl > i).detach() 
	mask = th.zeros(batch_size, v_ctx + p_ctx, prog_width)
	mask[:, 0:v_ctx, :] = 1.0
	if i > 1: 
		mask[:, v_ctx:v_ctx+i, :] = 1.0
	return (mask, y, y_mask)

slowloss = 0.0

for u in range(10000): 
	batch_a, batch_p, batch_pl = make_batch()

	for i in range(int(np.max(batch_pl))): 
		mask, y, y_mask = make_mask_y(i, batch_p, batch_pl)
		
		model.zero_grad()
		x = model(batch_a, batch_p, mask)
		loss = lossfunc(x,y)
		lossflat = th.sum(loss * y_mask)
		lossflat.backward()
		optimizer.step()
		slowloss = 0.99*slowloss + 0.01 * lossflat.detach()
		if u % 20 == 0 :
			print(f'{i} loss: {lossflat}; slowloss {slowloss}')
			# print(i, lossflat, loss[0], y_mask[0])
			# print(x[0])
			# print(y[0])
			# print(x[0] - y[0])

# see if it can create working programs
softmx = nn.Softmax(dim = 1)
batch_a, batch_p, batch_pl = make_batch()
batch_p_orig = batch_p.clone()
for i in range(int(np.max(batch_pl))):
	mask, y, y_mask = make_mask_y(i, batch_p, batch_pl)
	x = model(batch_a, batch_p, mask)
	# substitute this into batch_p
	x = softmx(x[:, 0:toklen])
	indx = th.argmax(x, dim=1)
	batch_p[:, i] = th.zeros(p_indim)
	batch_p[:, i, indx] = 1.0
	for j in range(batch_size): 
		typ = tktype[indx[j]]
		batch_p[j, i, toklen+typ] = 1.0

# now process it back to human-readable.
for j in range(batch_size): 
	prog_orig = []
	prog_new = []
	for i in range(int(np.max(batch_pl))):
		prog_orig.append(th.argmax(batch_p_orig[j, i, 0:toklen]))
		prog_new.append(th.argmax(batch_p[j, i, 0:toklen]))
	prog_orig_h = prog_to_human(prog_orig)
	prog_new_h = prog_to_human(prog_new)
	print(f"batch{j}\n\torig {prog_orig_h}\n\tnew {prog_new_h}\n")

logf.close()
sp.terminate()
sys.exit(0)
